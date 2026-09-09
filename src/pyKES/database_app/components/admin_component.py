"""
The admin view: the things a growing free-form schema needs somebody to look at.

Key drift and dangling references cannot be resolved automatically without
guessing, and guessing would corrupt searches in a way nobody would notice. So
they are surfaced here instead, where a person can decide.
"""

import streamlit as st

from pyKES.database.entity_schema import accepted_types, load_entity_schemas
from pyKES.database.index_query import (
    database_statistics,
    display_entity_type,
    read_uploads,
)
from pyKES.database.index_references import (
    find_cyclic_entities,
    read_dangling_references,
    read_reference_type_mismatches,
)
from pyKES.database.index_registry import (
    TYPE_MIXED,
    read_metadata_keys,
    rebuild_metadata_key_registry,
)
from pyKES.database.index_schema import analyse_index
from pyKES.database_app.config import DEFAULT_CONFIG, DatabaseAppConfig
from pyKES.database_app.deployment import render_version_caption
from pyKES.database_app.session import open_shared_index, read_identity


# =============================================================================
# Display constants
# =============================================================================

# Keys sharing a leaf name but written differently are the drift signal; two or
# more spellings of one field is what an admin needs to see.
DRIFT_MINIMUM_SPELLINGS = 2


def render_key_registry(connection) -> None:
    """
    Show every registered metadata key, and flag the two failure modes.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    None : None
    """
    import pandas as pd

    rows = read_metadata_keys(connection)
    if not rows:
        st.info("No metadata registered yet.")
        return

    frame = pd.DataFrame([{
        "Key": row["key"], "Leaf": row["leaf_name"],
        "Via": row["role_path"] or "—", "Type": row["inferred_type"],
        "Seen": row["occurrences"], "Entity types": row["entity_types"],
    } for row in rows])

    mixed = frame[frame["Type"] == TYPE_MIXED]
    if not mixed.empty:
        st.error(f"{len(mixed)} keys have been seen carrying more than one type. "
                 f"Their facets fall back to a text filter rather than drawing a "
                 f"broken slider.")
        st.dataframe(mixed, width="stretch", hide_index=True)

    # Drift: several distinct spellings that differ only in punctuation or case.
    normalised = {}
    for row in rows:
        if row["role_path"]:
            continue
        signature = "".join(character.lower() for character in row["leaf_name"]
                            if character.isalnum())
        normalised.setdefault(signature, []).append(row["key"])

    drifted = {signature: keys for signature, keys in normalised.items()
               if len(keys) >= DRIFT_MINIMUM_SPELLINGS}
    if drifted:
        st.warning(f"{len(drifted)} field names look like spellings of the same "
                   f"thing. An admin should alias one onto the other.")
        for keys in list(drifted.values())[:10]:
            st.write(" · ".join(f"`{key}`" for key in keys))

    with st.expander(f"All registered keys ({len(frame)})"):
        st.dataframe(frame, width="stretch", hide_index=True)


def render_reference_health(connection, config: DatabaseAppConfig) -> None:
    """
    Show references that point nowhere, chains that loop, and wrong-kind links.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    config : DatabaseAppConfig
        Deployment settings, supplying the schemas the accepted kinds come from.

    Returns
    -------
    None : None
    """
    import pandas as pd

    st.subheader("Reference Health")

    dangling = read_dangling_references(connection)
    if dangling:
        st.warning(f"{len(dangling)} references point at entries that are not in "
                   f"the database. A forward reference that never arrives looks "
                   f"exactly like a typo, so these need a person.")
        st.dataframe(pd.DataFrame([dict(row) for row in dangling]),
                     width="stretch", hide_index=True)
    else:
        st.success("Every reference resolves.")

    mismatches = read_reference_type_mismatches(
        connection, accepted_types(load_entity_schemas(config.schema_directory)))
    if mismatches:
        count = len(mismatches)
        st.warning(
            f"{count} reference{'s' if count > 1 else ''} "
            f"point{'' if count > 1 else 's'} at a kind of entry the field does "
            f"not accept. The metadata still merges — a wrong-kind reference is "
            f"a labelling mistake, and discarding the values would hide it "
            f"rather than show it — but the link is probably not what was meant."
        )
        st.dataframe(pd.DataFrame([
            {"Entry": row["source"], "Role": display_entity_type(row["role"]),
             "Points at": row["target"],
             "Which is a": display_entity_type(row["target_type"]),
             "Accepts": ", ".join(display_entity_type(kind)
                                  for kind in row["accepts"])}
            for row in mismatches]), width="stretch", hide_index=True)

    cycles = find_cyclic_entities(connection)
    if cycles:
        st.error(f"These entries are part of a reference cycle: "
                 f"{', '.join(cycles)}. Resolution stops at the cycle rather "
                 f"than looping, but the chain is wrong.")


def render_result_conflicts(connection) -> None:
    """
    Show result labels that two uploads defined differently.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    None : None
    """
    import pandas as pd

    rows = connection.execute(
        "SELECT label, path, conflicting FROM result_keys ORDER BY label"
    ).fetchall()
    if not rows:
        return

    frame = pd.DataFrame([dict(row) for row in rows])
    conflicting = frame[frame["conflicting"] == 1]

    st.subheader("Result Definitions")
    if not conflicting.empty:
        st.error(f"{len(conflicting)} labels were redefined with a different "
                 f"path. Both definitions are kept, because adopting the new "
                 f"one silently would change the meaning of a column for every "
                 f"entry already stored.")
    st.dataframe(frame, width="stretch", hide_index=True)


def render_uploads(connection) -> None:
    """
    Show the upload log.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    None : None
    """
    import pandas as pd

    rows = read_uploads(connection)
    if not rows:
        return

    st.subheader("Uploads")
    st.caption("Every file is kept verbatim, which is what makes the database "
               "rebuildable from scratch.")
    st.dataframe(pd.DataFrame([{
        "ID": row["id"], "File": row["filename"], "Kind": row["kind"],
        "Entries": row["entity_count"], "Uploaded by": row["uploaded_by"],
        "Uploaded at": row["uploaded_at"][:19],
        "Size (MB)": round(row["byte_count"] / 1e6, 2),
    } for row in rows]), width="stretch", hide_index=True)


def render_maintenance(connection, identity) -> None:
    """
    Offer the two maintenance actions that are safe to expose.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    identity : Identity
        Current user; only admins see this section.

    Returns
    -------
    None : None
    """
    if not identity.is_admin:
        return

    st.subheader("Maintenance")
    left, right = st.columns(2)

    if left.button("Rebuild Key Registry", width="stretch"):
        count = rebuild_metadata_key_registry(connection)
        st.success(f"Registry rebuilt from the entities table: {count} keys.")

    if right.button("Refresh Query Statistics (ANALYZE)", width="stretch"):
        analyse_index(connection)
        st.success("Statistics refreshed. This is what keeps the planner from "
                   "choosing a low-cardinality index over a selective one.")


def render_admin(config: DatabaseAppConfig = DEFAULT_CONFIG) -> None:
    """
    Render the admin page.

    Parameters
    ----------
    config : DatabaseAppConfig, optional
        Deployment settings.

    Returns
    -------
    None : None
    """
    connection = open_shared_index(str(config.data_root))
    identity = read_identity(config)

    st.title("Admin")
    render_version_caption(connection, config)

    statistics = database_statistics(connection)
    columns = st.columns(4)
    columns[0].metric("Entries", statistics["entities"])
    columns[1].metric("References", statistics["edges"])
    columns[2].metric("Metadata Keys", statistics["metadata_keys"])
    columns[3].metric("Uploads", statistics["uploads"])

    render_reference_health(connection, config)
    render_result_conflicts(connection)
    st.subheader("Metadata Keys")
    render_key_registry(connection)
    render_uploads(connection)
    render_maintenance(connection, identity)
