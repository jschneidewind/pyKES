"""
One entry: its metadata, its place in the reference graph, and its traces.

Works for every kind of entry. An experiment shows its measured curves; a
precursor chemical has none and shows what was made from it instead — which is
the question the reference graph exists to answer.
"""

import json

import streamlit as st

from pyKES.database.index_ingest import may_edit, update_entity_metadata
from pyKES.database.index_query import (
    display_entity_type,
    display_role_path,
    read_entity,
    read_neighbours,
    read_versions,
    search_entity_ids,
)
from pyKES.database.index_registry import split_qualified_key
from pyKES.database_app.components.time_series_panel import render_time_series_panel
from pyKES.database_app.config import DEFAULT_CONFIG, DatabaseAppConfig
from pyKES.database_app.session import index_paths, open_shared_index, read_identity


# =============================================================================
# Display constants
# =============================================================================

SELECTED_ENTITY_KEY = "selected_entity_id"

# Query parameter carrying the entity, so a detail page is a shareable link.
ENTITY_PARAMETER = "entity"

# Shown where a metadata value is absent.
MISSING_PLACEHOLDER = "—"


# =============================================================================
# Metadata display
# =============================================================================

def split_metadata(effective: dict, own: dict) -> tuple:
    """
    Separate an entry's own metadata from what it inherited.

    Parameters
    ----------
    effective : dict
        Own metadata plus everything inherited, qualified.
    own : dict
        The entry's own metadata.

    Returns
    -------
    own_rows, inherited_rows : list of dict
        Table rows, the inherited ones carrying the reference path they came
        through so the provenance of every value is visible.
    """
    own_rows, inherited_rows = [], []

    for key, value in sorted(effective.items()):
        display = MISSING_PLACEHOLDER if value is None else value
        role_path, leaf = split_qualified_key(key)

        if key in own:
            own_rows.append({"Field": leaf, "Value": display})
        else:
            inherited_rows.append({"Field": leaf, "Value": display,
                                   "Via": display_role_path(role_path)})

    return own_rows, inherited_rows


def render_metadata(row) -> None:
    """
    Draw the entry's own and inherited metadata as two tables.

    Parameters
    ----------
    row : sqlite3.Row
        Entity row.

    Returns
    -------
    None : None
    """
    import pandas as pd

    own = json.loads(row["metadata"])
    effective = json.loads(row["effective"])
    own_rows, inherited_rows = split_metadata(effective, own)

    with st.expander(f"Metadata ({len(own_rows)})", expanded=False):
        st.dataframe(pd.DataFrame(own_rows), width="stretch", hide_index=True)

    if inherited_rows:
        with st.expander(f"Inherited Through References ({len(inherited_rows)})",
                         expanded=False):
            st.caption("Each value carries the reference path it came through.")
            st.dataframe(pd.DataFrame(inherited_rows), width="stretch",
                         hide_index=True)


def render_results(row) -> None:
    """
    Draw the entry's mapped scalar results.

    Parameters
    ----------
    row : sqlite3.Row
        Entity row.

    Returns
    -------
    None : None
    """
    import pandas as pd

    results = json.loads(row["results"])
    if not results:
        return

    with st.expander(f"Results ({len(results)})", expanded=False):
        st.dataframe(pd.DataFrame([{"Result": key, "Value": value}
                                   for key, value in sorted(results.items())]),
                     width="stretch", hide_index=True)


# =============================================================================
# The reference graph around one entry
# =============================================================================

def render_neighbours(connection, entity_id: str) -> None:
    """
    Draw what this entry references and what references it.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entry in question.

    Returns
    -------
    None : None
    """
    neighbours = read_neighbours(connection, entity_id)
    left, right = st.columns(2)

    with left:
        st.subheader("References")
        if not neighbours["references"]:
            st.caption("This entry references nothing.")
        for edge in neighbours["references"]:
            state = "" if edge["resolved"] else "  ·  not uploaded yet"
            if st.button(f"{display_entity_type(edge['role'])} → "
                         f"{edge['entity_id']}{state}",
                         key=f"ref_{edge['role']}", width="stretch"):
                open_entity(edge["entity_id"])

    with right:
        st.subheader("Referenced by")
        if not neighbours["referenced_by"]:
            st.caption("Nothing references this entry.")
        for edge in neighbours["referenced_by"][:25]:
            if st.button(f"{edge['entity_id']}  "
                         f"({display_entity_type(edge['entity_type'])})",
                         key=f"back_{edge['entity_id']}", width="stretch"):
                open_entity(edge["entity_id"])

        if len(neighbours["referenced_by"]) > 25:
            st.caption(f"…and {len(neighbours['referenced_by']) - 25} more.")


def open_entity(entity_id: str) -> None:
    """
    Navigate to another entry.

    Parameters
    ----------
    entity_id : str
        Entry to open.

    Returns
    -------
    None : None
    """
    st.session_state[SELECTED_ENTITY_KEY] = entity_id
    st.query_params[ENTITY_PARAMETER] = entity_id
    st.rerun()


# =============================================================================
# Traces
# =============================================================================

def render_payload(row, config: DatabaseAppConfig) -> None:
    """
    Plot this entry's traces, if it has any, and offer the payload.

    The same panel the browse page uses for comparison, with the entry
    selection switched off: comparing entries is done on the browse page, where
    the filters decide the subset.

    Parameters
    ----------
    row : sqlite3.Row
        Entity row.
    config : DatabaseAppConfig
        Deployment settings, supplying the payload directory.

    Returns
    -------
    None : None
    """
    if not row["payload_path"]:
        return

    payload_file = index_paths(config).payload_directory / row["payload_path"]
    if not payload_file.exists():
        st.error(f"Payload {row['payload_path']} is missing from the store.")
        return

    st.subheader("Traces")
    render_time_series_panel((str(payload_file),), key_prefix="entity",
                             allow_deselection=False)

    st.download_button("Download This Entry as HDF5",
                       data=payload_file.read_bytes(),
                       file_name=row["payload_path"],
                       mime="application/x-hdf")


# =============================================================================
# Editing
# =============================================================================

def render_editor(connection, row, identity, config: DatabaseAppConfig) -> None:
    """
    Let the owner or an admin correct this entry's metadata.

    Editing an entry that others reference changes their effective metadata too,
    so the number of entries the edit moved is reported back rather than left
    implicit.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    row : sqlite3.Row
        Entity being edited.
    identity : Identity
        Current user.
    config : DatabaseAppConfig
        Deployment settings supplying the reference declaration.

    Returns
    -------
    None : None
    """
    permitted = may_edit(connection, row["entity_id"], identity.name,
                         identity.is_admin)

    with st.expander("Correct This Entry", expanded=False):
        if not permitted:
            st.info(f"Owned by **{row['owner']}**. Only the owner or an admin "
                    f"may correct it.")
            return

        own = json.loads(row["metadata"])
        field_name = st.selectbox("Field", sorted(own) + ["(new field)"])

        if field_name == "(new field)":
            field_name = st.text_input("New field name")
            current = ""
        else:
            current = "" if own.get(field_name) is None else str(own[field_name])

        new_value = st.text_input("Value", value=current)

        if st.button("Save correction", type="primary") and field_name:
            recomputed = update_entity_metadata(
                connection, row["entity_id"], {field_name: coerce_typed(new_value)},
                identity.name, identity.is_admin,
                config.reference_instructions_by_type.get(row["entity_type"], {}))
            st.success(f"Saved. {len(recomputed)} entries recomputed "
                       f"({', '.join(recomputed[:5])}…).")
            st.cache_data.clear()
            st.rerun()


def coerce_typed(text: str):
    """
    Interpret a typed correction as a number, a boolean or text.

    Parameters
    ----------
    text : str
        Value as typed.

    Returns
    -------
    value : Any
        Parsed value; an empty string becomes None, which removes the field.
    """
    stripped = text.strip()

    if not stripped:
        return None

    if stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        return stripped


def render_entity_picker(connection, entity_id: str) -> str:
    """
    Offer matching entries as the user types part of an id.

    Typing ``NB-6`` should present every entry that starts that way rather than
    requiring the whole id to be remembered and spelled correctly.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entry currently open, shown as the field's initial content.

    Returns
    -------
    chosen : str
        The entry the user picked, or an empty string if they have not picked
        one yet.
    """
    typed = st.text_input("Find an Entry", value=entity_id or "",
                          placeholder="Start typing an ID, e.g. NB-6",
                          key="entity_search")

    if not typed or typed == entity_id:
        return ""

    matches = search_entity_ids(connection, typed)

    if not matches:
        st.caption(f"No entry matches '{typed}'.")
        return ""

    # A single exact match needs no further choosing.
    if matches == [typed]:
        return typed

    return st.selectbox(f"{len(matches)} matching entries", matches,
                        index=None, placeholder="Select an entry…",
                        key="entity_matches") or ""


# =============================================================================
# Entry point
# =============================================================================

def render_entity(config: DatabaseAppConfig = DEFAULT_CONFIG) -> None:
    """
    Render the entry detail page.

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

    entity_id = (st.query_params.get(ENTITY_PARAMETER)
                 or st.session_state.get(SELECTED_ENTITY_KEY))

    chosen = render_entity_picker(connection, entity_id)
    if chosen and chosen != entity_id:
        open_entity(chosen)

    if not entity_id:
        st.info("Search for an entry above, or choose one from Browse & Search.")
        return

    row = read_entity(connection, entity_id)
    if row is None:
        st.error(f"No entry called '{entity_id}'.")
        return

    st.title(row["entity_id"])
    st.caption(f"{display_entity_type(row['entity_type'])}  ·  "
               f"Group: {row['display_group'] or '—'}  ·  "
               f"Owner: {row['owner']}  ·  Updated: {row['updated_at'][:19]}")

    versions = read_versions(connection, row["base_id"])
    if len(versions) > 1:
        st.warning(f"{len(versions)} versions of this name exist: "
                   f"{', '.join(version['entity_id'] for version in versions)}. "
                   f"Both were kept because uploading never overwrites.")

    render_results(row)
    render_neighbours(connection, row["entity_id"])
    render_metadata(row)
    render_payload(row, config)
    render_editor(connection, row, identity, config)
