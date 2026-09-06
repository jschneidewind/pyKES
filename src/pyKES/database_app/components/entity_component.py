"""
One entry: its metadata, its place in the reference graph, and its traces.

Works for every kind of entry. An experiment shows its measured curves; a
precursor chemical has none and shows what was made from it instead — which is
the question the reference graph exists to answer.
"""

import json

import streamlit as st

from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.database.index_ingest import may_edit, update_entity_metadata
from pyKES.database.index_query import read_entity, read_neighbours, read_versions
from pyKES.database.index_schema import ROLE_PATH_SEPARATOR
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

        if key in own:
            own_rows.append({"Field": key, "Value": display})
        else:
            path, _, leaf = key.rpartition(ROLE_PATH_SEPARATOR)
            inherited_rows.append({"Field": leaf, "Value": display, "Via": path})

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

    st.subheader("Metadata")
    st.dataframe(pd.DataFrame(own_rows), width="stretch", hide_index=True)

    if inherited_rows:
        st.subheader(f"Inherited through references ({len(inherited_rows)})")
        st.caption("Each value carries the reference path it came through.")
        st.dataframe(pd.DataFrame(inherited_rows), width="stretch", hide_index=True)


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

    st.subheader("Results")
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
            if st.button(f"{edge['role']} → {edge['entity_id']}{state}",
                         key=f"ref_{edge['role']}", width="stretch"):
                open_entity(edge["entity_id"])

    with right:
        st.subheader("Referenced by")
        if not neighbours["referenced_by"]:
            st.caption("Nothing references this entry.")
        for edge in neighbours["referenced_by"][:25]:
            if st.button(f"{edge['entity_id']}  ({edge['entity_type']})",
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
    Plot the entry's measured and processed traces, if it has any.

    The payload is a standalone pyKES dataset, so it is handed to the existing
    time-series machinery unchanged and can also be downloaded and opened
    locally.

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
    dataset = load_payload(str(payload_file))
    experiment = next(iter(dataset.experiments.values()))

    series = {name: value for name, value in
              {**experiment.raw_data, **experiment.processed_data}.items()
              if hasattr(value, "shape") and getattr(value, "ndim", 0) == 1
              and value.size > 1}

    if not series:
        st.caption("This payload holds no plottable series.")
    else:
        names = sorted(series)
        x_name = st.selectbox("x", names,
                              index=next((i for i, n in enumerate(names)
                                          if "time" in n.lower()), 0))
        y_names = st.multiselect("y", [n for n in names if n != x_name],
                                 default=[n for n in names if n != x_name][:2])
        if y_names:
            render_figure(series, x_name, y_names)

    st.download_button("Download this entry as HDF5",
                       data=payload_file.read_bytes(),
                       file_name=row["payload_path"],
                       mime="application/x-hdf")


@st.cache_data(show_spinner=False)
def load_payload(payload_file: str):
    """
    Load one payload, cached across reruns.

    Parameters
    ----------
    payload_file : str
        Absolute path to the payload.

    Returns
    -------
    dataset : ExperimentalDataset
        The single-experiment dataset stored there.
    """
    return ExperimentalDataset.load_from_hdf5(payload_file)


def render_figure(series: dict, x_name: str, y_names: list) -> None:
    """
    Draw the selected series.

    Parameters
    ----------
    series : dict
        Named one-dimensional arrays.
    x_name : str
        Series used as the x axis.
    y_names : list of str
        Series to plot against it.

    Returns
    -------
    None : None
    """
    import plotly.graph_objects as go

    figure = go.Figure()
    x_values = series[x_name]

    for name in y_names:
        y_values = series[name]
        length = min(len(x_values), len(y_values))
        figure.add_trace(go.Scatter(x=x_values[:length], y=y_values[:length],
                                    mode="lines", name=name))

    figure.update_layout(xaxis_title=x_name, height=420,
                         margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figure, width="stretch")


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

    with st.expander("Correct this entry"):
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

    typed = st.text_input("Entry id", value=entity_id or "")
    if typed and typed != entity_id:
        open_entity(typed)

    if not entity_id:
        st.info("Choose an entry from Browse & Search, or type an id above.")
        return

    row = read_entity(connection, entity_id)
    if row is None:
        st.error(f"No entry called '{entity_id}'.")
        return

    st.title(row["entity_id"])
    st.caption(f"{row['entity_type']}  ·  group {row['display_group'] or '—'}  "
               f"·  owner {row['owner']}  ·  updated {row['updated_at'][:19]}")

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
