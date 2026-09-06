"""
Browse and search the database.

Every filter here is generated from the metadata-key registry rather than
configured by hand, so a spreadsheet column that first appears in next year's
uploads gets a working filter as soon as it has been ingested. Filters are
encoded into the URL, which makes a search something you can paste into a group
chat.
"""

import json

import streamlit as st

from pyKES.database.index_query import (
    Facet,
    Filter,
    RESULT_PREFIX,
    build_facets,
    display_key,
    rows_to_frame,
    search_entities,
)
from pyKES.database.index_schema import ENTITY_TYPES
from pyKES.database_app.config import DEFAULT_CONFIG, DatabaseAppConfig
from pyKES.database_app.session import open_shared_index


# =============================================================================
# Display constants
# =============================================================================

# Session-state key holding the entity the user opened from the results table.
SELECTED_ENTITY_KEY = "selected_entity_id"

# Query-parameter names. Kept short because they end up in a pasted URL.
TEXT_PARAMETER = "q"
TYPE_PARAMETER = "type"
FILTER_PARAMETER = "f"

# How many facets to draw before hiding the rest behind an expander. A chain
# four deep can register a hundred keys, and a sidebar of a hundred widgets is
# not a filter panel.
VISIBLE_FACET_LIMIT = 8


# =============================================================================
# URL state
# =============================================================================

def read_filters_from_url() -> list:
    """
    Decode the filter state carried in the query string.

    Returns
    -------
    filters : list of Filter
        Filters encoded in the URL, or an empty list when it carries none or
        carries something unreadable — a mangled link should open the unfiltered
        page rather than an error.
    """
    encoded = st.query_params.get(FILTER_PARAMETER)
    if not encoded:
        return []

    try:
        return [Filter(**entry) for entry in json.loads(encoded)]
    except (ValueError, TypeError):
        return []


def write_filters_to_url(filters: list, text: str, entity_type: str) -> None:
    """
    Encode the current search into the query string.

    Parameters
    ----------
    filters : list of Filter
        Active filters.
    text : str
        Free-text term.
    entity_type : str
        Kind of entry being searched.

    Returns
    -------
    None : None
    """
    st.query_params[TYPE_PARAMETER] = entity_type

    if text:
        st.query_params[TEXT_PARAMETER] = text
    elif TEXT_PARAMETER in st.query_params:
        del st.query_params[TEXT_PARAMETER]

    if filters:
        st.query_params[FILTER_PARAMETER] = json.dumps(
            [{"key": item.key, "operator": item.operator, "value": item.value}
             for item in filters])
    elif FILTER_PARAMETER in st.query_params:
        del st.query_params[FILTER_PARAMETER]


# =============================================================================
# Facet widgets
# =============================================================================

def render_facet(facet: Facet, position: int):
    """
    Draw one filter widget and return the filter it produces.

    Parameters
    ----------
    facet : Facet
        Facet definition from the registry.
    position : int
        Index used to build a unique widget key.

    Returns
    -------
    search_filter : Filter or None
        The filter, or None when the widget is at its neutral setting.
    """
    caption = f"{facet.label}"
    if facet.role_path:
        caption = f"{facet.label}  ·  via {facet.role_path}"

    if facet.kind == "range":
        low, high = facet.bounds
        chosen = st.slider(caption, float(low), float(high),
                           (float(low), float(high)), key=f"facet_{position}")
        if chosen != (float(low), float(high)):
            return Filter(facet.key, "between", list(chosen))
        return None

    if facet.kind == "select":
        chosen = st.multiselect(caption, facet.options, key=f"facet_{position}")
        if chosen:
            return Filter(facet.key, "in", chosen)
        return None

    typed = st.text_input(caption, key=f"facet_{position}")
    if typed:
        return Filter(facet.key, "contains", typed)

    return None


def render_facet_panel(connection, entity_type: str) -> list:
    """
    Draw the whole filter sidebar.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_type : str
        Kind of entry being searched.

    Returns
    -------
    filters : list of Filter
        Every active filter.
    """
    facets = build_facets(connection, entity_type=entity_type)

    if not facets:
        st.info("No metadata registered for this kind of entry yet.")
        return []

    filters = []

    for position, facet in enumerate(facets[:VISIBLE_FACET_LIMIT]):
        chosen = render_facet(facet, position)
        if chosen:
            filters.append(chosen)

    remaining = facets[VISIBLE_FACET_LIMIT:]
    if remaining:
        with st.expander(f"More filters ({len(remaining)})"):
            for position, facet in enumerate(remaining, start=VISIBLE_FACET_LIMIT):
                chosen = render_facet(facet, position)
                if chosen:
                    filters.append(chosen)

    return filters


# =============================================================================
# Results
# =============================================================================

def choose_columns(connection, entity_type: str, config: DatabaseAppConfig) -> list:
    """
    Let the user pick which metadata and results appear as table columns.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_type : str
        Kind of entry being searched.
    config : DatabaseAppConfig
        Deployment settings supplying the default column set.

    Returns
    -------
    columns : list of str
        Selected keys, results prefixed with ``result:``.
    """
    from pyKES.database.index_registry import read_metadata_keys

    metadata_options = [row["key"] for row in
                        read_metadata_keys(connection, entity_type=entity_type)]
    result_options = [f"{RESULT_PREFIX}{row['label']}" for row in
                      connection.execute("SELECT label FROM result_keys ORDER BY label")]

    available = result_options + metadata_options
    default = [column for column in config.default_columns if column in available]

    return st.multiselect("Columns", available, default=default,
                          format_func=display_key)


def render_results(connection, rows, total: int, columns: list,
                   offset: int, page_size: int) -> None:
    """
    Draw the result table, its pagination and its export.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    rows : list of sqlite3.Row
        This page of results.
    total : int
        Total matches.
    columns : list of str
        Extra columns to show.
    offset, page_size : int
        Pagination state.

    Returns
    -------
    None : None
    """
    if not rows:
        st.warning("No entries match these filters.")
        return

    frame = rows_to_frame(rows, columns)

    st.caption(f"Showing {offset + 1}–{offset + len(rows)} of {total}")

    selection = st.dataframe(frame, width="stretch", hide_index=True,
                             on_select="rerun", selection_mode="single-row",
                             key="search_results")

    chosen_rows = selection.get("selection", {}).get("rows", [])
    if chosen_rows:
        st.session_state[SELECTED_ENTITY_KEY] = frame.iloc[chosen_rows[0]]["entity_id"]
        st.switch_page("pages/02_Entity.py")

    st.download_button("Download these results as CSV",
                       data=frame.to_csv(index=False).encode("utf-8"),
                       file_name="photocat_search.csv", mime="text/csv")


# =============================================================================
# Entry point
# =============================================================================

def render_search(config: DatabaseAppConfig = DEFAULT_CONFIG) -> None:
    """
    Render the Browse & Search page.

    Parameters
    ----------
    config : DatabaseAppConfig, optional
        Deployment settings.

    Returns
    -------
    None : None
    """
    connection = open_shared_index(str(config.data_root))

    st.title("Browse & Search")

    url_type = st.query_params.get(TYPE_PARAMETER, config.default_entity_type)
    entity_type = st.selectbox(
        "Kind of entry", ENTITY_TYPES,
        index=ENTITY_TYPES.index(url_type) if url_type in ENTITY_TYPES else 0)

    text = st.text_input("Search names, groups and all metadata",
                         value=st.query_params.get(TEXT_PARAMETER, ""))

    with st.sidebar:
        st.header("Filters")
        st.caption("Generated from the metadata actually present — including "
                   "fields inherited through references.")
        filters = render_facet_panel(connection, entity_type)
        latest_only = st.toggle("Latest version of each name only", value=True)

    with st.expander("Advanced"):
        columns = choose_columns(connection, entity_type, config)

    write_filters_to_url(filters, text, entity_type)

    page = st.number_input("Page", min_value=1, value=1, step=1)
    offset = (int(page) - 1) * config.page_size

    rows, total = search_entities(connection, entity_type=entity_type,
                                  filters=filters, text=text,
                                  latest_only=latest_only,
                                  limit=config.page_size, offset=offset)

    if filters or text:
        st.success(f"{total} matching entries — this search is in the URL, so "
                   f"the link reproduces it.")

    render_results(connection, rows, total, columns, offset, config.page_size)
