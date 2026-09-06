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
    display_entity_type,
    display_key,
    display_role_path,
    reference_depth,
    rows_to_frame,
    search_entities,
)
from pyKES.database.index_schema import ENTITY_TYPES
from pyKES.database_app.components.time_series_panel import (
    MAX_COMPARISON_ENTRIES,
    payload_files_for,
    render_time_series_panel,
)
from pyKES.database_app.config import DEFAULT_CONFIG, DatabaseAppConfig
from pyKES.database_app.session import index_paths, open_shared_index


# =============================================================================
# Display constants
# =============================================================================

# Session-state key holding the entity the user opened from the results table.
SELECTED_ENTITY_KEY = "selected_entity_id"

# Query-parameter names. Kept short because they end up in a pasted URL.
TEXT_PARAMETER = "q"
TYPE_PARAMETER = "type"
FILTER_PARAMETER = "f"

# Headings the facets are grouped under, by how many references away the field
# lives. Anything deeper than this falls back to the reference path itself.
DEPTH_HEADINGS = ("This experiment", "One reference away", "Two references away",
                  "Three references away")


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
    caption = facet.label

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


def seed_facet_widgets(facets: list, url_filters: list) -> None:
    """
    Pre-fill the filter widgets from a shared link.

    Without this the query string is written but never read, so a pasted search
    link opens the unfiltered page — which makes "a search is a URL" false.
    Streamlit widgets take their initial value from session state, so the
    seeding has to happen before they are created, and only while they have no
    state of their own: after that the user's own interaction wins.

    Parameters
    ----------
    facets : list of Facet
        Facets about to be drawn, in the order they will be drawn in.
    url_filters : list of Filter
        Filters decoded from the query string.

    Returns
    -------
    None : None
    """
    by_key = {search_filter.key: search_filter for search_filter in url_filters}

    for position, facet in enumerate(facets):
        widget_key = f"facet_{position}"
        search_filter = by_key.get(facet.key)

        if search_filter is None or widget_key in st.session_state:
            continue

        if facet.kind == "range":
            st.session_state[widget_key] = (float(search_filter.value[0]),
                                            float(search_filter.value[1]))
        else:
            st.session_state[widget_key] = search_filter.value


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

    seed_facet_widgets(facets, read_filters_from_url())

    filters = []
    current_depth = None

    # `build_facets` already orders by reference depth, so a single pass emits
    # the entry's own fields first and then each level of the chain. Every
    # filter is shown: hiding two thirds of them behind an expander hid exactly
    # the inherited ones the comparison is usually built from.
    for position, facet in enumerate(facets):
        depth = reference_depth(facet)

        if depth != current_depth:
            current_depth = depth
            st.markdown(f"**{depth_heading(depth, facet)}**")

        chosen = render_facet(facet, position)
        if chosen:
            filters.append(chosen)

    return filters


def depth_heading(depth: int, facet: Facet) -> str:
    """
    Name the group a facet belongs to.

    Parameters
    ----------
    depth : int
        How many references away the field lives.
    facet : Facet
        A facet at that depth, used for its reference path when the depth is
        beyond the named headings.

    Returns
    -------
    heading : str
        Group heading.
    """
    if depth < len(DEPTH_HEADINGS):
        return DEPTH_HEADINGS[depth]

    return display_role_path(facet.role_path)


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

    return st.multiselect("Columns shown", available, default=default,
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
        st.session_state[SELECTED_ENTITY_KEY] = frame.iloc[chosen_rows[0]]["Entity ID"]
        st.switch_page("pages/02_Entity.py")

    st.download_button("Download Results as CSV",
                       data=frame.to_csv(index=False).encode("utf-8"),
                       file_name="photocat_search.csv", mime="text/csv")


def render_comparison(connection, config: DatabaseAppConfig,
                      entity_type: str, filters: list, text: str,
                      latest_only: bool, total: int) -> None:
    """
    Plot the traces of everything the current search selects.

    Comparing a subset — every test of one catalyst, every catalyst descended
    from one semiconductor — is the main thing the database is for, so the plot
    follows the filters rather than needing a second selection. The panel's own
    multiselect narrows what is drawn without touching the search.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    config : DatabaseAppConfig
        Deployment settings, supplying the payload directory.
    entity_type : str
        Kind of entry being searched.
    filters : list of Filter
        Active filters.
    text : str
        Free-text term.
    latest_only : bool
        Whether superseded versions are hidden.
    total : int
        Number of matches, used to decide whether a comparison is worth drawing.

    Returns
    -------
    None : None
    """
    st.subheader("Compare traces")

    if total > MAX_COMPARISON_ENTRIES:
        st.info(f"{total} entries match. Narrow the search to "
                f"{MAX_COMPARISON_ENTRIES} or fewer to compare their traces.")
        return

    # The whole matching set, not just the visible page: the comparison is of
    # the subset the filters describe.
    rows, _ = search_entities(connection, entity_type=entity_type,
                              filters=filters, text=text,
                              latest_only=latest_only, limit=None)

    render_time_series_panel(
        payload_files_for(rows, index_paths(config).payload_directory),
        key_prefix="browse")


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
        index=ENTITY_TYPES.index(url_type) if url_type in ENTITY_TYPES else 0,
        format_func=display_entity_type)

    text = st.text_input("Search names, groups and all metadata",
                         value=st.query_params.get(TEXT_PARAMETER, ""),
                         placeholder="Catalyst, operator, note…")

    with st.sidebar:
        st.header("Filters")
        st.caption("Generated from the metadata actually present, grouped by "
                   "how far away the field lives.")
        latest_only = st.toggle("Latest version of each name only", value=True)
        filters = render_facet_panel(connection, entity_type)

    with st.expander("Table columns"):
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

    st.divider()
    render_comparison(connection, config, entity_type, filters, text,
                      latest_only, total)
