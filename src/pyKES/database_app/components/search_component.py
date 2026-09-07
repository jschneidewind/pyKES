"""
Browse and search the database.

Every filter here is generated from the metadata-key registry rather than
configured by hand, so a spreadsheet column that first appears in next year's
uploads gets a working filter as soon as it has been ingested. Filters are
encoded into the URL, which makes a search something you can paste into a group
chat.
"""

import json
from itertools import groupby

import streamlit as st

from pyKES.database.index_query import (
    Facet,
    Filter,
    RESULT_PREFIX,
    build_facets,
    column_labels,
    display_entity_type,
    display_key,
    facet_group_label,
    rows_to_frame,
    search_entities,
)
from pyKES.database.entity_schema import load_entity_schemas
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

# Session-state keys for the widgets whose state has to outlive a rerun.
# Streamlit derives an unkeyed widget's identity from its arguments, so a text
# box seeded with `value=` from the query string is a *different* widget once
# the search has been written into the URL, and it comes back holding the
# previous term. A key pins the identity; the URL only seeds it once.
TEXT_KEY = "search_text"
TYPE_KEY = "search_entity_type"
FACET_KEY_PREFIX = "facet:"
MATCH_SAME_PREFIX = "same:"


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
            [{"key": item.key, "operator": item.operator, "value": item.value,
              "sub_key": item.sub_key, "match_same": item.match_same}
             for item in filters])
    elif FILTER_PARAMETER in st.query_params:
        del st.query_params[FILTER_PARAMETER]


# =============================================================================
# Facet widgets
# =============================================================================

def slider_default(widget_key: str, bounds: tuple) -> dict:
    """
    Supply a slider's starting range, unless its state has already been set.

    Streamlit refuses to be told a widget's value twice: passing ``value=`` for
    a key that `seed_facet_widgets` has already written warns, and the default
    is ignored anyway. Which happens on exactly the run that opens a shared
    link, since that is when the seeding writes.

    Parameters
    ----------
    widget_key : str
        Key the slider will be drawn under.
    bounds : tuple
        ``(minimum, maximum)`` the slider spans.

    Returns
    -------
    arguments : dict
        ``{'value': bounds}``, or nothing when the state already holds a range.
    """
    if widget_key in st.session_state:
        return {}

    return {"value": (float(bounds[0]), float(bounds[1]))}


def facet_widget_key(facet: Facet) -> str:
    """
    Name the session-state entry backing one filter widget.

    Keyed by the metadata key rather than by position, so switching the kind of
    entry cannot hand a slider the state of the multiselect that happened to sit
    in the same place.

    Parameters
    ----------
    facet : Facet
        Facet the widget filters on.

    Returns
    -------
    key : str
        Widget key.
    """
    return f"{FACET_KEY_PREFIX}{facet.key}"


def render_facet(facet: Facet) -> list:
    """
    Draw one filter widget and return the filters it produces.

    Parameters
    ----------
    facet : Facet
        Facet definition from the registry.

    Returns
    -------
    filters : list of Filter
        Empty while the widget is at its neutral setting. A mapping facet can
        produce several — one per name the user picked.
    """
    caption = facet.label
    widget_key = facet_widget_key(facet)

    if facet.kind == "mapping":
        return render_mapping_facet(facet, widget_key)

    if facet.kind == "range":
        low, high = facet.bounds
        chosen = st.slider(caption, min_value=float(low), max_value=float(high),
                           key=widget_key,
                           **slider_default(widget_key, facet.bounds))
        if chosen != (float(low), float(high)):
            return [build_filter(facet, "between", list(chosen))]
        return []

    if facet.kind == "select":
        chosen = st.multiselect(caption, facet.options, key=widget_key)
        if chosen:
            return [build_filter(facet, "in", chosen)]
        return []

    typed = st.text_input(caption, key=widget_key)
    if typed:
        return [build_filter(facet, "contains", typed)]

    return []


def render_mapping_facet(facet: Facet, widget_key: str) -> list:
    """
    Draw the two-level filter for a field holding a set of named numbers.

    A set of dopant concentrations is not one number, so it gets a name picker
    and then a slider per chosen name, over the range that name actually spans.
    Choosing a name and leaving its slider alone is already a filter — it asks
    for entries that carry that dopant at all — which is what somebody picking
    it from the list means.

    Parameters
    ----------
    facet : Facet
        Mapping facet from the registry.
    widget_key : str
        Key of the name picker; each slider takes one derived from it.

    Returns
    -------
    filters : list of Filter
        One per chosen name.
    """
    chosen = st.multiselect(f"{facet.label} — {facet.key_label or 'name'}",
                            facet.options, key=widget_key)

    filters = []

    for name in chosen:
        low, high = facet.sub_bounds.get(name, (0.0, 0.0))
        label = f"{name} [{facet.value_label}]" if facet.value_label else name

        # A name every entry carries at one value has nothing to slide, but the
        # filter still means "carries this one", so it is emitted regardless.
        if low == high:
            st.caption(f"{label}: {low:g}")
            filters.append(build_filter(facet, "between", [low, high], name))
            continue

        slider_key = f"{widget_key}#{name}"
        selected = st.slider(label, min_value=float(low), max_value=float(high),
                             key=slider_key,
                             **slider_default(slider_key, (low, high)))
        filters.append(build_filter(facet, "between", list(selected), name))

    return filters


def build_filter(facet: Facet, operator: str, value, sub_key: str = None) -> Filter:
    """
    Make a filter from one facet's current setting.

    Parameters
    ----------
    facet : Facet
        Facet the filter came from.
    operator : str
        Comparison to apply.
    value : Any
        What to compare against.
    sub_key : str, optional
        Name inside a mapping-valued field.

    Returns
    -------
    search_filter : Filter
        The filter, carrying whether it must agree with the other filters over
        its kind of entry on *which* entry satisfies them.
    """
    return Filter(facet.key, operator, value, sub_key=sub_key,
                  match_same=st.session_state.get(match_same_key(facet), False))


def match_same_key(facet: Facet) -> str:
    """
    Name the session-state entry holding one group's "same entry" setting.

    Parameters
    ----------
    facet : Facet
        Any facet of the group.

    Returns
    -------
    key : str
        Widget key.
    """
    return f"{MATCH_SAME_PREFIX}{facet.role_path or ''}"


def reset_filters() -> None:
    """
    Clear every filter and the free-text term.

    Runs as a button callback, before the rerun, because Streamlit refuses to
    have a widget's state assigned once that widget has been drawn. The query
    string is cleared with it: the seeding below reads it, so a reset that left
    it in place would immediately restore what it had just cleared.

    Returns
    -------
    None : None
    """
    for name in [key for key in st.session_state
                 if key.startswith((FACET_KEY_PREFIX, MATCH_SAME_PREFIX))]:
        del st.session_state[name]

    st.session_state[TEXT_KEY] = ""

    for parameter in (FILTER_PARAMETER, TEXT_PARAMETER):
        if parameter in st.query_params:
            del st.query_params[parameter]


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
    for facet in facets:
        widget_key = facet_widget_key(facet)
        matching = [search_filter for search_filter in url_filters
                    if search_filter.key == facet.key]

        if not matching or widget_key in st.session_state:
            continue

        if facet.kind == "mapping":
            # The picker holds the names; each name's slider is seeded under
            # its own key, which is why they are set together here.
            st.session_state[widget_key] = [item.sub_key for item in matching]
            for item in matching:
                st.session_state[f"{widget_key}#{item.sub_key}"] = (
                    float(item.value[0]), float(item.value[1]))

        elif facet.kind == "range":
            st.session_state[widget_key] = (float(matching[0].value[0]),
                                            float(matching[0].value[1]))
        else:
            st.session_state[widget_key] = matching[0].value


def render_facet_group(facets: list) -> list:
    """
    Draw one group of filter widgets.

    Parameters
    ----------
    facets : list of Facet
        Facets of a single entity, in the order they should appear.

    Returns
    -------
    filters : list of Filter
        The filters those widgets are currently set to.
    """
    return [chosen for facet in facets for chosen in render_facet(facet)]


def render_facet_panel(connection, entity_type: str,
                       config: DatabaseAppConfig) -> list:
    """
    Draw the whole filter sidebar.

    Filters are grouped by the entity they describe and each group is named
    after it — *Finished semiconductor* rather than *two references away* — with
    the groups themselves ordered by how far away that entity sits. The entry's
    own fields are always open, since that is where a search starts; the
    inherited groups collapse, so a chain four entities deep does not bury the
    fields somebody came to filter on.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_type : str
        Kind of entry being searched.
    config : DatabaseAppConfig
        Deployment settings, supplying the schema directory the mapping filters
        take their labels from.

    Returns
    -------
    filters : list of Filter
        Every active filter.
    """
    facets = build_facets(connection, entity_type=entity_type,
                          schemas=load_entity_schemas(config.schema_directory))

    if not facets:
        st.info("No metadata registered for this kind of entry yet.")
        return []

    seed_facet_widgets(facets, read_filters_from_url())

    filters = []

    # `build_facets` orders by depth and then by reference path, so consecutive
    # facets sharing a path are exactly the fields of one referenced entity.
    for role_path, group in groupby(facets, key=facet_role_path):
        group = list(group)
        heading = facet_group_label(group[0], entity_type)

        if not role_path:
            st.markdown(f"### {heading}")
            filters.extend(render_facet_group(group))
            continue

        with st.expander(f"**{heading}**", expanded=False):
            render_match_same_toggle(group[0], heading)
            filters.extend(render_facet_group(group))

    return filters


def render_match_same_toggle(facet: Facet, heading: str) -> None:
    """
    Offer to require that one entry satisfy every filter in a group.

    An entry can reach several entries of one kind — three precursor chemicals,
    or a semiconductor reached both directly and through a modified batch — so a
    filter on such a field asks whether *some* of them satisfies it. Two filters
    are then satisfiable by two different entries, which is usually the question
    ("something from Merck and something very pure") and sometimes not.

    Parameters
    ----------
    facet : Facet
        Any facet of the group, used for its kind.
    heading : str
        Name of the kind, for the label.

    Returns
    -------
    None : None
    """
    st.checkbox(f"Match one {heading.lower()}", key=match_same_key(facet),
                help=f"Off: each filter may be satisfied by a different "
                     f"{heading.lower()}. On: one of them must satisfy all of "
                     f"them.")


def facet_role_path(facet: Facet) -> str:
    """
    Read the reference path a facet was inherited through.

    Parameters
    ----------
    facet : Facet
        Facet to group.

    Returns
    -------
    role_path : str
        The path, or an empty string for the entry's own fields.
    """
    return facet.role_path or ""


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


def provenance_tooltips(columns: list) -> dict:
    """
    Describe each chosen column for the table's own header tooltips.

    The reference chain a field was inherited through is what makes it
    unambiguous and also what makes its header too wide to fit, so it lives in
    the tooltip and the header carries the field name alone.

    Parameters
    ----------
    columns : list of str
        Chosen keys, results prefixed with ``result:``.

    Returns
    -------
    column_config : dict
        Streamlit column configuration keyed by header.
    """
    return {header: st.column_config.Column(help=qualifier)
            for header, qualifier in column_labels(columns).values()}


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

    frame = rows_to_frame(rows, columns, connection)

    st.caption(f"Showing {offset + 1}–{offset + len(rows)} of {total}")

    selection = st.dataframe(frame, width="stretch", hide_index=True,
                             column_config=provenance_tooltips(columns),
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

def seed_search_widgets(config: DatabaseAppConfig) -> None:
    """
    Fill the kind and free-text widgets from a shared link, once.

    Only while they have no state of their own: after that the user's own typing
    wins, which is the whole reason these two carry keys rather than a ``value``
    recomputed from the query string on every rerun.

    Parameters
    ----------
    config : DatabaseAppConfig
        Deployment settings supplying the default kind of entry.

    Returns
    -------
    None : None
    """
    if TYPE_KEY not in st.session_state:
        url_type = st.query_params.get(TYPE_PARAMETER, config.default_entity_type)
        st.session_state[TYPE_KEY] = (url_type if url_type in ENTITY_TYPES
                                      else ENTITY_TYPES[0])

    if TEXT_KEY not in st.session_state:
        st.session_state[TEXT_KEY] = st.query_params.get(TEXT_PARAMETER, "")


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

    seed_search_widgets(config)

    entity_type = st.selectbox("Kind of entry", ENTITY_TYPES, key=TYPE_KEY,
                               format_func=display_entity_type)

    text = st.text_input("Search names, groups and all metadata",
                         placeholder="Catalyst, operator, note…", key=TEXT_KEY)

    with st.sidebar:
        st.header("Filters")
        st.caption("Generated from the metadata actually present, grouped by "
                   "the entity each field describes.")
        latest_only = st.toggle("Latest version of each name only", value=True)
        st.button("Reset All Filters", on_click=reset_filters, width="stretch")
        filters = render_facet_panel(connection, entity_type, config)

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
