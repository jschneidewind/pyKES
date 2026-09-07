"""
Reading the database index: search, facets, the reference graph and statistics.

Every search runs against the materialised ``effective`` column, so a filter on
a metadata field two references away costs the same as one on the entity's own
metadata. The search page never has to know how deep a value came from.

Metadata keys are user-supplied strings that arrive from spreadsheet headers, so
they are never interpolated into SQL. SQLite accepts a JSON path as a bound
parameter — ``json_extract(effective, ?)`` — which is what keeps a column called
``Notes'; DROP TABLE`` harmless.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pyKES.database.database_experiments import sanitize_key
from pyKES.database.index_registry import (
    TYPE_BOOLEAN,
    TYPE_MAPPING,
    TYPE_NUMBER,
    TYPE_TEXT,
    split_qualified_key,
)
from pyKES.database.index_schema import ROLE_PATH_SEPARATOR


# =============================================================================
# Query construction
# =============================================================================

# Columns a filter may address directly rather than through the JSON metadata.
COLUMN_FILTERS = {
    "entity_type": "entity_type",
    "owner": "owner",
    "display_group": "display_group",
    "active": "active",
}

# Prefix marking a filter or axis that addresses a mapped result rather than a
# metadata key, e.g. 'result:Max. rate (umol/s)'.
RESULT_PREFIX = "result:"

# Rows returned per page. Rendering ten thousand rows into a table is the one
# easy way to make a millisecond query feel slow.
DEFAULT_PAGE_SIZE = 50

# A text key with more distinct values than this is offered as a search box
# rather than a multiselect.
MULTISELECT_MAXIMUM_OPTIONS = 30

# Shown between the roles of a reference chain, and where a value is absent.
ROLE_DISPLAY_SEPARATOR = " \u203a "
MISSING_PLACEHOLDER = "\u2014"

# What a column's tooltip says when its values are not inherited.
OWN_QUALIFIER = "This entry"
RESULT_QUALIFIER = "Result"


@dataclass
class Filter:
    """
    One search predicate.

    Parameters
    ----------
    key : str
        Column name, *stored* metadata key, or ``'result:<label>'``. Stored
        keys have their own slashes escaped, which is the form the registry and
        `build_facets` report; `stored_metadata_key` converts a key written out
        by hand.
    operator : {'between', 'in', 'contains', 'equals'}
        How ``value`` is compared.
    value : Any
        ``(low, high)`` for ``between``, a list for ``in``, otherwise a scalar.
    sub_key : str, optional
        Name inside a mapping-valued field — one dopant of a set of dopant
        concentrations. Its own field rather than something encoded into
        ``key``, because the key already carries the reference path and
        stacking a second separator into it is how the escaping bug happened.
    alternative_keys : list of str, optional
        Other paths reaching the same field, matched as well as ``key``. One
        field is reachable by several paths as soon as a role accepts entries
        of different kinds: an experiment reaches its semiconductor in two hops
        through an ordinary batch and in three through a modified one. Without
        this a filter would answer only for half the experiments, which is
        worse than not offering it.
    """

    key: str
    operator: str
    value: Any
    sub_key: Optional[str] = None
    alternative_keys: List[str] = field(default_factory=list)


def stored_metadata_key(key: str) -> str:
    """
    Convert a human-written metadata key into the form the index stores.

    Only needed when building a filter by hand: every key that comes from the
    registry, from `build_facets` or from `list_axis_options` is already stored
    form.

    Parameters
    ----------
    key : str
        Metadata key as written on the entity, possibly containing slashes.

    Returns
    -------
    stored : str
        Key with its own slashes escaped.
    """
    return sanitize_key(key)


def display_entity_type(entity_type: str) -> str:
    """
    Render a stored entity type the way it should read on screen.

    ``'finished_semiconductor'`` becomes ``'Finished semiconductor'``: the
    underscores are a storage convention, not something a user should have to
    look at.

    Parameters
    ----------
    entity_type : str
        Stored entity type.

    Returns
    -------
    label : str
        Sentence-cased name.
    """
    return entity_type.replace("_", " ").capitalize()


def display_role_path(role_path: Optional[str]) -> str:
    """
    Render a reference path as a readable chain.

    ``'catalyst_batch/finished_semiconductor'`` becomes
    ``'Catalyst batch › Finished semiconductor'``.

    Parameters
    ----------
    role_path : str or None
        Stored role path, or None for an entity's own field.

    Returns
    -------
    label : str
        Readable chain, empty for an own field.
    """
    if not role_path:
        return ""

    return ROLE_DISPLAY_SEPARATOR.join(display_entity_type(role)
                                       for role in role_path.split(ROLE_PATH_SEPARATOR))


def display_key(key: str) -> str:
    """
    Render a stored key the way a person wrote it.

    The escaping that keeps reference paths splittable is storage detail, so it
    is undone everywhere a key reaches the screen — a column header reading
    ``Irradiance A [mW__SLASH__cm2]`` is not an improvement on the bug it fixed.

    Parameters
    ----------
    key : str
        Stored metadata key, possibly qualified, or ``'result:<label>'``.

    Returns
    -------
    label : str
        Key with its escaped slashes restored; a result label loses its prefix.
    """
    if key.startswith(RESULT_PREFIX):
        return key[len(RESULT_PREFIX):]

    role_path, leaf = split_qualified_key(key)

    return f"{leaf}  ·  via {display_role_path(role_path)}" if role_path else leaf


def column_leaf(key: str) -> str:
    """
    Name a column the way its own entity wrote it, without provenance.

    Parameters
    ----------
    key : str
        Stored metadata key, possibly qualified, or ``'result:<label>'``.

    Returns
    -------
    leaf : str
        Bare field name.
    """
    if key.startswith(RESULT_PREFIX):
        return key[len(RESULT_PREFIX):]

    return split_qualified_key(key)[1]


def column_qualifier(key: str) -> str:
    """
    Say where a column's values come from.

    Parameters
    ----------
    key : str
        Stored metadata key, possibly qualified, or ``'result:<label>'``.

    Returns
    -------
    qualifier : str
        Reference chain the field was inherited through, ``'Result'`` for a
        mapped result, or ``'This entry'`` for the entity's own field.
    """
    if key.startswith(RESULT_PREFIX):
        return RESULT_QUALIFIER

    role_path, _ = split_qualified_key(key)

    return display_role_path(role_path) or OWN_QUALIFIER


def column_labels(columns: List[str]) -> Dict[str, Tuple[str, str]]:
    """
    Choose short, unique table headers for a set of chosen columns.

    A header carrying its whole reference chain — ``Synthesis temperature [°C]
    · via Catalyst batch › Finished semiconductor`` — is wider than the table,
    so the next chosen column lands off-screen and the user concludes their
    selection did nothing. The chain moves to the column's tooltip and only the
    field name is shown, lengthened just enough to stay unambiguous when two
    chosen columns share a name.

    Parameters
    ----------
    columns : list of str
        Stored keys, results prefixed with ``result:``.

    Returns
    -------
    labels : dict
        ``{key: (header, qualifier)}``, in the order the columns were given.
        Headers are unique, so no column can silently overwrite another.
    """
    leaves = [column_leaf(column) for column in columns]
    repeated = {leaf for leaf in leaves if leaves.count(leaf) > 1}

    labels, taken = {}, set()

    for column, leaf in zip(columns, leaves):
        qualifier = column_qualifier(column)
        header = f"{leaf} ({qualifier.split(ROLE_DISPLAY_SEPARATOR)[-1]})" \
            if leaf in repeated else leaf

        # Two chains can end in the same role at different depths. Falling back
        # to the whole chain separates them; two columns cannot get past that,
        # since equal leaf and equal chain means the same key.
        if header in taken:
            header = f"{leaf} ({qualifier})"

        taken.add(header)
        labels[column] = (header, qualifier)

    return labels


def format_value(value: Any) -> str:
    """
    Render one metadata or result value as display text.

    Metadata columns hold whatever the spreadsheets carried, so a single column
    can mix numbers, text and booleans. Handing pandas that mixture produces an
    object column Arrow cannot serialise, which Streamlit reports as a
    traceback and then silently repairs. Rendering to text first is what keeps
    that off the console — and it is also what will let a value that is itself a
    mapping, such as a set of dopant concentrations, appear in a table at all.

    Parameters
    ----------
    value : Any
        Value taken from stored metadata or results.

    Returns
    -------
    text : str
        Display text; a missing value becomes the placeholder.
    """
    if value is None:
        return MISSING_PLACEHOLDER

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def arrow_safe_frame(records: List[Dict[str, Any]]):
    """
    Build a display table every column of which Arrow can serialise.

    Streamlit hands each table to Arrow, which needs one type per column. A
    column pandas could type — a column of numbers, or of booleans — is left
    alone, so it keeps its alignment, its sort and Streamlit's own number
    formatting; only a column pandas fell back to ``object`` for, which is
    exactly the mixture Arrow rejects, is rendered as text.

    Parameters
    ----------
    records : list of dict
        Table rows.

    Returns
    -------
    frame : pandas.DataFrame
        The table, with its untyped columns rendered as text.
    """
    import pandas as pd

    frame = pd.DataFrame(records)

    for name in frame.columns[frame.dtypes == object]:
        frame[name] = frame[name].map(format_value)

    return frame


def json_path(key: str, sub_key: Optional[str] = None) -> str:
    """
    Build the JSON path addressing one metadata or result key.

    Parameters
    ----------
    key : str
        Metadata key, possibly qualified and possibly containing quotes.
    sub_key : str, optional
        Name inside a mapping-valued field, addressed one level deeper.

    Returns
    -------
    path : str
        SQLite JSON path, bound as a parameter rather than interpolated.
    """
    path = '$."' + key.replace('"', '\\"') + '"'

    if sub_key is None:
        return path

    return path + '."' + str(sub_key).replace('"', '\\"') + '"'


def build_expression(key: str, sub_key: Optional[str] = None,
                     alternative_keys: Optional[List[str]] = None) -> Tuple[str, list]:
    """
    Render the SQL expression a filter or axis addresses.

    Several paths to one field are combined with ``COALESCE`` rather than by
    repeating the predicate, so every operator keeps working unchanged and the
    filter reads as one value. An entity carries at most one of the paths — a
    batch is either modified or it is not — and where it somehow carried two,
    the shortest wins, which is the one nearest the entity being searched.

    Parameters
    ----------
    key : str
        Column name, qualified metadata key, or ``'result:<label>'``.
    sub_key : str, optional
        Name inside a mapping-valued field.
    alternative_keys : list of str, optional
        Other paths reaching the same field.

    Returns
    -------
    expression : str
        SQL fragment with a placeholder for each JSON path.
    parameters : list
        Values to bind for that fragment.
    """
    if key in COLUMN_FILTERS:
        return COLUMN_FILTERS[key], []

    if key.startswith(RESULT_PREFIX):
        return "json_extract(results, ?)", [json_path(key[len(RESULT_PREFIX):])]

    paths = [json_path(one, sub_key) for one in [key] + list(alternative_keys or [])]

    if len(paths) == 1:
        return "json_extract(effective, ?)", paths

    extracts = ", ".join("json_extract(effective, ?)" for _ in paths)

    return f"COALESCE({extracts})", paths


def build_predicate(search_filter: Filter) -> Tuple[str, list]:
    """
    Render one filter as a SQL predicate.

    Parameters
    ----------
    search_filter : Filter
        Predicate to render.

    Returns
    -------
    predicate : str
        SQL fragment.
    parameters : list
        Values to bind.

    Raises
    ------
    ValueError
        If the operator is not one of the four supported ones.
    """
    expression, parameters = build_expression(search_filter.key,
                                              search_filter.sub_key,
                                              search_filter.alternative_keys)

    if search_filter.operator == "between":
        low, high = search_filter.value
        return f"CAST({expression} AS REAL) BETWEEN ? AND ?", parameters + [low, high]

    if search_filter.operator == "in":
        options = list(search_filter.value)
        if not options:
            # An empty selection means "no constraint", not "match nothing".
            return "1 = 1", []
        placeholders = ", ".join("?" for _ in options)
        return f"{expression} IN ({placeholders})", parameters + options

    if search_filter.operator == "contains":
        return f"CAST({expression} AS TEXT) LIKE ?", parameters + [f"%{search_filter.value}%"]

    if search_filter.operator == "equals":
        return f"{expression} = ?", parameters + [search_filter.value]

    raise ValueError(f"Unsupported filter operator '{search_filter.operator}'.")


def build_free_text_predicate(text: str) -> Tuple[str, list]:
    """
    Render a free-text search across identity and the whole metadata blob.

    Searching the serialised metadata rather than named columns is what makes a
    note somebody typed into an unexpected column findable at all.

    Parameters
    ----------
    text : str
        Search term.

    Returns
    -------
    predicate : str
        SQL fragment.
    parameters : list
        Values to bind.
    """
    pattern = f"%{text}%"

    return ("(entity_id LIKE ? OR display_group LIKE ? OR owner LIKE ? "
            "OR effective LIKE ?)", [pattern, pattern, pattern, pattern])


# =============================================================================
# Search
# =============================================================================

def search_entities(connection,
                    entity_type: Optional[str] = None,
                    filters: Optional[List[Filter]] = None,
                    text: Optional[str] = None,
                    latest_only: bool = True,
                    order_by: str = "entity_id",
                    descending: bool = False,
                    limit: Optional[int] = DEFAULT_PAGE_SIZE,
                    offset: int = 0) -> Tuple[List, int]:
    """
    Find entities matching a set of predicates.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_type : str, optional
        Restrict to one kind of entry.
    filters : list of Filter, optional
        Predicates, combined with AND.
    text : str, optional
        Free-text term.
    latest_only : bool, optional
        Hide superseded versions of a name, showing only the highest version of
        each ``base_id``.
    order_by : str, optional
        Column to sort by; anything not a real column falls back to
        ``entity_id`` rather than being interpolated into SQL.
    descending : bool, optional
        Sort direction.
    limit, offset : int, optional
        Pagination. ``limit=None`` returns everything, which the property map
        and the export use.

    Returns
    -------
    rows : list of sqlite3.Row
        Matching entities for this page.
    total : int
        Number of matches before pagination.
    """
    clauses, parameters = [], []

    if entity_type:
        clauses.append("entity_type = ?")
        parameters.append(entity_type)

    for search_filter in filters or []:
        predicate, values = build_predicate(search_filter)
        clauses.append(predicate)
        parameters.extend(values)

    if text:
        predicate, values = build_free_text_predicate(text)
        clauses.append(predicate)
        parameters.extend(values)

    if latest_only:
        # 'other', not 'inner': INNER is a SQL keyword and aliasing a table
        # with it is a syntax error.
        clauses.append(
            "version = (SELECT MAX(other.version) FROM entities AS other "
            "WHERE other.base_id = entities.base_id)"
        )

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    total = connection.execute(
        f"SELECT COUNT(*) AS n FROM entities {where}", parameters
    ).fetchone()["n"]

    sortable = {"entity_id", "entity_type", "display_group", "owner",
                "updated_at", "created_at", "version"}
    column = order_by if order_by in sortable else "entity_id"
    direction = "DESC" if descending else "ASC"

    pagination = ""
    if limit is not None:
        pagination = "LIMIT ? OFFSET ?"
        parameters = parameters + [limit, offset]

    rows = connection.execute(
        f"SELECT * FROM entities {where} ORDER BY {column} {direction} {pagination}",
        parameters,
    ).fetchall()

    return rows, total


def rows_to_frame(rows: List, columns: Optional[List[str]] = None):
    """
    Flatten search results into a table for display.

    Parameters
    ----------
    rows : list of sqlite3.Row
        Search results.
    columns : list of str, optional
        Effective-metadata and result keys to include as columns, in order.
        Result keys carry the ``result:`` prefix.

    Returns
    -------
    frame : pandas.DataFrame
        One row per entity, indexed by nothing so Streamlit can select rows.
        The chosen columns come directly after the identifier and the fixed
        context columns follow them, so a column somebody just asked for is
        visible without scrolling a wide table sideways.
    """
    labels = column_labels(list(columns or []))
    records = []

    for row in rows:
        effective = json.loads(row["effective"])
        results = json.loads(row["results"])

        record = {"Entity ID": row["entity_id"]}

        for column, (header, _) in labels.items():
            record[header] = (results.get(column[len(RESULT_PREFIX):])
                              if column.startswith(RESULT_PREFIX)
                              else effective.get(column))

        record.update({"Kind": display_entity_type(row["entity_type"]),
                       "Group": row["display_group"], "Owner": row["owner"]})
        records.append(record)

    return arrow_safe_frame(records)


# =============================================================================
# Facets
# =============================================================================

@dataclass
class Facet:
    """
    One filter widget the search page should draw.

    Parameters
    ----------
    key : str
        Qualified metadata key the facet filters on.
    label : str
        Leaf name, which is what the user recognises.
    kind : {'range', 'select', 'contains', 'mapping'}
        Widget to draw.
    options : list
        Choices for a ``select`` facet, or the names inside a ``mapping`` one.
    bounds : tuple
        ``(minimum, maximum)`` for a ``range`` facet.
    role_path : str or None
        Reference path the key was inherited through, shown as provenance.
    alternative_keys : list of str
        Other paths reaching the same field, filtered on alongside ``key``.
    sub_bounds : dict
        ``{name: (minimum, maximum)}`` for a ``mapping`` facet, so choosing a
        dopant can raise a slider over the concentrations that dopant actually
        spans.
    key_label, value_label : str, optional
        What a mapping's names and numbers are called, from the schema.
    """

    key: str
    label: str
    kind: str
    options: list = field(default_factory=list)
    bounds: tuple = ()
    role_path: Optional[str] = None
    sub_bounds: Dict[str, tuple] = field(default_factory=dict)
    key_label: Optional[str] = None
    value_label: Optional[str] = None
    alternative_keys: List[str] = field(default_factory=list)


def numeric_bounds(connection, keys, entity_type: Optional[str],
                   sub_key: Optional[str] = None) -> tuple:
    """
    Find the range a numeric metadata key spans.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    keys : str or list of str
        Qualified metadata key, or every path reaching one field.
    entity_type : str, optional
        Restrict to one kind of entry.
    sub_key : str, optional
        Name inside a mapping-valued field, whose own range is wanted.

    Returns
    -------
    bounds : tuple
        ``(minimum, maximum)``, or ``(0.0, 0.0)`` when the key holds no numbers.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    expression, paths = build_expression(keys[0], sub_key, keys[1:])

    clause = "WHERE entity_type = ?" if entity_type else ""
    parameters = paths + paths + ([entity_type] if entity_type else [])

    row = connection.execute(
        f"""SELECT MIN(CAST({expression} AS REAL)) AS low,
                   MAX(CAST({expression} AS REAL)) AS high
            FROM entities {clause}""",
        parameters,
    ).fetchone()

    if row["low"] is None:
        return (0.0, 0.0)

    return (float(row["low"]), float(row["high"]))


def build_facets(connection,
                 entity_type: Optional[str] = None,
                 minimum_occurrences: int = 1,
                 schemas: Optional[Dict[str, Any]] = None) -> List[Facet]:
    """
    Generate the filter widgets for a kind of entry from the key registry.

    Nothing here is configured by hand: a metadata column that appears in next
    year's uploads gets a facet as soon as it has been ingested, and its widget
    follows from the type the registry observed.

    The registry is scoped by entity type because in a reference chain one leaf
    name occurs at one path per type — bare on the precursor, one hop away on
    the batch, two on the experiment.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_type : str, optional
        Kind of entry the facets are for.
    minimum_occurrences : int, optional
        Skip keys seen fewer times than this, so a one-off typo does not become
        a permanent filter.
    schemas : dict, optional
        Entity schemas, used only to label a mapping facet's two levels with
        the names the group gave them.

    Returns
    -------
    facets : list of Facet
        Ordered by reference depth — the entry's own fields first, then fields
        one reference away, and so on — and grouped by the entity each field
        was inherited from within a depth. That is the order somebody narrowing
        a search thinks in: what was done in this experiment, then what it was
        made from.
    """
    from pyKES.database.index_registry import read_metadata_keys

    schemas = schemas or {}
    facets = []

    rows = [row for row in read_metadata_keys(connection, entity_type=entity_type)
            if row["occurrences"] >= minimum_occurrences and not row["canonical_key"]]

    for group in merge_key_paths(rows):
        facet = build_facet(connection, group, entity_type, schemas)
        if facet is not None:
            facets.append(facet)

    # Sorting by the path as well as the depth is what keeps the fields of one
    # referenced entity together, so the page can put them under its name.
    return sorted(facets, key=lambda facet: (reference_depth(facet),
                                             facet.role_path or ""))


def merge_key_paths(rows: List) -> List[dict]:
    """
    Collect the registry rows that describe one field reached several ways.

    A role that accepts more than one kind of entry puts the same field at more
    than one depth: an experiment reaches its semiconductor in two hops through
    an ordinary catalyst batch and in three through a modified one. Those are
    the *same* field of the *same* entity, so they belong in one filter — one
    per path would answer for half the experiments each, and neither would say
    so.

    The grouping is by the last role and the field name, not by the field name
    alone: the last role names the entity the field belongs to, while the rest
    of the path is only how it was reached. That keeps a batch's ``Notes`` and
    a semiconductor's ``Notes`` apart, which sharing a name is not enough to
    merge.

    Parameters
    ----------
    rows : list of sqlite3.Row
        Registry rows for one kind of entry.

    Returns
    -------
    groups : list of dict
        One entry per field, carrying every path to it, the shortest of them as
        the one the widget is named after, and the merged type, sample and
        mapping names.
    """
    groups = {}

    for row in rows:
        role_path = row["role_path"] or ""
        identity = (role_path.split(ROLE_PATH_SEPARATOR)[-1], row["leaf_name"])

        group = groups.setdefault(identity, {
            "keys": [], "leaf_name": row["leaf_name"], "role_path": role_path,
            "inferred_type": row["inferred_type"], "sample": [], "sub_keys": [],
        })

        group["keys"].append(row["key"])
        group["sample"] = merge_unique(group["sample"],
                                       json.loads(row["distinct_sample"] or "[]"))
        group["sub_keys"] = merge_unique(group["sub_keys"],
                                         json.loads(row["sub_keys"] or "[]"))

        # The shortest path is the one nearest the entity being searched, and
        # is what the filter is named after and keyed by.
        if len(row["key"]) < len(group["keys"][0]):
            group["keys"].insert(0, group["keys"].pop())
            group["role_path"] = role_path

        if row["inferred_type"] != group["inferred_type"]:
            group["inferred_type"] = TYPE_TEXT

    return list(groups.values())


def merge_unique(existing: list, incoming: list) -> list:
    """
    Add what is new, keeping the order stable.

    Parameters
    ----------
    existing, incoming : list
        Values collected so far, and values to add.

    Returns
    -------
    merged : list
        Existing values followed by the ones not already present.
    """
    return existing + [value for value in incoming if value not in existing]


def build_facet(connection, group: dict, entity_type: Optional[str],
                schemas: Dict[str, Any]) -> Optional[Facet]:
    """
    Choose the widget one field gets, from the type the registry observed.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    group : dict
        One entry from `merge_key_paths`.
    entity_type : str, optional
        Kind of entry the facet is for.
    schemas : dict
        Entity schemas, for the labels of a mapping's two levels.

    Returns
    -------
    facet : Facet or None
        The facet, or None where the field cannot usefully be filtered on — a
        number every entry shares, or a mapping whose names are not yet
        registered.
    """
    keys, alternatives = group["keys"][0], group["keys"][1:]
    common = dict(label=group["leaf_name"], role_path=group["role_path"] or None,
                  alternative_keys=alternatives)

    if group["inferred_type"] == TYPE_MAPPING:
        return mapping_facet(connection, group, entity_type, schemas, common)

    if group["inferred_type"] == TYPE_NUMBER:
        bounds = numeric_bounds(connection, group["keys"], entity_type)
        if bounds[0] == bounds[1]:
            return None
        return Facet(keys, kind="range", bounds=bounds, **common)

    if group["inferred_type"] == TYPE_BOOLEAN:
        return Facet(keys, kind="select", options=[True, False], **common)

    if len(group["sample"]) <= MULTISELECT_MAXIMUM_OPTIONS:
        return Facet(keys, kind="select",
                     options=sorted(str(value) for value in group["sample"]),
                     **common)

    return Facet(keys, kind="contains", **common)


def mapping_facet(connection, group: dict, entity_type: Optional[str],
                  schemas: Dict[str, Any], common: dict) -> Optional[Facet]:
    """
    Build the two-level filter for a mapping-valued key.

    A set of dopant concentrations is not one number, so it gets a name picker
    and then a slider over the concentrations that name actually spans — which
    is the question somebody asks of it ("anything with 0.02 to 0.05 mol% Ir").
    The bounds are resolved per name here rather than in the page, so the widget
    knows its own range before it is drawn.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    group : dict
        One entry from `merge_key_paths`.
    entity_type : str, optional
        Kind of entry the facet is for.
    schemas : dict
        Entity schemas, for the labels the group gave the two levels.
    common : dict
        Label, reference path and alternative keys shared by every facet.

    Returns
    -------
    facet : Facet or None
        The facet, or None when the key has no names recorded — which happens
        for an index built before mappings existed and until it is rebuilt.
    """
    names = group["sub_keys"]
    if not names:
        return None

    bounds = {name: numeric_bounds(connection, group["keys"], entity_type, name)
              for name in names}

    declared = mapping_field_schema(group["leaf_name"], schemas)

    return Facet(group["keys"][0], kind="mapping", options=names,
                 sub_bounds=bounds,
                 key_label=declared.key_label if declared else None,
                 value_label=declared.value_label if declared else None,
                 **common)


def mapping_field_schema(leaf_name: str, schemas: Dict[str, Any]):
    """
    Find the schema declaration of a mapping field, wherever it was declared.

    A mapping inherited through a reference belongs to the entity it came from,
    not to the one being searched, so the lookup is by field name across every
    schema rather than within one.

    Parameters
    ----------
    leaf_name : str
        Field name as its own entity wrote it.
    schemas : dict
        Entity schemas.

    Returns
    -------
    field_schema : FieldSchema or None
        The declaration, or None when nobody declared this field.
    """
    for schema in schemas.values():
        declared = schema.field_named(leaf_name)
        if declared is not None and declared.type == "mapping":
            return declared

    return None


def reference_depth(facet: Facet) -> int:
    """
    Count how many references away a facet's field lives.

    Parameters
    ----------
    facet : Facet
        Facet to place.

    Returns
    -------
    depth : int
        0 for the entity's own metadata, 1 for one reference away, and so on.
    """
    if not facet.role_path:
        return 0

    return facet.role_path.count(ROLE_PATH_SEPARATOR) + 1


def facet_group_label(facet: Facet, entity_type: Optional[str]) -> str:
    """
    Name the entity a facet's field belongs to.

    A filter group is easier to recognise by the thing it describes — *Finished
    semiconductor* — than by how far away it sits, so the heading names the
    entity rather than counting hops. The depth still decides the order.

    Parameters
    ----------
    facet : Facet
        Facet to place.
    entity_type : str or None
        Kind of entry being searched, used to name the group of its own fields.

    Returns
    -------
    label : str
        Entity name for the group heading.
    """
    if not facet.role_path:
        return display_entity_type(entity_type) if entity_type else OWN_QUALIFIER

    return display_entity_type(facet.role_path.split(ROLE_PATH_SEPARATOR)[-1])


# =============================================================================
# One entity
# =============================================================================

def search_entity_ids(connection,
                      prefix: str,
                      entity_type: Optional[str] = None,
                      limit: int = 25) -> List[str]:
    """
    List entity ids beginning with, or containing, a typed fragment.

    Backs the type-ahead on the entry page: typing ``NB-6`` should offer every
    entry whose id starts that way without the user having to remember the rest.
    Prefix matches are offered before mere containments, since that is what a
    person typing an id means.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    prefix : str
        What the user has typed.
    entity_type : str, optional
        Restrict to one kind of entry.
    limit : int, optional
        Most matches to return.

    Returns
    -------
    entity_ids : list of str
        Matching ids, prefix matches first.
    """
    if not prefix.strip():
        return []

    fragment = prefix.strip()
    clause = "AND entity_type = ?" if entity_type else ""
    parameters = [f"{fragment}%", f"%{fragment}%", f"{fragment}%"]
    if entity_type:
        parameters.insert(2, entity_type)

    rows = connection.execute(
        f"""SELECT entity_id FROM entities
             WHERE (entity_id LIKE ? OR entity_id LIKE ?) {clause}
             ORDER BY entity_id NOT LIKE ?, entity_id
             LIMIT {int(limit)}""",
        parameters,
    ).fetchall()

    return [row["entity_id"] for row in rows]


def read_entity(connection, entity_id: str):
    """
    Read one entity row.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity to read.

    Returns
    -------
    row : sqlite3.Row or None
        The entity, or None if it is not in the database.
    """
    return connection.execute(
        "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()


def read_neighbours(connection, entity_id: str) -> Dict[str, list]:
    """
    Read what an entity references and what references it.

    The reverse direction is what answers questions the group cannot currently
    ask at all — *"every test ever run on material descended from BC-2"*.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity in question.

    Returns
    -------
    neighbours : dict
        ``{'references': [...], 'referenced_by': [...]}``, each a list of rows
        carrying the role and the other entity.
    """
    references = connection.execute(
        """SELECT edges.role, edges.target AS entity_id, edges.resolved,
                  entities.entity_type
             FROM edges LEFT JOIN entities ON entities.entity_id = edges.target
            WHERE edges.source = ? ORDER BY edges.role""",
        (entity_id,),
    ).fetchall()

    referenced_by = connection.execute(
        """SELECT edges.role, edges.source AS entity_id, entities.entity_type
             FROM edges JOIN entities ON entities.entity_id = edges.source
            WHERE edges.target = ? ORDER BY edges.source""",
        (entity_id,),
    ).fetchall()

    return {"references": references, "referenced_by": referenced_by}


def read_versions(connection, base_id: str) -> List:
    """
    List every stored version of one name.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    base_id : str
        Name as originally uploaded.

    Returns
    -------
    rows : list of sqlite3.Row
        Versions, oldest first.
    """
    return connection.execute(
        """SELECT entity_id, version, owner, created_at, upload_id
             FROM entities WHERE base_id = ? ORDER BY version""",
        (base_id,),
    ).fetchall()


# =============================================================================
# Cross-entity views
# =============================================================================

def property_map_data(connection,
                      x_key: str,
                      y_key: str,
                      color_key: Optional[str] = None,
                      entity_type: str = "experiment",
                      filters: Optional[List[Filter]] = None):
    """
    Collect two or three values per entity for a cross-experiment scatter.

    This is the view that makes an archive worth more than the sum of its
    files: quantum yield against synthesis temperature over everything the group
    has ever measured, including values inherited from three references away.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    x_key, y_key : str
        Metadata keys or ``'result:<label>'`` for the two axes.
    color_key : str, optional
        Third key used to colour the points.
    entity_type : str, optional
        Kind of entry to plot.
    filters : list of Filter, optional
        Restrict the plotted set.

    Returns
    -------
    frame : pandas.DataFrame
        Columns ``entity_id``, ``x``, ``y`` and optionally ``color``, with rows
        lacking either axis dropped.
    """
    import pandas as pd

    rows, _ = search_entities(connection, entity_type=entity_type,
                              filters=filters, limit=None)

    records = []
    for row in rows:
        effective = json.loads(row["effective"])
        results = json.loads(row["results"])

        record = {"entity_id": row["entity_id"],
                  "x": _lookup(x_key, effective, results),
                  "y": _lookup(y_key, effective, results)}
        if color_key:
            record["color"] = _lookup(color_key, effective, results)

        records.append(record)

    frame = pd.DataFrame(records)

    return frame.dropna(subset=["x", "y"]) if not frame.empty else frame


def _lookup(key: str, effective: dict, results: dict):
    """
    Read one key from an entity's effective metadata or its results.

    Parameters
    ----------
    key : str
        Metadata key or ``'result:<label>'``.
    effective, results : dict
        The entity's decoded JSON columns.

    Returns
    -------
    value : Any or None
        The value, or None when the key is absent.
    """
    if key.startswith(RESULT_PREFIX):
        return results.get(key[len(RESULT_PREFIX):])

    return effective.get(key)


def list_axis_options(connection, entity_type: str = "experiment") -> List[str]:
    """
    List the keys a property-map axis can be set to.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_type : str, optional
        Kind of entry the axes describe.

    Returns
    -------
    options : list of str
        Numeric metadata keys, then result labels prefixed with ``result:``.
    """
    from pyKES.database.index_registry import read_metadata_keys

    numeric = [row["key"] for row in read_metadata_keys(connection,
                                                        entity_type=entity_type)
               if row["inferred_type"] == TYPE_NUMBER]

    results = [f"{RESULT_PREFIX}{row['label']}" for row in
               connection.execute("SELECT label FROM result_keys ORDER BY label")]

    return results + numeric


# =============================================================================
# Statistics
# =============================================================================

def database_statistics(connection) -> Dict[str, Any]:
    """
    Summarise the database for the home page.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    statistics : dict
        Counts by entity type, totals, and when the database last changed.
    """
    by_type = {row["entity_type"]: row["n"] for row in connection.execute(
        "SELECT entity_type, COUNT(*) AS n FROM entities GROUP BY entity_type")}

    totals = connection.execute(
        """SELECT COUNT(*) AS entities,
                  COUNT(payload_path) AS with_payload,
                  MAX(updated_at) AS last_change FROM entities"""
    ).fetchone()

    return {
        "by_type": by_type,
        "entities": totals["entities"],
        "with_payload": totals["with_payload"],
        "last_change": totals["last_change"],
        "edges": connection.execute(
            "SELECT COUNT(*) AS n FROM edges").fetchone()["n"],
        "metadata_keys": connection.execute(
            "SELECT COUNT(*) AS n FROM metadata_keys").fetchone()["n"],
        "uploads": connection.execute(
            "SELECT COUNT(*) AS n FROM uploads").fetchone()["n"],
    }


def read_uploads(connection, limit: int = 100) -> List:
    """
    List recent uploads for the admin page.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    limit : int, optional
        How many to return, most recent first.

    Returns
    -------
    rows : list of sqlite3.Row
        Upload records.
    """
    return connection.execute(
        "SELECT * FROM uploads ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
