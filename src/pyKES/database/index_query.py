"""
Reading the database index: search, facets, the reference graph and statistics.

An entity's own metadata is a JSON column; everything it inherits lives in the
`contributions` table, one row per contributing entry per field. A field is
named by the *kind of entry that owns it* rather than by the route that reached
it, so ``finished_semiconductor/Dopants [mol%]`` is one filter whether the
semiconductor was reached through an ordinary catalyst batch or a modified one.

That makes such a field set-valued, and a filter on it **existential**: it asks
whether *some* contributing entry satisfies it. Two filters over one kind may
therefore be satisfied by two different entries — one precursor chemical from
Merck and a different one 99.95% pure — which is usually the question being
asked. `Filter.match_same` requires one entry to satisfy all of them instead.

Metadata keys are user-supplied strings that arrive from spreadsheet headers, so
they are never interpolated into SQL. Both a JSON path and a contributions key
are bound as parameters, which is what keeps a column called
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
from pyKES.database.index_schema import ROLE_PATH_SEPARATOR, TYPE_KEY_SEPARATOR


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

# Between the values of a field several entries contributed to.
MULTIPLE_VALUE_SEPARATOR = "; "

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
        ``key``, because the key already carries the kind of entry and stacking
        a second separator into it is how the escaping bug happened.
    match_same : bool, optional
        Whether this filter must be satisfied by the *same* contributing entry
        as the other filters over its kind. A filter on inherited metadata asks
        whether *some* contributor satisfies it, so ``Supplier = Merck`` and
        ``Purity > 99.9`` over three precursor chemicals match an entry where
        one chemical is from Merck and a different one is 99.95% pure. That is
        usually the question — "a sample involving something from Merck and
        something very pure" — and where it is not, this asks for one chemical
        that is both.
    """

    key: str
    operator: str
    value: Any
    sub_key: Optional[str] = None
    match_same: bool = False

    def contributor_type(self) -> Optional[str]:
        """
        Name the kind of entry this filter reads, or None for an own field.

        Returns
        -------
        contributor_type : str or None
            Kind of entry the values belong to.
        """
        if self.key in COLUMN_FILTERS or self.key.startswith(RESULT_PREFIX):
            return None

        return split_type_key(self.key)[0]


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
    Render a stored key's prefix, or a reference route, as a readable name.

    A key's prefix is one entity type — ``'finished_semiconductor'`` becomes
    ``'Finished semiconductor'``. The same function renders the routes kept
    for the entry page, which are still several roles long:
    ``'catalyst_batch/finished_semiconductor'`` becomes
    ``'Catalyst batch › Finished semiconductor'``.

    Parameters
    ----------
    role_path : str or None
        Stored prefix or route, or None for an entity's own field.

    Returns
    -------
    label : str
        Readable name, empty for an own field.
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

    # Named after the kind of entry that owns the field, which is what makes it
    # one name however the graph reached that entry.
    return f"{leaf} [{display_role_path(role_path).lower()}]" if role_path else leaf


def split_type_key(key: str) -> Tuple[Optional[str], str]:
    """
    Separate a stored key into the kind of entry that owns it and the field.

    ``'finished_semiconductor/Dopants [mol%]'`` splits into that kind and that
    field; an entity's own key has no kind. The check is for the separator
    itself rather than for a non-empty prefix, because a key with no separator
    partitions into the *whole key* as its prefix — which reads an own field as
    though it were inherited, and silently answers a search with nothing.

    Parameters
    ----------
    key : str
        Stored metadata key.

    Returns
    -------
    contributor_type : str or None
        Kind of entry that owns the field, or None for an own field.
    field_name : str
        The field, escaped as stored.
    """
    if TYPE_KEY_SEPARATOR not in key:
        return None, key

    contributor_type, _, field_name = key.partition(TYPE_KEY_SEPARATOR)

    return contributor_type or None, field_name


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

    # A field with several contributors reads as a list. Joined the way a
    # reference cell is written, rather than as JSON, so the table shows
    # `Merck; Alfa` instead of `["Merck", "Alfa"]`.
    if isinstance(value, (list, tuple)):
        return MULTIPLE_VALUE_SEPARATOR.join(format_value(item) for item in value)

    if isinstance(value, dict):
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


def build_expression(key: str, sub_key: Optional[str] = None) -> Tuple[str, list]:
    """
    Render the SQL expression an own-metadata or result key addresses.

    Only for values held on the entity itself. Inherited values live in the
    contributions table and are reached with `build_contribution_predicate`.

    Parameters
    ----------
    key : str
        Column name, own metadata key, or ``'result:<label>'``.
    sub_key : str, optional
        Name inside a mapping-valued field.

    Returns
    -------
    expression : str
        SQL fragment with a placeholder for the JSON path where needed.
    parameters : list
        Values to bind for that fragment.
    """
    if key in COLUMN_FILTERS:
        return COLUMN_FILTERS[key], []

    if key.startswith(RESULT_PREFIX):
        return "json_extract(results, ?)", [json_path(key[len(RESULT_PREFIX):])]

    return "json_extract(metadata, ?)", [json_path(key, sub_key)]


def build_comparison(expression: str, operator: str, value: Any) -> Tuple[str, list]:
    """
    Render one comparison against an already-built expression.

    Parameters
    ----------
    expression : str
        SQL fragment holding the value.
    operator : {'between', 'in', 'contains', 'equals'}
        How ``value`` is compared.
    value : Any
        ``(low, high)`` for ``between``, a list for ``in``, otherwise a scalar.

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
    if operator == "between":
        low, high = value
        return f"CAST({expression} AS REAL) BETWEEN ? AND ?", [low, high]

    if operator == "in":
        options = list(value)
        if not options:
            # An empty selection means "no constraint", not "match nothing".
            return "1 = 1", []
        placeholders = ", ".join("?" for _ in options)
        return f"{expression} IN ({placeholders})", options

    if operator == "contains":
        return f"CAST({expression} AS TEXT) LIKE ?", [f"%{value}%"]

    if operator == "equals":
        return f"{expression} = ?", [value]

    raise ValueError(f"Unsupported filter operator '{operator}'.")


def contribution_condition(search_filter: Filter, alias: str,
                           entity_type: Optional[str]) -> Tuple[str, list]:
    """
    Render the part of a contributions lookup that identifies one field's rows.

    The kind of entry being searched leads the condition, matching the index,
    because without it the planner drives from the per-entity index and probes
    once per entity.

    Parameters
    ----------
    search_filter : Filter
        Filter addressing inherited metadata.
    alias : str
        Table alias the condition is written against.
    entity_type : str or None
        Kind of entry being searched.

    Returns
    -------
    condition : str
        SQL fragment.
    parameters : list
        Values to bind.
    """
    field_name = split_type_key(search_filter.key)[1]

    # A range reads the numeric column, which is indexed; everything else reads
    # the text. Storing both is what makes a range a lookup rather than a scan.
    column = f"{alias}.number" if search_filter.operator == "between" \
        else f"{alias}.value"
    comparison, values = build_comparison(column, search_filter.operator,
                                          contribution_value(search_filter))

    scope = f"{alias}.entity_type = ? AND " if entity_type else ""
    parameters = ([entity_type] if entity_type else []) + [
        search_filter.contributor_type(), field_name, search_filter.sub_key or ""]

    return (f"{scope}{alias}.contributor_type = ? AND {alias}.key = ? "
            f"AND {alias}.sub_key = ? AND {comparison}", parameters + values)


def contribution_value(search_filter: Filter) -> Any:
    """
    Render a filter's value the way the contributions table stores it.

    Values are held as text, so an equality or membership test has to compare
    against text; a boolean stored as ``'true'`` would never match Python's
    ``True``.

    Parameters
    ----------
    search_filter : Filter
        Filter whose value is being compared.

    Returns
    -------
    value : Any
        The value, rendered as stored.
    """
    if search_filter.operator == "between":
        return search_filter.value

    if search_filter.operator == "in":
        return [stored_text(item) for item in search_filter.value]

    return stored_text(search_filter.value)


def stored_text(value: Any) -> str:
    """
    Render one value as the contributions table holds it.

    Parameters
    ----------
    value : Any
        Value from a filter widget.

    Returns
    -------
    text : str
        Its stored form.
    """
    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def build_contribution_predicate(filters: List[Filter],
                                 entity_type: Optional[str] = None) -> Tuple[str, list]:
    """
    Render filters over one kind of contributing entry as one predicate.

    Filters that need not agree on a contributor become a semi-join each: the
    question is whether *some* precursor chemical is from Merck, not which. A
    semi-join rather than a correlated ``EXISTS`` because it lets SQLite read
    the value index once instead of probing per entity — 26 ms against 78 ms
    over ten thousand entries.

    Filters marked ``match_same`` are folded into a single ``EXISTS`` anchored
    on one contributor, so all of them have to hold of the same entry. That one
    has to be correlated: which contributor satisfies the first condition is
    exactly what the rest depend on.

    Parameters
    ----------
    filters : list of Filter
        Filters over one contributor type, all with the same ``match_same``.
    entity_type : str, optional
        Kind of entry being searched.

    Returns
    -------
    predicate : str
        SQL fragment against ``entities``.
    parameters : list
        Values to bind.
    """
    if not filters:
        return "1 = 1", []

    if not filters[0].match_same:
        clauses, parameters = [], []
        for search_filter in filters:
            condition, values = contribution_condition(search_filter, "c",
                                                       entity_type)
            clauses.append(f"entities.entity_id IN (SELECT c.entity_id "
                           f"FROM contributions AS c WHERE {condition})")
            parameters.extend(values)
        return " AND ".join(clauses), parameters

    # The first condition is the semi-join, so the value index picks the
    # candidate contributors; the rest are required of that same contributor and
    # probe the primary key. Anchoring on a row that already satisfies something
    # rather than on any row of the right kind is worth 28 ms against 237 ms.
    anchor, values = contribution_condition(filters[0], "anchor", entity_type)
    clauses, parameters = [], list(values)

    for index, search_filter in enumerate(filters[1:]):
        field_name = split_type_key(search_filter.key)[1]
        column = f"m{index}.number" if search_filter.operator == "between" \
            else f"m{index}.value"
        comparison, compared = build_comparison(
            column, search_filter.operator, contribution_value(search_filter))

        clauses.append(
            f"EXISTS (SELECT 1 FROM contributions AS m{index} "
            f"WHERE m{index}.entity_id = anchor.entity_id "
            f"AND m{index}.contributor = anchor.contributor "
            f"AND m{index}.key = ? AND m{index}.sub_key = ? AND {comparison})")
        parameters.extend([field_name, search_filter.sub_key or ""] + compared)

    required = (" AND " + " AND ".join(clauses)) if clauses else ""

    return (f"entities.entity_id IN (SELECT anchor.entity_id FROM contributions "
            f"AS anchor WHERE {anchor}{required})", parameters)


def build_predicate(search_filter: Filter,
                   entity_type: Optional[str] = None) -> Tuple[str, list]:
    """
    Render one filter as a SQL predicate.

    Parameters
    ----------
    search_filter : Filter
        Predicate to render.
    entity_type : str, optional
        Kind of entry being searched, which narrows an inherited lookup to the
        slice of the index that holds it.

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
    if search_filter.contributor_type() is not None:
        return build_contribution_predicate([search_filter], entity_type)

    expression, parameters = build_expression(search_filter.key,
                                              search_filter.sub_key)
    comparison, values = build_comparison(expression, search_filter.operator,
                                          search_filter.value)

    return comparison, parameters + values


def group_filters(filters: List[Filter],
                  entity_type: Optional[str] = None) -> List[Tuple[str, list]]:
    """
    Render a whole filter set, folding together those that must agree.

    Parameters
    ----------
    filters : list of Filter
        Every active filter.
    entity_type : str, optional
        Kind of entry being searched.

    Returns
    -------
    predicates : list of tuple
        ``(sql, parameters)`` for each predicate.
    """
    same, independent = {}, []

    for search_filter in filters:
        contributor_type = search_filter.contributor_type()

        if contributor_type is not None and search_filter.match_same:
            same.setdefault(contributor_type, []).append(search_filter)
        else:
            independent.append(search_filter)

    predicates = [build_predicate(search_filter, entity_type)
                  for search_filter in independent]
    predicates.extend(build_contribution_predicate(group, entity_type)
                      for group in same.values())

    return predicates


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
            "OR search_text LIKE ?)", [pattern, pattern, pattern, pattern])


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

    for predicate, values in group_filters(filters or [], entity_type):
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


def read_inherited_values(connection, entity_ids: List[str],
                          keys: List[str]) -> Dict[tuple, list]:
    """
    Read the inherited values of a page of entities, in one query.

    A field may have several contributors, so a cell holds a list rather than a
    value. One query for the whole page rather than one per row: fifty rows and
    ten chosen columns would otherwise be five hundred lookups.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_ids : list of str
        Entities on this page.
    keys : list of str
        Type-qualified keys wanted.

    Returns
    -------
    values : dict
        ``{(entity_id, key): [value, ...]}``, contributors in a stable order.
    """
    if not entity_ids or not keys:
        return {}

    wanted = {}
    for key in keys:
        wanted[split_type_key(key)] = key

    entity_places = ", ".join("?" for _ in entity_ids)
    field_places = ", ".join("?" for _ in wanted)

    rows = connection.execute(
        f"""SELECT entity_id, contributor_type, contributor, key, sub_key,
                   value, number
            FROM contributions
            WHERE entity_id IN ({entity_places})
              AND key IN ({field_places})
            ORDER BY contributor, sub_key""",
        list(entity_ids) + [field for _, field in wanted],
    ).fetchall()

    values = {}
    for row in rows:
        key = wanted.get((row["contributor_type"], row["key"]))
        if key is None:
            continue

        held = row["number"] if row["number"] is not None else row["value"]
        if row["sub_key"]:
            held = f"{row['sub_key']}={held}"

        values.setdefault((row["entity_id"], key), []).append(held)

    return values


def rows_to_frame(rows: List, columns: Optional[List[str]] = None,
                  connection=None):
    """
    Flatten search results into a table for display.

    Parameters
    ----------
    rows : list of sqlite3.Row
        Search results.
    columns : list of str, optional
        Own-metadata, inherited and result keys to include as columns, in order.
        Result keys carry the ``result:`` prefix.
    connection : sqlite3.Connection, optional
        Open connection, needed only when an inherited column was chosen.

    Returns
    -------
    frame : pandas.DataFrame
        One row per entity, indexed by nothing so Streamlit can select rows.
        The chosen columns come directly after the identifier and the fixed
        context columns follow them, so a column somebody just asked for is
        visible without scrolling a wide table sideways.
    """
    labels = column_labels(list(columns or []))
    inherited_keys = [column for column in labels
                      if TYPE_KEY_SEPARATOR in column
                      and not column.startswith(RESULT_PREFIX)]

    inherited = read_inherited_values(
        connection, [row["entity_id"] for row in rows], inherited_keys) \
        if connection is not None else {}

    records = []

    for row in rows:
        own = json.loads(row["metadata"])
        results = json.loads(row["results"])

        record = {"Entity ID": row["entity_id"]}

        for column, (header, _) in labels.items():
            record[header] = cell_value(column, row, own, results, inherited)

        record.update({"Kind": display_entity_type(row["entity_type"]),
                       "Group": row["display_group"], "Owner": row["owner"]})
        records.append(record)

    return arrow_safe_frame(records)


def cell_value(column: str, row, own: dict, results: dict, inherited: dict):
    """
    Read one cell of the results table.

    Parameters
    ----------
    column : str
        Chosen key.
    row : sqlite3.Row
        The entity's row.
    own, results : dict
        Its own metadata and its mapped results.
    inherited : dict
        Inherited values for the whole page, from `read_inherited_values`.

    Returns
    -------
    value : Any
        A single value, or every contributor's value where a field has several
        — which is what an inherited field now is.
    """
    if column.startswith(RESULT_PREFIX):
        return results.get(column[len(RESULT_PREFIX):])

    if TYPE_KEY_SEPARATOR not in column:
        return own.get(column)

    held = inherited.get((row["entity_id"], column))

    if not held:
        return None

    return held[0] if len(held) == 1 else held


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
    key = keys if isinstance(keys, str) else list(keys)[0]
    contributor_type, field_name = split_type_key(key)

    if contributor_type is None:
        path = json_path(key, sub_key)
        clause = "WHERE entity_type = ?" if entity_type else ""
        row = connection.execute(
            f"""SELECT MIN(CAST(json_extract(metadata, ?) AS REAL)) AS low,
                       MAX(CAST(json_extract(metadata, ?) AS REAL)) AS high
                FROM entities {clause}""",
            [path, path] + ([entity_type] if entity_type else []),
        ).fetchone()
    else:
        # Leads with the kind of entry being searched, matching the index, so
        # this is a covering read of one slice rather than a probe per entity:
        # 1.7 ms against 97.8 ms, paid once per numeric facet on every page
        # load.
        scope = "entity_type = ? AND " if entity_type else ""
        row = connection.execute(
            f"""SELECT MIN(number) AS low, MAX(number) AS high FROM contributions
                WHERE {scope}contributor_type = ? AND key = ? AND sub_key = ?""",
            ([entity_type] if entity_type else [])
            + [contributor_type, field_name, sub_key or ""],
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

    for row in rows:
        facet = build_facet(connection, row, entity_type, schemas)
        if facet is not None:
            facets.append(facet)

    # Sorting by the kind as well as the depth keeps one entry's fields
    # together, so the page can put them under its name.
    depths = contributor_depths(connection, entity_type)

    return sorted(facets, key=lambda facet: (reference_depth(facet, depths),
                                             facet.role_path or ""))


def build_facet(connection, row, entity_type: Optional[str],
                schemas: Dict[str, Any]) -> Optional[Facet]:
    """
    Choose the widget one field gets, from the type the registry observed.

    One registry row is one filter. It used to take several — a field reached
    through a modified catalyst batch was registered at a second depth and had
    to be folded back — but a key names the kind of entry that owns the field
    rather than the route to it, so the duplicates no longer arise.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    row : sqlite3.Row
        Registry row for the field.
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
    sample = json.loads(row["distinct_sample"] or "[]")
    common = dict(label=row["leaf_name"], role_path=row["role_path"] or None)

    if row["inferred_type"] == TYPE_MAPPING:
        return mapping_facet(connection, row, entity_type, schemas, common)

    if row["inferred_type"] == TYPE_NUMBER:
        bounds = numeric_bounds(connection, row["key"], entity_type)
        if bounds[0] == bounds[1]:
            return None
        return Facet(row["key"], kind="range", bounds=bounds, **common)

    if row["inferred_type"] == TYPE_BOOLEAN:
        return Facet(row["key"], kind="select", options=[True, False], **common)

    if len(sample) <= MULTISELECT_MAXIMUM_OPTIONS:
        return Facet(row["key"], kind="select",
                     options=sorted(str(value) for value in sample), **common)

    return Facet(row["key"], kind="contains", **common)


def mapping_facet(connection, row, entity_type: Optional[str],
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
    row : sqlite3.Row
        Registry row for the field.
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
    names = json.loads(row["sub_keys"] or "[]")
    if not names:
        return None

    bounds = {name: numeric_bounds(connection, row["key"], entity_type, name)
              for name in names}

    declared = mapping_field_schema(row["leaf_name"], schemas)

    return Facet(row["key"], kind="mapping", options=names, sub_bounds=bounds,
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


def contributor_depths(connection, entity_type: Optional[str] = None) -> Dict[str, int]:
    """
    Find how far away each kind of contributing entry usually sits.

    A key no longer carries its route, so the depth that orders the filter
    sidebar has to come from the data. The shortest route to each kind is used,
    which is what puts the catalyst batch above the semiconductor above the
    precursor chemicals.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_type : str, optional
        Kind of entry being searched.

    Returns
    -------
    depths : dict
        ``{contributor_type: shortest depth}``.
    """
    clause = ("WHERE entity_id IN (SELECT entity_id FROM entities WHERE entity_type = ?)"
              if entity_type else "")

    rows = connection.execute(
        f"""SELECT contributor_type, MIN(depth) AS depth FROM contribution_sources
            {clause} GROUP BY contributor_type""",
        [entity_type] if entity_type else [],
    ).fetchall()

    return {row["contributor_type"]: row["depth"] for row in rows}


def reference_depth(facet: Facet, depths: Optional[Dict[str, int]] = None) -> int:
    """
    Count how many references away a facet's field lives.

    Parameters
    ----------
    facet : Facet
        Facet to place.
    depths : dict, optional
        Shortest depth per contributor type, from `contributor_depths`. Without
        it every inherited facet sorts equal, which is only enough for a test.

    Returns
    -------
    depth : int
        0 for the entity's own metadata, otherwise how far the owning kind of
        entry sits from it.
    """
    if not facet.role_path:
        return 0

    return (depths or {}).get(facet.role_path, 1)


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
        carrying the role and the other entity. A role may appear more than
        once, since one field may name several entries, so a row is identified
        by role *and* entity rather than by role alone.
    """
    references = connection.execute(
        """SELECT edges.role, edges.target AS entity_id, edges.resolved,
                  entities.entity_type
             FROM edges LEFT JOIN entities ON entities.entity_id = edges.target
            WHERE edges.source = ?
            ORDER BY edges.role, edges.ordinal, edges.target""",
        (entity_id,),
    ).fetchall()

    referenced_by = connection.execute(
        """SELECT edges.role, edges.source AS entity_id, entities.entity_type
             FROM edges JOIN entities ON entities.entity_id = edges.source
            WHERE edges.target = ? ORDER BY edges.source, edges.role""",
        (entity_id,),
    ).fetchall()

    return {"references": references, "referenced_by": referenced_by}


def read_contributions(connection, entity_id: str) -> List:
    """
    Read everything one entry inherits, with where each value came from.

    What the entry page shows: the value, the entry that owns it, that entry's
    kind — which is what the field is named after — and the routes it was
    reached by, which no longer name anything but are still worth seeing.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entry in question.

    Returns
    -------
    rows : list of sqlite3.Row
        Ordered nearest contributor first, then by field.
    """
    return connection.execute(
        """SELECT contributions.contributor, contributions.contributor_type,
                  contribution_sources.role_paths, contribution_sources.depth,
                  contributions.key, contributions.sub_key,
                  contributions.value, contributions.number
             FROM contributions
             JOIN contribution_sources
               ON contribution_sources.entity_id = contributions.entity_id
              AND contribution_sources.contributor = contributions.contributor
            WHERE contributions.entity_id = ?
            ORDER BY contribution_sources.depth, contributions.contributor_type,
                     contributions.contributor, contributions.key,
                     contributions.sub_key""",
        (entity_id,),
    ).fetchall()


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

    axes = [key for key in (x_key, y_key, color_key) if key]
    inherited = read_inherited_values(connection, [row["entity_id"] for row in rows],
                                      [key for key in axes
                                       if TYPE_KEY_SEPARATOR in key
                                       and not key.startswith(RESULT_PREFIX)])

    records = []
    for row in rows:
        own = json.loads(row["metadata"])
        results = json.loads(row["results"])

        record = {"entity_id": row["entity_id"],
                  "x": _lookup(x_key, row, own, results, inherited),
                  "y": _lookup(y_key, row, own, results, inherited)}
        if color_key:
            record["color"] = _lookup(color_key, row, own, results, inherited)

        records.append(record)

    frame = pd.DataFrame(records)

    return frame.dropna(subset=["x", "y"]) if not frame.empty else frame


def _lookup(key: str, row, own: dict, results: dict, inherited: dict):
    """
    Read one key for one point of the property map.

    Only fields with a single contributor are offered as axes
    (`list_axis_options`), so there is one value to read. Averaging several
    would put a point at a number no experiment has, which is easy to misread
    and impossible to notice.

    Parameters
    ----------
    key : str
        Metadata key or ``'result:<label>'``.
    row : sqlite3.Row
        The entity's row.
    own, results : dict
        Its own metadata and its mapped results.
    inherited : dict
        Inherited values for the whole set, from `read_inherited_values`.

    Returns
    -------
    value : Any or None
        The value, or None when the key is absent.
    """
    if key.startswith(RESULT_PREFIX):
        return results.get(key[len(RESULT_PREFIX):])

    if TYPE_KEY_SEPARATOR not in key:
        return own.get(key)

    held = inherited.get((row["entity_id"], key))

    return held[0] if held else None


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
        Numeric keys with exactly one contributor, then result labels prefixed
        with ``result:``.
    """
    from pyKES.database.index_registry import read_metadata_keys

    shared = multi_contributor_keys(connection, entity_type)

    # A point whose x is an average of three precursor chemicals sits at a
    # number no experiment has, and nothing on the chart would say so. Such a
    # field is left off the axis menu rather than silently aggregated.
    numeric = [row["key"] for row in read_metadata_keys(connection,
                                                        entity_type=entity_type)
               if row["inferred_type"] == TYPE_NUMBER and row["key"] not in shared]

    results = [f"{RESULT_PREFIX}{row['label']}" for row in
               connection.execute("SELECT label FROM result_keys ORDER BY label")]

    return results + numeric


def multi_contributor_keys(connection, entity_type: Optional[str] = None) -> set:
    """
    Find the fields that more than one contributing entry can supply.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_type : str, optional
        Kind of entry being searched.

    Returns
    -------
    keys : set of str
        Type-qualified keys for which at least one entry has two contributors.
    """
    scope = "AND entity_type = ?" if entity_type else ""

    rows = connection.execute(
        f"""SELECT contributor_type, key FROM contributions
            WHERE sub_key = '' {scope}
            GROUP BY entity_id, contributor_type, key
            HAVING COUNT(DISTINCT contributor) > 1""",
        [entity_type] if entity_type else [],
    ).fetchall()

    return {f"{row['contributor_type']}{TYPE_KEY_SEPARATOR}{row['key']}"
            for row in rows}


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
