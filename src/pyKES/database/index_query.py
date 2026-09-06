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
    """

    key: str
    operator: str
    value: Any


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

    return " › ".join(display_entity_type(role)
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


def json_path(key: str) -> str:
    """
    Build the JSON path addressing one metadata or result key.

    Parameters
    ----------
    key : str
        Metadata key, possibly qualified and possibly containing quotes.

    Returns
    -------
    path : str
        SQLite JSON path, bound as a parameter rather than interpolated.
    """
    return '$."' + key.replace('"', '\\"') + '"'


def build_expression(key: str) -> Tuple[str, list]:
    """
    Render the SQL expression a filter or axis addresses.

    Parameters
    ----------
    key : str
        Column name, qualified metadata key, or ``'result:<label>'``.

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

    return "json_extract(effective, ?)", [json_path(key)]


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
    expression, parameters = build_expression(search_filter.key)

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
    """
    import pandas as pd

    records = []
    for row in rows:
        effective = json.loads(row["effective"])
        results = json.loads(row["results"])

        record = {"Entity ID": row["entity_id"],
                  "Kind": display_entity_type(row["entity_type"]),
                  "Group": row["display_group"], "Owner": row["owner"]}

        for column in columns or []:
            if column.startswith(RESULT_PREFIX):
                record[display_key(column)] = results.get(
                    column[len(RESULT_PREFIX):])
            else:
                record[display_key(column)] = effective.get(column)

        records.append(record)

    return pd.DataFrame(records)


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
    kind : {'range', 'select', 'contains'}
        Widget to draw.
    options : list
        Choices for a ``select`` facet.
    bounds : tuple
        ``(minimum, maximum)`` for a ``range`` facet.
    role_path : str or None
        Reference path the key was inherited through, shown as provenance.
    """

    key: str
    label: str
    kind: str
    options: list = field(default_factory=list)
    bounds: tuple = ()
    role_path: Optional[str] = None


def numeric_bounds(connection, key: str, entity_type: Optional[str]) -> tuple:
    """
    Find the range a numeric metadata key spans.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    key : str
        Qualified metadata key.
    entity_type : str, optional
        Restrict to one kind of entry.

    Returns
    -------
    bounds : tuple
        ``(minimum, maximum)``, or ``(0.0, 0.0)`` when the key holds no numbers.
    """
    clause = "WHERE entity_type = ?" if entity_type else ""
    parameters = ([json_path(key), json_path(key)] +
                  ([entity_type] if entity_type else []))

    row = connection.execute(
        f"""SELECT MIN(CAST(json_extract(effective, ?) AS REAL)) AS low,
                   MAX(CAST(json_extract(effective, ?) AS REAL)) AS high
            FROM entities {clause}""",
        parameters,
    ).fetchone()

    if row["low"] is None:
        return (0.0, 0.0)

    return (float(row["low"]), float(row["high"]))


def build_facets(connection,
                 entity_type: Optional[str] = None,
                 minimum_occurrences: int = 1) -> List[Facet]:
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

    Returns
    -------
    facets : list of Facet
        Ordered by reference depth — the entry's own fields first, then fields
        one reference away, and so on — and by frequency within each depth.
        That is the order somebody narrowing a search thinks in: what was done
        in this experiment, then what it was made from.
    """
    from pyKES.database.index_registry import read_metadata_keys

    facets = []

    for row in read_metadata_keys(connection, entity_type=entity_type):
        if row["occurrences"] < minimum_occurrences or row["canonical_key"]:
            continue

        sample = json.loads(row["distinct_sample"] or "[]")

        if row["inferred_type"] == TYPE_NUMBER:
            bounds = numeric_bounds(connection, row["key"], entity_type)
            if bounds[0] == bounds[1]:
                continue
            facets.append(Facet(row["key"], row["leaf_name"], "range",
                                bounds=bounds, role_path=row["role_path"]))

        elif row["inferred_type"] == TYPE_BOOLEAN:
            facets.append(Facet(row["key"], row["leaf_name"], "select",
                                options=[True, False], role_path=row["role_path"]))

        elif len(sample) <= MULTISELECT_MAXIMUM_OPTIONS:
            facets.append(Facet(row["key"], row["leaf_name"], "select",
                                options=sorted(str(value) for value in sample),
                                role_path=row["role_path"]))

        else:
            facets.append(Facet(row["key"], row["leaf_name"], "contains",
                                role_path=row["role_path"]))

    return sorted(facets, key=reference_depth)


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
