"""
The reference graph: edges between entities, and the metadata they carry.

A photocatalysis experiment names the catalyst batch it used, that batch names
the finished semiconductor it was deposited on, and that semiconductor names the
precursor chemicals it was made from. The metadata of the whole chain has to be
reachable from the experiment, so that a search for *"tests of samples
synthesised at 1150 °C and photodeposited at 365 nm"* finds it.

Two decisions make this work:

**Inherited keys are qualified by the role path that reached them.** An
experiment's own ``Temperature [°C]`` and a precursor's own ``Temperature [°C]``
become two distinct keys, so the merge is *incapable* of collision. A flat merge
would have to pick one and discard the other, which for a scientific record is
not an acceptable failure mode. The path also carries the provenance: reading
``catalyst_batch/finished_semiconductor/Synthesis temperature [°C]`` tells you
which entry the value came from.

**The merge is materialised, not resolved at query time.** Answering the same
question with a recursive CTE has been measured at 92 ms for a *single*
predicate and needs another join per additional one; reading a materialised
column answers the whole three-predicate query in 33 ms, or 2.7 ms with the hot
keys promoted. The cost is that descendants must be recomputed when an ancestor
changes — which takes 0.2 ms to find and is done here.
"""

import json
import re
from typing import Any, Dict, List, Optional

from pyKES.database.database_experiments import sanitize_key
from pyKES.database.index_schema import MAX_REFERENCE_DEPTH, ROLE_PATH_SEPARATOR


# =============================================================================
# What is not inherited
# =============================================================================

# Display hints belong to the entry that carries them: a precursor's plotting
# colour says nothing about an experiment made from it, and inheriting them
# clutters every descendant's facet list. They are already stored as columns.
NON_INHERITED_KEYS = ("color", "group")

# Separators between identifiers in a reference cell naming several entries.
# The same ones a mapping field uses, so there is one convention to learn.
REFERENCE_SEPARATORS = ";\n"


# =============================================================================
# Errors
# =============================================================================

class ReferenceError(ValueError):
    """Raised when a declared reference cannot be recorded."""


# =============================================================================
# Recording edges
# =============================================================================

def extract_references(metadata: Dict[str, Any],
                       reference_instructions: Dict[str, Any]) -> List[tuple]:
    """
    Read the references an entity declares out of its metadata.

    A reference is a metadata field whose *value* is another entry's id. Which
    fields those are is declared by the uploaded file rather than guessed:
    scanning every value for something that looks like an id would link a lot
    number to a precursor by accident, and the mistake would be invisible.

    One field may name several entries — a semiconductor made from three
    precursor chemicals — written the way a mapping field is written, separated
    by a semicolon or a newline. Two fields may also share a role. Neither used
    to be allowed, because two targets under one role produced identical
    qualified keys and the second silently overwrote the first; inherited
    metadata is now a set of contributions, so nothing overwrites anything.

    Parameters
    ----------
    metadata : dict
        The entity's own metadata.
    reference_instructions : dict
        ``{metadata_key: {'role': role_name}}`` as declared by the upload.

    Returns
    -------
    references : list of tuple
        ``(role, target_entity_id, ordinal)`` for every declared field that
        holds a non-empty value, in declaration order. Fields that are absent or
        blank are skipped, since a catalyst batch made without a coating simply
        has no coating reference.
    """
    references = []

    for metadata_key, instruction in reference_instructions.items():
        role = instruction.get("role", metadata_key)

        # Stored metadata keys are escaped, so a declared column name has to be
        # escaped the same way before it will match.
        target = metadata.get(sanitize_key(metadata_key))

        if target is None or (isinstance(target, str) and not target.strip()):
            continue

        for ordinal, entity_id in enumerate(split_reference_cell(target)):
            references.append((role, entity_id, ordinal))

    return references


def split_reference_cell(value: Any) -> List[str]:
    """
    Read a reference cell that may name more than one entry.

    Parameters
    ----------
    value : Any
        Cell contents, such as ``'EA-1; EA-2; EA-3'`` or a single identifier.

    Returns
    -------
    entity_ids : list of str
        The identifiers, in the order written, blanks dropped. Duplicates are
        removed: naming an entry twice in one cell is a typo, and recording it
        twice would say the entry was used twice.
    """
    parts = [part.strip() for part in
             re.split(f"[{REFERENCE_SEPARATORS}]", str(value))]

    return list(dict.fromkeys(part for part in parts if part))


def record_references(connection, source: str, references: List[tuple]) -> None:
    """
    Replace the edges of one entity with the references it declares.

    Edges are replaced rather than merged, so re-ingesting a corrected entry
    removes references its earlier version had.

    An edge to an entity that does not exist yet is recorded as unresolved
    rather than rejected: entries are uploaded in whatever order suits the
    people making them, and an experiment routinely arrives before the catalyst
    batch it names. `resolve_pending_references` promotes such edges when the
    target appears.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    source : str
        Entity the references belong to.
    references : list of tuple
        ``(role, target_entity_id, ordinal)``.

    Returns
    -------
    None : None
    """
    connection.execute("DELETE FROM edges WHERE source = ?", (source,))

    for role, target, ordinal in references:
        exists = connection.execute(
            "SELECT 1 FROM entities WHERE entity_id = ?", (target,)
        ).fetchone()

        connection.execute(
            """INSERT OR REPLACE INTO edges (source, role, target, ordinal, resolved)
               VALUES (?, ?, ?, ?, ?)""",
            (source, role, target, ordinal, 1 if exists else 0),
        )


def resolve_pending_references(connection, target: str) -> List[str]:
    """
    Mark edges pointing at a newly arrived entity as resolved.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    target : str
        Entity that has just been added.

    Returns
    -------
    sources : list of str
        Entities whose edges were promoted, and which therefore need their
        effective metadata recomputed.
    """
    rows = connection.execute(
        "SELECT source FROM edges WHERE target = ? AND resolved = 0", (target,)
    ).fetchall()

    connection.execute(
        "UPDATE edges SET resolved = 1 WHERE target = ? AND resolved = 0", (target,)
    )

    return [row["source"] for row in rows]


def read_dangling_references(connection) -> List:
    """
    List edges whose target is not in the database.

    A forward reference that never arrives is indistinguishable from a typo, so
    these have to be visible somewhere or they accumulate unnoticed.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    edges : list of sqlite3.Row
        Unresolved edges, with their source, role and target.
    """
    return connection.execute(
        """SELECT source, role, target FROM edges
           WHERE target NOT IN (SELECT entity_id FROM entities)
           ORDER BY source"""
    ).fetchall()


def read_reference_type_mismatches(connection,
                                   accepted: Dict[tuple, List[str]]) -> List:
    """
    List edges pointing at a kind of entry their field does not accept.

    A role may accept several kinds — an experiment's catalyst batch can be an
    ordinary batch or a modified one — and which kinds is declared in the
    schema. The check runs here, over resolved edges, rather than at upload:
    an entry may legitimately name a target that has not been uploaded yet, and
    refusing the upload for a fact not yet knowable would break the forward
    references the whole ingestion order depends on.

    A mismatch is reported, never corrected. The metadata still merges, because
    a wrong-kind reference is a labelling mistake and discarding the values
    would hide it instead of showing it.

    Recomputed from the schemas each time rather than stored, so editing the
    accepted kinds in a YAML file is reflected immediately.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    accepted : dict
        ``{(source_entity_type, role): [entity_type, ...]}``, from
        `entity_schema.accepted_types`.

    Returns
    -------
    mismatches : list of dict
        One entry per offending edge, naming the source, the role, the target,
        the kind the target actually is, and the kinds the field accepts.
    """
    rows = connection.execute(
        """SELECT edges.source, edges.role, edges.target,
                  source_entity.entity_type AS source_type,
                  target_entity.entity_type AS target_type
           FROM edges
           JOIN entities AS source_entity ON source_entity.entity_id = edges.source
           JOIN entities AS target_entity ON target_entity.entity_id = edges.target
           ORDER BY edges.source"""
    ).fetchall()

    mismatches = []

    for row in rows:
        allowed = accepted.get((row["source_type"], row["role"]))

        # A role nobody declared is not checked: an upload may declare its own
        # references, and inventing a constraint for it would refuse data the
        # group deliberately sent.
        if not allowed or row["target_type"] in allowed:
            continue

        mismatches.append({"source": row["source"], "role": row["role"],
                           "target": row["target"],
                           "target_type": row["target_type"],
                           "accepts": allowed})

    return mismatches


# =============================================================================
# Resolving inherited metadata
# =============================================================================

def read_own_metadata(connection, entity_id: str) -> Optional[Dict[str, Any]]:
    """
    Read one entity's own metadata, without anything inherited.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity to read.

    Returns
    -------
    metadata : dict or None
        The entity's own metadata, or None if it is not in the database.
    """
    row = connection.execute(
        "SELECT metadata FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()

    return json.loads(row["metadata"]) if row is not None else None


def read_outgoing_edges(connection, entity_id: str) -> List[tuple]:
    """
    Read the references one entity makes.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity whose edges are read.

    Returns
    -------
    references : list of tuple
        ``(role, target_entity_id)``, in the order the references were written.
        A list rather than a mapping because one field may name several entries
        — a semiconductor made from three precursor chemicals — so a role no
        longer identifies a single target.
    """
    rows = connection.execute(
        "SELECT role, target FROM edges WHERE source = ? ORDER BY role, ordinal, target",
        (entity_id,),
    ).fetchall()

    return [(row["role"], row["target"]) for row in rows]


def reachable_contributors(connection, entity_id: str,
                           cycles: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    Find every entry reachable from one entity, and how it was reached.

    A breadth-first walk rather than the recursive merge it replaces, because
    what matters now is the *set* of entries reached — an entry found twice by
    two routes is one contributor, not two. Both routes are kept for display;
    the shorter decides the depth.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity to walk from.
    cycles : list of str, optional
        Collects the entities at which a cycle was found, so the caller can
        report them instead of discovering a hang.

    Returns
    -------
    contributors : dict
        ``{contributor_id: {'depth': int, 'role_paths': [str, ...]}}``, not
        including the entity itself. An entry that is referenced but not yet
        uploaded contributes nothing and is simply absent.
    """
    contributors = {}
    frontier = [(entity_id, (), ())]

    for _ in range(MAX_REFERENCE_DEPTH):
        if not frontier:
            break

        next_frontier = []

        for source, visited, role_path in frontier:
            for role, target in read_outgoing_edges(connection, source):
                if target in visited or target == entity_id:
                    if cycles is not None:
                        cycles.append(target)
                    continue

                reached = role_path + (role,)
                path_text = ROLE_PATH_SEPARATOR.join(reached)
                known = contributors.get(target)

                if known is None:
                    contributors[target] = {"depth": len(reached),
                                            "role_paths": [path_text]}
                    next_frontier.append((target, visited + (source,), reached))
                    continue

                # Already reached by another route: record the route, keep the
                # shorter depth, and do not walk it again -- its own
                # contributors were collected the first time.
                if path_text not in known["role_paths"]:
                    known["role_paths"].append(path_text)
                known["depth"] = min(known["depth"], len(reached))

        frontier = next_frontier

    return contributors


def resolve_contributions(connection, entity_id: str,
                          cycles: Optional[List[str]] = None) -> List[dict]:
    """
    Collect every metadata value one entity inherits, with where it came from.

    Each row names the entry that owns the value and that entry's *kind*, which
    is what the value is filtered by: `Dopants [finished semiconductor]` is one
    column however the graph reached the semiconductor. The route is carried
    alongside for display only — nothing filters on it, which is exactly what
    makes an experiment referencing either an ordinary or a modified catalyst
    batch a non-event.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity to resolve.
    cycles : list of str, optional
        Collects entities at which a cycle was found.

    Returns
    -------
    contributions : list of dict
        One entry per contributing entry per field, ready for `store_contributions`.
        A mapping-valued field is expanded to one row per name inside it, so a
        dopant filter is an index lookup rather than a scan.
    """
    contributions = []

    for contributor, reached in reachable_contributors(connection, entity_id,
                                                       cycles).items():
        row = connection.execute(
            "SELECT entity_type, metadata FROM entities WHERE entity_id = ?",
            (contributor,),
        ).fetchone()

        if row is None:
            continue

        shared = {"contributor": contributor,
                  "contributor_type": row["entity_type"],
                  "role_paths": json.dumps(reached["role_paths"]),
                  "depth": reached["depth"]}

        for key, value in json.loads(row["metadata"]).items():
            if key in NON_INHERITED_KEYS or value is None:
                continue
            contributions.extend(expand_value(shared, key, value))

    return contributions


def contribution_sources(contributions: List[dict]) -> List[dict]:
    """
    Reduce resolved contributions to one row per contributing entry.

    How an entry was reached belongs to the pair of entries, not to each of the
    thirty fields it contributes, so it is stored once rather than repeated.

    Parameters
    ----------
    contributions : list of dict
        Rows from `resolve_contributions`.

    Returns
    -------
    sources : list of dict
        One per contributor, carrying its kind, its routes and its depth.
    """
    sources = {}

    for row in contributions:
        sources.setdefault(row["contributor"], {
            "contributor": row["contributor"],
            "contributor_type": row["contributor_type"],
            "role_paths": row["role_paths"], "depth": row["depth"]})

    return list(sources.values())


def expand_value(shared: dict, key: str, value: Any) -> List[dict]:
    """
    Turn one metadata value into the rows that hold it.

    Parameters
    ----------
    shared : dict
        Contributor fields every row of this contributor carries.
    key : str
        Stored metadata key, already escaped.
    value : Any
        The value, possibly a mapping of names to numbers.

    Returns
    -------
    rows : list of dict
        One row, or one per name for a mapping — which is what lets a filter on
        a single dopant use an index instead of reading every mapping.
    """
    if isinstance(value, dict):
        return [dict(shared, key=key, sub_key=str(name), **as_columns(item))
                for name, item in value.items()]

    return [dict(shared, key=key, sub_key="", **as_columns(value))]


def as_columns(value: Any) -> dict:
    """
    Split one value into its searchable text and its number, where it has one.

    Storing the number separately is what lets a range filter use an index
    rather than casting every row's text: measured at 9.5 ms against 43.8 ms
    over ten thousand entries.

    Parameters
    ----------
    value : Any
        Coerced metadata value.

    Returns
    -------
    columns : dict
        ``{'value': str, 'number': float or None}``.
    """
    if isinstance(value, bool):
        return {"value": "true" if value else "false", "number": float(value)}

    if isinstance(value, (int, float)):
        return {"value": str(value), "number": float(value)}

    if isinstance(value, (list, dict)):
        return {"value": json.dumps(value, ensure_ascii=False), "number": None}

    return {"value": str(value), "number": None}


def store_contributions(connection, entity_id: str) -> List[dict]:
    """
    Resolve one entity's inherited metadata and write it back.

    Rows are replaced rather than merged, so an entity whose references changed
    loses the contributions of the entries it no longer reaches.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity to recompute.

    Returns
    -------
    contributions : list of dict
        What was stored.
    """
    contributions = resolve_contributions(connection, entity_id)
    entity_type = connection.execute(
        "SELECT entity_type FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()["entity_type"]

    connection.execute("DELETE FROM contributions WHERE entity_id = ?", (entity_id,))
    connection.execute("DELETE FROM contribution_sources WHERE entity_id = ?",
                       (entity_id,))

    connection.executemany(
        """INSERT INTO contribution_sources
           (entity_id, contributor, contributor_type, role_paths, depth)
           VALUES (:entity_id, :contributor, :contributor_type, :role_paths, :depth)""",
        [dict(row, entity_id=entity_id)
         for row in contribution_sources(contributions)],
    )

    connection.executemany(
        """INSERT INTO contributions
           (entity_id, entity_type, contributor, contributor_type,
            key, sub_key, value, number)
           VALUES (:entity_id, :entity_type, :contributor, :contributor_type,
                   :key, :sub_key, :value, :number)""",
        [dict(row, entity_id=entity_id, entity_type=entity_type)
         for row in contributions],
    )

    connection.execute("UPDATE entities SET search_text = ? WHERE entity_id = ?",
                       (build_search_text(connection, entity_id, contributions),
                        entity_id))

    return contributions


def build_search_text(connection, entity_id: str, contributions: List[dict]) -> str:
    """
    Flatten everything an entity carries into one string for free-text search.

    Free text has to reach a note somebody typed into an unexpected column, and
    that is a substring question rather than a structured one. One column
    answers it with one `LIKE`, where searching the contributions table would
    need a join per term.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity being recomputed.
    contributions : list of dict
        Its inherited values, already resolved.

    Returns
    -------
    text : str
        Own keys and values plus every inherited value, separated by newlines.
    """
    own = json.loads(connection.execute(
        "SELECT metadata FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()["metadata"])

    parts = [f"{key} {value}" for key, value in own.items() if value is not None]
    parts.extend(row["value"] for row in contributions if row["value"])

    return "\n".join(parts)


# =============================================================================
# Invalidation
# =============================================================================

def find_dependents(connection, entity_id: str) -> List[str]:
    """
    List every entity that transitively references the given one.

    Correcting a precursor changes the effective metadata of every experiment
    descended from it, so this is what decides which rows a single edit touches.
    Measured at 0.2 ms over a twelve-thousand-entity graph.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity that changed.

    Returns
    -------
    dependents : list of str
        Entities to recompute, excluding ``entity_id`` itself.
    """
    rows = connection.execute(
        """WITH RECURSIVE dependent(node) AS (
               SELECT ?
               UNION
               SELECT edges.source FROM edges
               JOIN dependent ON edges.target = dependent.node)
           SELECT node FROM dependent WHERE node <> ?""",
        (entity_id, entity_id),
    ).fetchall()

    return [row["node"] for row in rows]


def recompute_entity_and_dependents(connection, entity_id: str) -> List[str]:
    """
    Recompute the inherited metadata of an entity and everything below it.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity that changed.

    Returns
    -------
    recomputed : list of str
        Every entity whose inherited metadata was rewritten, the entity itself
        first.
    """
    recomputed = [entity_id]
    store_contributions(connection, entity_id)

    for dependent in find_dependents(connection, entity_id):
        store_contributions(connection, dependent)
        recomputed.append(dependent)

    return recomputed


def find_cyclic_entities(connection) -> List[str]:
    """
    List entities whose reference chain contains a cycle.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    entities : list of str
        Entities at which resolution detected a cycle.
    """
    cycles = []

    for row in connection.execute("SELECT entity_id FROM entities"):
        reachable_contributors(connection, row["entity_id"], cycles=cycles)

    return sorted(set(cycles))
