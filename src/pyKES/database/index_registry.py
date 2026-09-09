"""
Metadata- and result-key registries for the database index.

The index stores scientific metadata as JSON rather than as typed columns, so
something has to remember which keys exist. These two registries do, and they
are maintained at ingestion rather than computed on demand: walking ``json_each``
over every entity to rebuild the key list has been measured at 152 ms, which is
too slow to pay on every page view and free to pay once per upload.

The registries are also where the two failure modes of a free-form metadata
schema become visible instead of silent:

* **Key drift** — ``Irradiance [mW/cm2]`` and ``Irradiance (mW/cm2)`` are two
  keys, created by two people typing two spreadsheet headers. ``canonical_key``
  lets an admin alias one onto the other.
* **Type conflict** — the same key arriving as a number in one upload and as
  text in another. ``inferred_type`` becomes ``'mixed'`` and the search page
  degrades that facet to a text filter rather than drawing a broken slider.

Neither is resolved by guessing. Guessing here would corrupt searches in a way
nobody would notice.
"""

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from pyKES.database.database_experiments import restore_key, sanitize_key
from pyKES.database.index_schema import ROLE_PATH_SEPARATOR, TYPE_KEY_SEPARATOR


# =============================================================================
# Type inference
# =============================================================================

# Names used in `metadata_keys.inferred_type`. 'mixed' is recorded when a key
# has been seen carrying more than one of the others.
TYPE_NUMBER = "number"
TYPE_TEXT = "text"
TYPE_BOOLEAN = "boolean"
TYPE_MAPPING = "mapping"
TYPE_MIXED = "mixed"

# How many distinct values are kept per key to build facet widgets from. Enough
# to tell a low-cardinality key (a multiselect) from a free-text one, small
# enough that the registry stays a few hundred kilobytes at ten thousand rows.
DISTINCT_SAMPLE_LIMIT = 25


# =============================================================================
# Value coercion
# =============================================================================

def coerce_index_value(value: Any) -> Any:
    """
    Reduce a metadata or result value to something JSON and SQLite can hold.

    Two conversions matter in practice and neither is optional. NumPy scalars
    are not JSON-serialisable, and a float NaN — which is what an empty Excel
    cell becomes by the time it reaches here — serialises to the bare token
    ``NaN``, which is not valid JSON and which SQLite's ``json_extract`` refuses
    to read. Both are turned into something storable rather than allowed to fail
    at query time, months later.

    Parameters
    ----------
    value : Any
        Raw value taken from an experiment's metadata or processed data.

    Returns
    -------
    coerced : Any
        A bool, int, float, str, list, dict or None.
    """
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (np.integer, np.floating)):
        value = value.item()

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    if isinstance(value, (int, float, str)):
        return value

    if isinstance(value, np.ndarray):
        return coerce_index_value(value.tolist())

    if isinstance(value, (list, tuple)):
        return [coerce_index_value(item) for item in value]

    if isinstance(value, dict):
        return {str(key): coerce_index_value(item) for key, item in value.items()}

    return str(value)


def coerce_index_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply `coerce_index_value` to every entry of a mapping.

    Parameters
    ----------
    mapping : dict
        Metadata or results dictionary as read from an experiment.

    Keys are escaped with `sanitize_key`, so a key of its own containing a
    slash survives being joined into a reference path and split back out. No
    key has to be rejected.

    Returns
    -------
    coerced : dict
        Mapping with escaped keys and storable values.
    """
    return {sanitize_key(key): coerce_index_value(value)
            for key, value in mapping.items()}


def infer_value_type(value: Any) -> str:
    """
    Classify a value for the purpose of choosing a facet widget.

    Parameters
    ----------
    value : Any
        Coerced metadata value.

    Returns
    -------
    type_name : str
        One of ``TYPE_NUMBER``, ``TYPE_BOOLEAN``, ``TYPE_MAPPING`` or
        ``TYPE_TEXT``.
    """
    if isinstance(value, bool):
        return TYPE_BOOLEAN

    if isinstance(value, (int, float)):
        return TYPE_NUMBER

    # Without this a set of dopant concentrations would be classified as text
    # and offered as a multiselect of stringified dictionaries.
    if isinstance(value, dict):
        return TYPE_MAPPING

    return TYPE_TEXT


def combine_types(existing: Optional[str], observed: str) -> str:
    """
    Merge a newly observed type into the type already recorded for a key.

    Parameters
    ----------
    existing : str or None
        Type recorded so far, or None for a key seen for the first time.
    observed : str
        Type of the value just seen.

    Returns
    -------
    combined : str
        ``observed`` for a new key, the shared type when they agree, and
        ``TYPE_MIXED`` when they disagree.
    """
    if existing is None:
        return observed

    if existing == observed:
        return existing

    return TYPE_MIXED


# =============================================================================
# Key naming
# =============================================================================

def split_qualified_key(key: str) -> tuple:
    """
    Separate a qualified metadata key into its role path and its leaf name.

    Inherited keys carry the path that reached them, so
    ``'catalyst_batch/finished_semiconductor/Synthesis temperature [°C]'``
    splits into that role path and ``'Synthesis temperature [°C]'``. An
    entity's own keys have no role path.

    The split is unambiguous because a stored key has its own slashes escaped:
    ``'Catalyst concentration [g/L]'`` is stored as
    ``'Catalyst concentration [g__SLASH__L]'`` and comes back out of here
    restored.

    Parameters
    ----------
    key : str
        Qualified or unqualified metadata key.

    Returns
    -------
    role_path : str or None
        Role path, or None when the key belongs to the entity itself.
    leaf_name : str
        The metadata key as it was written on the entity that owns it, with its
        own slashes restored.
    """
    if ROLE_PATH_SEPARATOR not in key:
        return None, restore_key(key)

    role_path, _, leaf_name = key.rpartition(ROLE_PATH_SEPARATOR)

    return role_path, restore_key(leaf_name)


def qualify_key(role: str, key: str) -> str:
    """
    Prefix an inherited key with the role that reached it.

    Parameters
    ----------
    role : str
        Role of the edge the value was inherited through.
    key : str
        Key as it appears on the referenced entity, itself possibly qualified.
        Already escaped, since it comes from that entity's stored metadata.

    Returns
    -------
    qualified : str
        ``'<role>/<key>'``.
    """
    return f"{role}{ROLE_PATH_SEPARATOR}{key}"


# =============================================================================
# Registry maintenance
# =============================================================================

def register_metadata_keys(connection, entity_id: str) -> None:
    """
    Record the keys one entity carries, own and inherited, in the registry.

    Called once per entity per ingestion. An own key is registered bare; an
    inherited one is registered under the *kind of entry that owns it* —
    `finished_semiconductor/Dopants [mol%]` — which is what makes it one filter
    however the graph reached that semiconductor.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity whose keys are registered.

    Returns
    -------
    None : None
        ``metadata_keys`` is updated in place.
    """
    row = connection.execute(
        "SELECT entity_type, metadata FROM entities WHERE entity_id = ?",
        (entity_id,),
    ).fetchone()

    if row is None:
        return

    for key, value in json.loads(row["metadata"]).items():
        if value is not None:
            register_key(connection, key, value, row["entity_type"])

    contributions = connection.execute(
        """SELECT contributor_type, key, sub_key, value, number
           FROM contributions WHERE entity_id = ?""",
        (entity_id,),
    ).fetchall()

    for contribution in contributions:
        register_key(connection,
                     f"{contribution['contributor_type']}{ROLE_PATH_SEPARATOR}{contribution['key']}",
                     registered_value(contribution), row["entity_type"],
                     sub_key=contribution["sub_key"])


def registered_value(contribution) -> Any:
    """
    Read a contribution row back as the value the registry should record.

    Parameters
    ----------
    contribution : sqlite3.Row
        One row of the contributions table.

    Returns
    -------
    value : Any
        The number where the row holds one, otherwise its text.
    """
    return contribution["number"] if contribution["number"] is not None \
        else contribution["value"]


def register_key(connection, key: str, value: Any, entity_type: Optional[str],
                 sub_key: str = "") -> None:
    """
    Record one key of one entity.

    Occurrence counts accumulate, observed types are merged, and a bounded
    sample of distinct values is kept so the search page can decide between a
    slider, a multiselect and a text box without querying the entities table.

    The entity types a key occurs on are recorded too: a key inherited by an
    experiment and a key of the same name on the batch itself are the same
    registry row, and the search page needs to know which kinds of entry offer
    it as a filter.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    key : str
        Stored key, bare for an entity's own field and type-qualified for an
        inherited one.
    value : Any
        The value observed.
    entity_type : str or None
        Kind of entry the key was observed on.
    sub_key : str, optional
        Name inside a mapping-valued field, recorded so the filter can offer it.

    Returns
    -------
    None : None
    """
    now = datetime.now(timezone.utc).isoformat()
    role_path, leaf_name = split_qualified_key(key)
    observed_type = TYPE_MAPPING if sub_key else infer_value_type(value)

    row = connection.execute(
        """SELECT inferred_type, distinct_sample, entity_types, sub_keys
           FROM metadata_keys WHERE key = ?""",
        (key,),
    ).fetchone()

    if row is None:
        connection.execute(
            """INSERT INTO metadata_keys
               (key, leaf_name, role_path, inferred_type, occurrences,
                first_seen, last_seen, distinct_sample, entity_types, sub_keys)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
            (key, leaf_name, role_path, observed_type, now, now,
             json.dumps([] if sub_key else [value]),
             json.dumps([entity_type] if entity_type else []),
             json.dumps([sub_key] if sub_key else [])),
        )
        return

    sample = json.loads(row["distinct_sample"] or "[]")
    if not sub_key and value not in sample and len(sample) < DISTINCT_SAMPLE_LIMIT:
        sample.append(value)

    # Sorted, like `sub_keys` below: the only reader is a `json_each`
    # membership join, so the order means nothing — but appended, it recorded
    # whichever order the kinds happened to arrive in. Both repair paths walk
    # the entries in a different order from the ingestions that first built the
    # registry, so the same archive came back with the same kinds listed
    # differently after a rebuild or after Admin's Rebuild Key Registry.
    entity_types = json.loads(row["entity_types"] or "[]")
    if entity_type and entity_type not in entity_types:
        entity_types = sorted(entity_types + [entity_type])

    sub_keys = json.loads(row["sub_keys"] or "[]")
    if sub_key and sub_key not in sub_keys:
        sub_keys = sorted(sub_keys + [sub_key])

    connection.execute(
        """UPDATE metadata_keys
           SET inferred_type = ?, occurrences = occurrences + 1,
               last_seen = ?, distinct_sample = ?, entity_types = ?, sub_keys = ?
           WHERE key = ?""",
        (combine_types(row["inferred_type"], observed_type), now,
         json.dumps(sample), json.dumps(entity_types), json.dumps(sub_keys), key),
    )


def register_result_keys(connection,
                         index_instructions: Dict[str, Any],
                         upload_id: Optional[int] = None) -> list:
    """
    Record the result labels an upload declares, and flag redefinitions.

    A label whose path differs from the one an earlier upload declared is not
    overwritten: both are kept and the label is marked ``conflicting``, because
    silently adopting the new path would change the meaning of a column for
    every entity already in the database.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    index_instructions : dict
        ``{label: {'result': path, 'unit': …, 'format': …}}`` as declared by the
        uploaded file.
    upload_id : int, optional
        Upload the declaration came from, recorded as the definer.

    Returns
    -------
    conflicts : list of str
        Labels whose declared path disagrees with the stored one.
    """
    now = datetime.now(timezone.utc).isoformat()
    conflicts = []

    for label, instruction in index_instructions.items():
        path = instruction.get("result")
        if path is None:
            continue

        row = connection.execute(
            "SELECT path FROM result_keys WHERE label = ?", (label,)
        ).fetchone()

        if row is None:
            connection.execute(
                """INSERT INTO result_keys
                   (label, path, unit, value_format, defined_by, conflicting, first_seen)
                   VALUES (?, ?, ?, ?, ?, 0, ?)""",
                (label, path, instruction.get("unit"), instruction.get("format"),
                 upload_id, now),
            )
            continue

        if row["path"] != path:
            conflicts.append(label)
            connection.execute(
                "UPDATE result_keys SET conflicting = 1 WHERE label = ?", (label,)
            )

    return conflicts


def rebuild_metadata_key_registry(connection) -> int:
    """
    Recompute the metadata-key registry from the entities table.

    The registry is normally maintained incrementally; this is the repair path,
    used after entities have been deleted or after a rebuild.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    key_count : int
        Number of distinct keys registered.
    """
    connection.execute("DELETE FROM metadata_keys")

    for row in connection.execute("SELECT entity_id FROM entities").fetchall():
        register_metadata_keys(connection, row["entity_id"])

    connection.commit()

    return connection.execute("SELECT COUNT(*) AS n FROM metadata_keys").fetchone()["n"]


def read_metadata_keys(connection,
                       leaf_name: Optional[str] = None,
                       entity_type: Optional[str] = None) -> list:
    """
    List registered metadata keys, optionally narrowed to one leaf or type.

    Searching by leaf name is what lets a user filter on ``Synthesis
    temperature [°C]`` without knowing it is reached through
    ``catalyst_batch/finished_semiconductor``. A leaf in a reference chain
    resolves to one key *per entity type*, not one overall, so a facet built for
    the experiment search should pass ``entity_type='experiment'`` and will then
    usually get a single path back.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    leaf_name : str, optional
        Restrict the result to keys with this leaf name.
    entity_type : str, optional
        Restrict the result to keys seen on entities of this type.

    Returns
    -------
    keys : list of sqlite3.Row
        Registry rows, most frequently occurring first.
    """
    clauses, parameters = [], []

    if leaf_name is not None:
        clauses.append("leaf_name = ?")
        parameters.append(leaf_name)

    if entity_type is not None:
        # entity_types is a JSON array, so membership is a json_each join.
        clauses.append("""EXISTS (SELECT 1 FROM json_each(metadata_keys.entity_types)
                                  WHERE json_each.value = ?)""")
        parameters.append(entity_type)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    return connection.execute(
        f"SELECT * FROM metadata_keys {where} ORDER BY occurrences DESC",
        parameters,
    ).fetchall()
