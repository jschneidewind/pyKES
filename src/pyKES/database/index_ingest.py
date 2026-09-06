"""
Ingestion of uploads into the database index.

Two kinds of upload arrive. An **HDF5 batch** is what the processing app
produces: a dataset holding a number of experiments with their metadata, raw
data and processed data. An **entity sheet** is an Excel or CSV listing entries
that carry metadata but no measurements — catalyst batches, semiconductors,
precursor chemicals — because the person who synthesised a precursor has no raw
traces and will never open the processing app, yet their metadata is what the
search depends on.

Both end up in the same `entities` table, and both go through the same steps:
hash the file, validate it, resolve name collisions, split out payloads, apply
the declared mappings, register keys, record references, and recompute anything
whose inherited metadata just changed.

Every uploaded file is stored verbatim and kept indefinitely, which is what
makes `rebuild_index` possible: the whole database can be reconstructed when the
index schema changes or a mapping turns out to have been wrong, without asking
anybody to upload anything again.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pyKES.database.database_experiments import ExperimentalDataset, Experiment
from pyKES.database.index_references import (
    extract_references,
    recompute_entity_and_dependents,
    record_references,
    resolve_pending_references,
    store_effective_metadata,
)
from pyKES.database.index_registry import (
    coerce_index_mapping,
    coerce_index_value,
    register_metadata_keys,
    register_result_keys,
)
from pyKES.database.index_schema import (
    DEFAULT_ENTITY_TYPE,
    ENTITY_TYPES,
    IndexPaths,
    VERSION_SUFFIX_SEPARATOR,
)
from pyKES.utilities.resolve_attributes import resolve_experiment_attributes


# =============================================================================
# Upload kinds and instruction keys
# =============================================================================

UPLOAD_KIND_HDF5 = "hdf5"
UPLOAD_KIND_ENTITY_SHEET = "entity_sheet"

# Entries of `ExperimentalDataset.plotting_instruction` an upload uses to
# declare how it maps into the database. Both are optional; a file that declares
# neither still ingests, it simply contributes no results and no references.
INDEX_INSTRUCTION_KEY = "index_instructions"
REFERENCE_INSTRUCTION_KEY = "reference_instructions"

# Existing files predate `index_instructions`, but the results table they
# already declare has exactly the shape it needs, so it is used as the default.
FALLBACK_INDEX_INSTRUCTION_KEY = "results_table_instructions"

# Metadata fields that describe the entity itself rather than the science.
IDENTITY_METADATA_KEYS = ("experiment_name", "raw_data_file", "Processed")

# Compression filter applied to payload files. Measured on a real dataset:
# about 1 ms per experiment, and a third off the bytes.
PAYLOAD_COMPRESSION = "gzip"

# Read in binary chunks of this size when hashing an upload, so a large file
# does not have to be held in memory to be identified.
HASH_CHUNK_BYTES = 1 << 20


# =============================================================================
# Errors and reports
# =============================================================================

class IngestionError(ValueError):
    """Raised when an upload cannot be ingested at all."""


@dataclass
class IngestionReport:
    """
    What one ingestion run did.

    Parameters
    ----------
    upload_id : int
        Row id of the stored upload.
    added, versioned, skipped : list of str
        Entity ids newly added, added under a version suffix because their name
        was taken, and skipped because the identical file was already ingested.
    recomputed : list of str
        Entities whose inherited metadata was rewritten as a consequence.
    result_conflicts : list of str
        Result labels this upload redefined with a different path.
    already_ingested : bool
        True when the file's hash was already known and nothing was done.
    """

    upload_id: int
    added: List[str] = field(default_factory=list)
    versioned: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    recomputed: List[str] = field(default_factory=list)
    result_conflicts: List[str] = field(default_factory=list)
    already_ingested: bool = False

    def summary(self) -> str:
        """
        Render the report as one line for the upload page.

        Returns
        -------
        description : str
            Human-readable summary.
        """
        if self.already_ingested:
            return "This file has already been ingested; nothing was changed."

        return (f"{len(self.added)} entries added "
                f"({len(self.versioned)} under a version suffix), "
                f"{len(self.recomputed)} recomputed, "
                f"{len(self.result_conflicts)} result conflicts.")


# =============================================================================
# Upload storage
# =============================================================================

def hash_file(file_path: Path) -> str:
    """
    Compute the SHA-256 of a file.

    Parameters
    ----------
    file_path : Path
        File to hash.

    Returns
    -------
    digest : str
        Hexadecimal digest, used both to detect a re-uploaded file and to name
        it in the upload store.
    """
    digest = hashlib.sha256()

    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)

    return digest.hexdigest()


def store_upload(connection,
                 paths: IndexPaths,
                 file_path: Path,
                 uploaded_by: str,
                 kind: str) -> tuple:
    """
    Record an upload and keep the file verbatim.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    paths : IndexPaths
        Filesystem layout of the database.
    file_path : Path
        Uploaded file, as staged by the application.
    uploaded_by : str
        Authenticated user the upload is attributed to.
    kind : str
        ``UPLOAD_KIND_HDF5`` or ``UPLOAD_KIND_ENTITY_SHEET``.

    Returns
    -------
    upload_id : int
        Row id of the upload.
    already_ingested : bool
        True when a file with this hash was already stored, in which case
        nothing is written.
    """
    file_path = Path(file_path)
    digest = hash_file(file_path)

    existing = connection.execute(
        "SELECT id FROM uploads WHERE sha256 = ?", (digest,)
    ).fetchone()
    if existing is not None:
        return existing["id"], True

    stored_path = paths.upload_directory / f"{digest}{file_path.suffix}"

    # A rebuild re-ingests straight out of the upload store, so the file is
    # already where it belongs and copying it would raise SameFileError.
    if stored_path.resolve() != file_path.resolve():
        shutil.copyfile(file_path, stored_path)

    cursor = connection.execute(
        """INSERT INTO uploads
           (sha256, filename, stored_path, byte_count, kind, uploaded_by, uploaded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (digest, file_path.name, str(stored_path), stored_path.stat().st_size,
         kind, uploaded_by, datetime.now(timezone.utc).isoformat()),
    )

    return cursor.lastrowid, False


# =============================================================================
# Entity identity
# =============================================================================

def allocate_entity_id(connection, base_id: str) -> tuple:
    """
    Choose the id a new entry is stored under, versioning on collision.

    The group's policy is that a name already in the database is never
    overwritten: both entries are kept, the newcomer under ``__v2``, ``__v3``
    and so on. Correcting a mistake is a deliberate edit by the owner or an
    admin, not a side effect of uploading a file twice.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    base_id : str
        Name the entry carries in its upload.

    Returns
    -------
    entity_id : str
        Id to store the entry under.
    version : int
        1 for a first occurrence, 2 upwards for a collision.
    """
    taken = connection.execute(
        "SELECT COUNT(*) AS n FROM entities WHERE base_id = ?", (base_id,)
    ).fetchone()["n"]

    if taken == 0:
        return base_id, 1

    version = taken + 1

    return f"{base_id}{VERSION_SUFFIX_SEPARATOR}{version}", version


# =============================================================================
# Mapping declarations
# =============================================================================

def read_index_instructions(dataset: ExperimentalDataset) -> Dict[str, Any]:
    """
    Find the result mapping an uploaded dataset declares.

    Falls back to ``results_table_instructions``, which existing files already
    carry in exactly the required shape, so a file written before this feature
    existed still contributes its results.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Uploaded dataset.

    Returns
    -------
    instructions : dict
        ``{label: {'result': path, …}}``; empty when the file declares neither.
    """
    instruction = dataset.plotting_instruction or {}

    if INDEX_INSTRUCTION_KEY in instruction:
        return instruction[INDEX_INSTRUCTION_KEY]

    return instruction.get(FALLBACK_INDEX_INSTRUCTION_KEY, {})


def read_reference_instructions(dataset: ExperimentalDataset) -> Dict[str, Any]:
    """
    Find the reference mapping an uploaded dataset declares.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Uploaded dataset.

    Returns
    -------
    instructions : dict
        ``{metadata_key: {'role': role}}``; empty when none is declared.
    """
    return (dataset.plotting_instruction or {}).get(REFERENCE_INSTRUCTION_KEY, {})


def apply_index_instructions(experiment: Experiment,
                             instructions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve an experiment's declared results into a flat mapping.

    Resolution is permissive: a path that does not resolve for this experiment
    is simply absent from its results, which is the correct behaviour when one
    batch mixes liquid-phase and gas-phase runs measuring different analytes.

    Parameters
    ----------
    experiment : Experiment
        Experiment the paths are resolved against.
    instructions : dict
        ``{label: {'result': path, …}}``.

    Returns
    -------
    results : dict
        ``{label: scalar}`` for every path that resolved to a scalar.
    """
    results = {}

    for label, instruction in instructions.items():
        path = instruction.get("result")
        if path is None:
            continue

        resolved = resolve_experiment_attributes(
            {"value": path}, experiment, mode="permissive"
        )
        value = coerce_index_value(resolved.get("value"))

        if isinstance(value, (int, float, str, bool)):
            results[label] = value

    return results


def split_identity_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop the bookkeeping fields from an experiment's metadata.

    ``experiment_name``, ``raw_data_file`` and ``Processed`` describe the record
    rather than the chemistry, and become columns or are discarded, so they are
    kept out of the searchable metadata.

    Parameters
    ----------
    metadata : dict
        Metadata as stored on the experiment.

    Returns
    -------
    scientific : dict
        Metadata without the identity fields.
    """
    return {key: value for key, value in metadata.items()
            if key not in IDENTITY_METADATA_KEYS}


# =============================================================================
# Payload writing
# =============================================================================

def write_payload(experiment: Experiment,
                  paths: IndexPaths,
                  entity_id: str) -> tuple:
    """
    Write one experiment out as a standalone, compressed HDF5 payload.

    The payload is a single-experiment `ExperimentalDataset`, so it loads back
    through `load_from_hdf5` unchanged and is independently a valid pyKES
    dataset — which is what makes "download this experiment and work on it
    locally" free.

    Parameters
    ----------
    experiment : Experiment
        Experiment to write.
    paths : IndexPaths
        Filesystem layout of the database.
    entity_id : str
        Id the payload is named after.

    Returns
    -------
    payload_path : str
        Path relative to the payload directory.
    byte_count : int
        Size of the written file.
    digest : str
        SHA-256 of the written file.
    """
    payload_path = paths.payload_directory / f"{entity_id}.h5"

    single = ExperimentalDataset(experiments={experiment.experiment_name: experiment})
    single.save_to_hdf5(str(payload_path), compression=PAYLOAD_COMPRESSION,
                        verbose=False)

    return (payload_path.name, payload_path.stat().st_size, hash_file(payload_path))


# =============================================================================
# Writing entities
# =============================================================================

def insert_entity(connection,
                  entity_id: str,
                  base_id: str,
                  version: int,
                  entity_type: str,
                  metadata: Dict[str, Any],
                  results: Dict[str, Any],
                  owner: str,
                  upload_id: int,
                  display_group: Optional[str] = None,
                  color: Optional[str] = None,
                  payload: Optional[tuple] = None,
                  provenance: Optional[Dict[str, Any]] = None) -> None:
    """
    Write one entity row.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id, base_id : str
        Stored id and the name it was uploaded under.
    version : int
        Version of this name, 1 unless the name collided.
    entity_type : str
        One of ``ENTITY_TYPES``.
    metadata, results : dict
        The entity's own metadata and its mapped scalar results.
    owner : str
        User the entry belongs to; only they and an admin may edit it.
    upload_id : int
        Upload the entry came from.
    display_group, color : str, optional
        Presentation hints carried over from the upload.
    payload : tuple, optional
        ``(path, byte_count, digest)`` for entries that have measurements.
    provenance : dict, optional
        The experiment's ``version`` dictionary.

    Returns
    -------
    None : None

    Raises
    ------
    IngestionError
        If ``entity_type`` is not one of the agreed types.
    """
    if entity_type not in ENTITY_TYPES:
        raise IngestionError(
            f"Unknown entity type '{entity_type}'; expected one of {ENTITY_TYPES}."
        )

    # Escaping is enforced here rather than trusted from every caller, so a key
    # containing a slash can never be stored raw and later mistaken for a
    # reference path. `sanitize_key` is idempotent, so coercing again costs
    # nothing on the ingestion path that has already done it.
    metadata = coerce_index_mapping(metadata)

    now = datetime.now(timezone.utc).isoformat()
    provenance = provenance or {}
    external = provenance.get("external_version") or {}
    payload_path, payload_bytes, payload_digest = payload or (None, None, None)

    connection.execute(
        """INSERT INTO entities
           (entity_id, base_id, version, entity_type, display_group, color, active,
            owner, created_at, updated_at, upload_id, payload_path, payload_bytes,
            payload_sha256, pykes_version, external_app, external_version,
            last_processed, metadata, results, effective)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (entity_id, base_id, version, entity_type, display_group, color,
         _as_active_flag(metadata.get("Active")), owner, now, now, upload_id,
         payload_path, payload_bytes, payload_digest,
         provenance.get("pykes_version"), external.get("app"),
         external.get("version"), provenance.get("last_processed"),
         json.dumps(metadata), json.dumps(results), json.dumps(metadata)),
    )


def _as_active_flag(value: Any) -> Optional[int]:
    """
    Interpret the ``Active`` metadata field as a flag.

    The field arrives as a real bool from some sheets and as the strings
    ``'True'`` / ``'False'`` from others, because it round-trips through Excel
    and HDF5. Both are accepted; anything else leaves the flag unset rather than
    guessing.

    Parameters
    ----------
    value : Any
        Raw ``Active`` value.

    Returns
    -------
    flag : int or None
        1, 0, or None when the value says nothing.
    """
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        return {"true": 1, "false": 0}.get(value.strip().lower())

    return None


def finalise_entity(connection,
                    entity_id: str,
                    metadata: Dict[str, Any],
                    reference_instructions: Dict[str, Any]) -> List[str]:
    """
    Record an entity's references and settle the inherited metadata around it.

    Three things have to happen once a row exists, and in this order: its own
    references are recorded, any entry that had been waiting for this one is
    promoted from unresolved, and everything affected is recomputed.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entity just written.
    metadata : dict
        Its own metadata, holding the reference fields.
    reference_instructions : dict
        Declaration of which metadata fields are references.

    Returns
    -------
    recomputed : list of str
        Entities whose effective metadata was rewritten.
    """
    record_references(connection, entity_id, extract_references(metadata,
                                                                reference_instructions))

    recomputed = recompute_entity_and_dependents(connection, entity_id)

    for waiting in resolve_pending_references(connection, entity_id):
        recomputed.extend(recompute_entity_and_dependents(connection, waiting))

    for affected in set(recomputed):
        row = connection.execute(
            "SELECT effective, entity_type FROM entities WHERE entity_id = ?",
            (affected,)).fetchone()
        register_metadata_keys(connection, json.loads(row["effective"]),
                               row["entity_type"])

    return sorted(set(recomputed))


# =============================================================================
# HDF5 batch ingestion
# =============================================================================

def ingest_hdf5_upload(connection,
                       paths: IndexPaths,
                       file_path: Path,
                       uploaded_by: str,
                       entity_type: str = "experiment") -> IngestionReport:
    """
    Ingest one HDF5 batch produced by the processing app.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    paths : IndexPaths
        Filesystem layout of the database.
    file_path : Path
        Uploaded HDF5 file.
    uploaded_by : str
        Authenticated user the entries are attributed to.
    entity_type : str, optional
        Type assigned to every experiment in the file.

    Returns
    -------
    report : IngestionReport
        What was added, versioned, recomputed and flagged.

    Raises
    ------
    IngestionError
        If the file holds no experiments.
    """
    dataset = ExperimentalDataset.load_from_hdf5(str(file_path))

    if not dataset.experiments:
        raise IngestionError(f"{Path(file_path).name} holds no experiments.")

    upload_id, already = store_upload(connection, paths, file_path,
                                      uploaded_by, UPLOAD_KIND_HDF5)
    report = IngestionReport(upload_id=upload_id, already_ingested=already)
    if already:
        return report

    index_instructions = read_index_instructions(dataset)
    reference_instructions = read_reference_instructions(dataset)
    report.result_conflicts = register_result_keys(connection, index_instructions,
                                                   upload_id)

    for name in sorted(dataset.experiments):
        experiment = dataset.experiments[name]
        entity_id, version = allocate_entity_id(connection, name)
        metadata = coerce_index_mapping(split_identity_metadata(experiment.metadata))

        insert_entity(
            connection,
            entity_id=entity_id,
            base_id=name,
            version=version,
            entity_type=entity_type,
            metadata=metadata,
            results=apply_index_instructions(experiment, index_instructions),
            owner=uploaded_by,
            upload_id=upload_id,
            display_group=experiment.group,
            color=experiment.color,
            payload=write_payload(experiment, paths, entity_id),
            provenance=experiment.version,
        )

        report.added.append(entity_id)
        if version > 1:
            report.versioned.append(entity_id)

        report.recomputed.extend(
            finalise_entity(connection, entity_id, metadata, reference_instructions)
        )

    connection.execute("UPDATE uploads SET entity_count = ? WHERE id = ?",
                       (len(report.added), upload_id))
    connection.commit()

    report.recomputed = sorted(set(report.recomputed))

    return report


# =============================================================================
# Entity sheet ingestion
# =============================================================================

def ingest_entity_sheet(connection,
                        paths: IndexPaths,
                        file_path: Path,
                        entity_type: str,
                        uploaded_by: str,
                        reference_instructions: Optional[Dict[str, Any]] = None,
                        identifier_column: str = "Experiment",
                        sheet_name: str = "Sheet1") -> IngestionReport:
    """
    Ingest a sheet of entries that carry metadata but no measurements.

    This is the route for catalyst batches, semiconductors and precursor
    chemicals: one row per entry, an identifier column, and whatever metadata
    columns the sheet happens to have. New columns need no migration — they
    appear in the entry's metadata and in the key registry, and a facet for them
    shows up by itself.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    paths : IndexPaths
        Filesystem layout of the database.
    file_path : Path
        Uploaded ``.xlsx`` or ``.csv``.
    entity_type : str
        Type assigned to every row, one of ``ENTITY_TYPES``.
    uploaded_by : str
        Authenticated user the entries are attributed to.
    reference_instructions : dict, optional
        ``{column: {'role': role}}`` declaring which columns are references.
    identifier_column : str, optional
        Column holding the entry's id.
    sheet_name : str, optional
        Worksheet to read from an Excel file.

    Returns
    -------
    report : IngestionReport
        What was added, versioned, recomputed and flagged.

    Raises
    ------
    IngestionError
        If the identifier column is missing.
    """
    file_path = Path(file_path)
    frame = (pd.read_csv(file_path) if file_path.suffix.lower() == ".csv"
             else pd.read_excel(file_path, sheet_name=sheet_name))

    if identifier_column not in frame.columns:
        raise IngestionError(
            f"{file_path.name} has no '{identifier_column}' column; "
            f"found {list(frame.columns)}."
        )

    upload_id, already = store_upload(connection, paths, file_path,
                                      uploaded_by, UPLOAD_KIND_ENTITY_SHEET)
    report = IngestionReport(upload_id=upload_id, already_ingested=already)
    if already:
        return report

    reference_instructions = reference_instructions or {}

    for row in frame.to_dict(orient="records"):
        base_id = str(row[identifier_column]).strip()
        entity_id, version = allocate_entity_id(connection, base_id)
        metadata = coerce_index_mapping(
            {key: value for key, value in row.items() if key != identifier_column}
        )

        insert_entity(
            connection,
            entity_id=entity_id,
            base_id=base_id,
            version=version,
            entity_type=entity_type,
            metadata=metadata,
            results={},
            owner=uploaded_by,
            upload_id=upload_id,
            display_group=metadata.get("group"),
            color=metadata.get("color"),
        )

        report.added.append(entity_id)
        if version > 1:
            report.versioned.append(entity_id)

        report.recomputed.extend(
            finalise_entity(connection, entity_id, metadata, reference_instructions)
        )

    connection.execute("UPDATE uploads SET entity_count = ? WHERE id = ?",
                       (len(report.added), upload_id))
    connection.commit()

    report.recomputed = sorted(set(report.recomputed))

    return report


# =============================================================================
# Editing and permissions
# =============================================================================

def may_edit(connection, entity_id: str, user: str, is_admin: bool = False) -> bool:
    """
    Decide whether a user may correct or delete an entry.

    The group's rule is owner or admin. Everyone can read everything.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entry in question.
    user : str
        Authenticated user.
    is_admin : bool, optional
        Whether that user holds the admin role.

    Returns
    -------
    permitted : bool
        True when the user owns the entry or is an admin. An entry that does not
        exist is not editable by anyone.
    """
    row = connection.execute(
        "SELECT owner FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()

    if row is None:
        return False

    return is_admin or row["owner"] == user


def update_entity_metadata(connection,
                           entity_id: str,
                           updates: Dict[str, Any],
                           user: str,
                           is_admin: bool = False,
                           reference_instructions: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Correct an entry's metadata in place, and recompute what depends on it.

    Editing an entry that many others reference — a precursor, say — changes the
    effective metadata of every descendant. That is the point of the reference
    structure, and it is why the return value names everything that moved.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    entity_id : str
        Entry to correct.
    updates : dict
        Metadata fields to set. A value of None removes the field.
    user : str
        Authenticated user making the change.
    is_admin : bool, optional
        Whether that user holds the admin role.
    reference_instructions : dict, optional
        Declaration of which fields are references, so that changing one
        re-points the edge.

    Returns
    -------
    recomputed : list of str
        Entities whose effective metadata was rewritten.

    Raises
    ------
    PermissionError
        If the user neither owns the entry nor is an admin.
    """
    if not may_edit(connection, entity_id, user, is_admin):
        raise PermissionError(
            f"{user} may not edit {entity_id}: only its owner or an admin can."
        )

    row = connection.execute(
        "SELECT metadata FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    metadata = json.loads(row["metadata"])

    for key, value in coerce_index_mapping(updates).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value

    connection.execute(
        "UPDATE entities SET metadata = ?, active = ?, updated_at = ? WHERE entity_id = ?",
        (json.dumps(metadata), _as_active_flag(metadata.get("Active")),
         datetime.now(timezone.utc).isoformat(), entity_id),
    )

    recomputed = finalise_entity(connection, entity_id, metadata,
                                 reference_instructions or {})
    connection.commit()

    return recomputed


# =============================================================================
# Rebuilding
# =============================================================================

def rebuild_index(connection,
                  paths: IndexPaths,
                  entity_type_by_upload: Optional[Dict[int, str]] = None) -> List[IngestionReport]:
    """
    Rebuild the whole index from the retained uploads.

    This is why every upload is kept verbatim. When the index schema changes, or
    a mapping turns out to have been wrong, the database is reconstructed from
    the original files rather than from anybody's memory. Measured at roughly
    2.5 minutes for ten thousand experiments.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    paths : IndexPaths
        Filesystem layout of the database.
    entity_type_by_upload : dict, optional
        ``{upload_id: entity_type}`` for uploads whose type is not
        ``'experiment'``; entity sheets in particular.

    Returns
    -------
    reports : list of IngestionReport
        One report per re-ingested upload, in upload order.
    """
    uploads = connection.execute(
        "SELECT * FROM uploads ORDER BY id"
    ).fetchall()
    entity_type_by_upload = entity_type_by_upload or {}

    connection.execute("DELETE FROM edges")
    connection.execute("DELETE FROM entities")
    connection.execute("DELETE FROM metadata_keys")
    connection.execute("DELETE FROM result_keys")
    connection.execute("DELETE FROM uploads")
    connection.commit()

    reports = []
    for upload in uploads:
        stored_path = Path(upload["stored_path"])

        if upload["kind"] == UPLOAD_KIND_HDF5:
            entity_type = entity_type_by_upload.get(upload["id"], "experiment")
            reports.append(ingest_hdf5_upload(connection, paths, stored_path,
                                              upload["uploaded_by"], entity_type))
        else:
            entity_type = entity_type_by_upload.get(upload["id"], DEFAULT_ENTITY_TYPE)
            reports.append(ingest_entity_sheet(connection, paths, stored_path,
                                               entity_type, upload["uploaded_by"]))

    return reports
