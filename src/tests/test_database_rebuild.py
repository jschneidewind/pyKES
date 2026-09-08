"""
Tests for rebuilding the index from the retained uploads.

Keeping every upload verbatim buys exactly one guarantee: the database can be
reconstructed from the original files. That guarantee is what a schema change
and a bad deployment both fall back on, and it is only as good as the rebuild
being *faithful* — a rebuild that returns a different database has not
restored anything, it has quietly replaced it.

The properties asserted here are the ones that were silently untrue: an
upload has to be re-read the way it was read the first time, which means the
entity type, the identifier column, the worksheet and the reference
declarations the operator chose have to be recorded rather than guessed; the
schemas still have to *read* the values even though history is deliberately
not re-validated; the chronology has to survive; and a rebuild that fails
part-way through has to leave the index alone.
"""

import json
import sqlite3

import pandas as pd
import pytest

from pyKES.database.entity_schema import load_entity_schemas
from pyKES.database.index_ingest import (
    ingest_entity_sheet,
    rebuild_index,
    recover_entity_types,
    resolve_stored_path,
)
from pyKES.database.index_schema import IndexPaths, open_index


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def paths(tmp_path):
    """Filesystem layout for one throwaway index."""
    return IndexPaths(root=tmp_path)


@pytest.fixture
def connection(paths):
    """Open, initialised index database."""
    handle = open_index(paths)
    yield handle
    handle.close()


@pytest.fixture
def schemas():
    """The schemas the group actually ships."""
    return load_entity_schemas()


def write_sheet(directory, name, rows):
    """Write one metadata sheet and return its path."""
    path = directory / name
    pd.DataFrame(rows).to_excel(path, index=False)

    return path


def entity_types(connection):
    """Count entries by kind, which is what a mis-typed rebuild destroys."""
    return {row["entity_type"]: row["entries"] for row in connection.execute(
        "SELECT entity_type, COUNT(*) AS entries FROM entities GROUP BY 1")}


def edge_count(connection):
    return connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]


def metadata_of(connection, entity_id):
    row = connection.execute("SELECT metadata FROM entities WHERE entity_id = ?",
                             (entity_id,)).fetchone()

    return json.loads(row["metadata"])


# =============================================================================
# An upload is re-read the way it was read
# =============================================================================

def test_a_sheet_keyed_on_another_column_survives_a_rebuild(connection, paths,
                                                            tmp_path):
    """
    The upload form lets the operator name the identifier column. Guessing
    `Experiment` instead of recording it did not merely mis-read that sheet:
    the rebuild raised, and it raised *after* the old code had committed its
    deletes, so the whole index was gone.
    """
    sheet = write_sheet(tmp_path, "batches.xlsx",
                        [{"Batch": "B-1", "Notes": "first"},
                         {"Batch": "B-2", "Notes": "second"}])
    ingest_entity_sheet(connection, paths, sheet, "other_entity", "alice",
                        identifier_column="Batch", schemas={})

    rebuild_index(connection, paths, schemas={})

    assert sorted(row["entity_id"] for row in connection.execute(
        "SELECT entity_id FROM entities")) == ["B-1", "B-2"]


def test_entity_types_survive_a_rebuild(connection, paths, tmp_path, schemas):
    """
    Without the recorded type every sheet upload came back as `other_entity`,
    so every batch, semiconductor and precursor collapsed into one bucket:
    unfindable by kind, checked against no schema, and reported as a wrong-kind
    reference everywhere it was used.
    """
    semiconductors = write_sheet(tmp_path, "semis.xlsx", [
        {"Experiment": "SEMI-1", "Precursor Chemicals": "EA-1",
         "Catalyst material": "SrTiO3", "Synthesis route": "Flux",
         "Synthesis temperature [°C]": 1150}])
    ingest_entity_sheet(connection, paths, semiconductors,
                        "finished_semiconductor", "alice", schemas=schemas)

    before = entity_types(connection)

    rebuild_index(connection, paths, schemas=schemas)

    assert entity_types(connection) == before == {"finished_semiconductor": 1}


def test_sheet_references_survive_a_rebuild(connection, paths, tmp_path, schemas):
    """
    A sheet's reference columns are declared by the schema at upload time and
    were not recorded, so a rebuild re-read the sheet with no declarations at
    all. For the group's own chain — batch to semiconductor to precursor —
    that is the entire reference graph, and losing it silently collapses every
    inherited-metadata filter to nothing.
    """
    semiconductors = write_sheet(tmp_path, "semis.xlsx", [
        {"Experiment": "SEMI-1", "Precursor Chemicals": "EA-1",
         "Catalyst material": "SrTiO3", "Synthesis route": "Flux",
         "Synthesis temperature [°C]": 1150}])
    ingest_entity_sheet(connection, paths, semiconductors,
                        "finished_semiconductor", "alice", schemas=schemas)

    batches = write_sheet(tmp_path, "batches.xlsx", [
        {"Experiment": "BATCH-1", "Finished Semiconductor": "SEMI-1",
         "Loading method [photodeposition/wet impregnation]": "photodeposition"}])
    ingest_entity_sheet(
        connection, paths, batches, "catalyst_batch", "alice",
        reference_instructions=schemas["catalyst_batch"].reference_instructions(),
        schemas=schemas)

    before = edge_count(connection)
    assert before == 1

    rebuild_index(connection, paths, schemas=schemas)

    assert edge_count(connection) == before


# =============================================================================
# Reading is not validating
# =============================================================================

def test_mapping_fields_and_derived_scalars_survive_a_rebuild(connection, paths,
                                                              tmp_path, schemas):
    """
    History is deliberately not re-validated: a field made required today
    would otherwise make every file accepted yesterday un-rebuildable, which
    destroys the very guarantee the upload store exists for. But validating
    and *reading* shared one argument, so declining the first also declined
    the second — a mapping came back as the raw text `900=0.5; 1150=10`
    instead of a mapping, and every scalar derived from it simply vanished.
    """
    sheet = write_sheet(tmp_path, "semis.xlsx", [
        {"Experiment": "SEMI-1", "Precursor Chemicals": "EA-1",
         "Catalyst material": "SrTiO3", "Synthesis route": "Flux",
         "Synthesis temperature [°C]": 1150,
         "Dopants [mol%]": "Ir=0.02; Cr=0.03",
         "Temperature steps [°C and h]": "900=0.5; 1150=10"}])
    ingest_entity_sheet(connection, paths, sheet, "finished_semiconductor",
                        "alice", schemas=schemas)

    before = metadata_of(connection, "SEMI-1")
    assert before["Dopants [mol%]"] == {"Ir": 0.02, "Cr": 0.03}
    assert before["Peak temperature [°C]"] == 1150

    rebuild_index(connection, paths, schemas=schemas)

    assert metadata_of(connection, "SEMI-1") == before


def test_a_rebuild_does_not_re_validate_history(connection, paths, tmp_path,
                                                schemas):
    """
    The other half of the same property: a file that today's schema would
    refuse must still rebuild. Here the sheet omits a required field.
    """
    sheet = write_sheet(tmp_path, "semis.xlsx",
                        [{"Experiment": "SEMI-1", "Catalyst material": "SrTiO3"}])
    ingest_entity_sheet(connection, paths, sheet, "finished_semiconductor",
                        "alice", schemas={})

    rebuild_index(connection, paths, schemas=schemas)

    assert entity_types(connection) == {"finished_semiconductor": 1}


# =============================================================================
# Chronology
# =============================================================================

def test_the_chronology_survives_a_rebuild(connection, paths, tmp_path, schemas):
    """
    Every timestamp used to be reset to the moment of the rebuild, which makes
    the admin upload log and every `ORDER BY updated_at` meaningless — and
    those are what somebody consults to work out what changed and when.
    """
    sheet = write_sheet(tmp_path, "semis.xlsx", [
        {"Experiment": "SEMI-1", "Precursor Chemicals": "EA-1",
         "Catalyst material": "SrTiO3", "Synthesis route": "Flux",
         "Synthesis temperature [°C]": 1150}])
    ingest_entity_sheet(connection, paths, sheet, "finished_semiconductor",
                        "alice", schemas=schemas)

    connection.execute("UPDATE uploads SET uploaded_at = '2020-01-02T03:04:05+00:00'")
    connection.commit()

    rebuild_index(connection, paths, schemas=schemas)

    uploaded_at = connection.execute(
        "SELECT uploaded_at FROM uploads").fetchone()["uploaded_at"]
    created_at = connection.execute(
        "SELECT created_at FROM entities").fetchone()["created_at"]

    assert uploaded_at == "2020-01-02T03:04:05+00:00"
    assert created_at == "2020-01-02T03:04:05+00:00"


# =============================================================================
# A rebuild that fails leaves the index alone
# =============================================================================

def test_an_interrupted_rebuild_changes_nothing(connection, paths, tmp_path,
                                                schemas):
    """
    The deletes were committed before any re-ingestion, so a failure part-way
    through — a corrupt file, a full disk, a restart — left a half-built index
    *and* an upload log that no longer listed the files needed to finish it.
    The retained files were still on disk, but nothing knew what they were.
    """
    first = write_sheet(tmp_path, "first.xlsx",
                        [{"Experiment": "SEMI-1", "Precursor Chemicals": "EA-1",
                          "Catalyst material": "SrTiO3",
                          "Synthesis route": "Flux",
                          "Synthesis temperature [°C]": 1150}])
    ingest_entity_sheet(connection, paths, first, "finished_semiconductor",
                        "alice", schemas=schemas)

    second = write_sheet(tmp_path, "second.xlsx",
                         [{"Experiment": "SEMI-2", "Precursor Chemicals": "EA-2",
                           "Catalyst material": "BaTaO2N",
                           "Synthesis route": "Flux",
                           "Synthesis temperature [°C]": 950}])
    ingest_entity_sheet(connection, paths, second, "finished_semiconductor",
                        "alice", schemas=schemas)

    entities_before = sorted(row["entity_id"] for row in connection.execute(
        "SELECT entity_id FROM entities"))
    uploads_before = connection.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]

    # The second retained file disappears, so re-ingesting it raises.
    stored = connection.execute(
        "SELECT stored_path FROM uploads ORDER BY id DESC LIMIT 1").fetchone()
    resolve_stored_path(paths, stored["stored_path"]).unlink()

    with pytest.raises(Exception):
        rebuild_index(connection, paths, schemas=schemas)

    assert sorted(row["entity_id"] for row in connection.execute(
        "SELECT entity_id FROM entities")) == entities_before
    assert connection.execute(
        "SELECT COUNT(*) FROM uploads").fetchone()[0] == uploads_before


# =============================================================================
# Uploads stored before their options were recorded
# =============================================================================

def test_the_type_of_an_older_upload_is_recovered_from_its_entries(
        connection, paths, tmp_path, schemas):
    """
    Uploads stored before `ingest_options` existed recorded no type, and the
    entries they produced are the only remaining evidence of it — which is why
    it has to be read before the rebuild empties the table.
    """
    sheet = write_sheet(tmp_path, "semis.xlsx", [
        {"Experiment": "SEMI-1", "Precursor Chemicals": "EA-1",
         "Catalyst material": "SrTiO3", "Synthesis route": "Flux",
         "Synthesis temperature [°C]": 1150}])
    ingest_entity_sheet(connection, paths, sheet, "finished_semiconductor",
                        "alice", schemas=schemas)

    connection.execute("UPDATE uploads SET ingest_options = NULL")
    connection.commit()

    assert recover_entity_types(connection) == {1: "finished_semiconductor"}

    rebuild_index(connection, paths, schemas=schemas)

    assert entity_types(connection) == {"finished_semiconductor": 1}


def test_an_explicit_type_overrides_what_was_recorded(connection, paths,
                                                      tmp_path, schemas):
    """The argument exists to correct an upload ingested as the wrong kind."""
    sheet = write_sheet(tmp_path, "things.xlsx",
                        [{"Experiment": "THING-1", "Notes": "misfiled"}])
    ingest_entity_sheet(connection, paths, sheet, "other_entity", "alice",
                        schemas={})

    rebuild_index(connection, paths, {1: "stock_solution"}, schemas={})

    assert entity_types(connection) == {"stock_solution": 1}


def test_an_absolute_stored_path_still_resolves(connection, paths, tmp_path,
                                                schemas):
    """
    Rows written before the store became relative hold an absolute path, and
    an absolute path is wrong the moment the data root moves — which a restore
    into a scratch directory and a container mounting the data elsewhere both
    do.
    """
    sheet = write_sheet(tmp_path, "things.xlsx",
                        [{"Experiment": "THING-1", "Notes": "kept"}])
    ingest_entity_sheet(connection, paths, sheet, "other_entity", "alice",
                        schemas={})

    stored = connection.execute("SELECT stored_path FROM uploads").fetchone()
    connection.execute("UPDATE uploads SET stored_path = ?",
                       (f"/somewhere/else/uploads/{stored['stored_path']}",))
    connection.commit()

    rebuild_index(connection, paths, schemas={})

    assert entity_types(connection) == {"other_entity": 1}


# =============================================================================
# The whole reference structure, reproduced exactly
# =============================================================================

def test_a_rebuild_reproduces_the_inherited_metadata_exactly(connection, paths,
                                                             tmp_path, schemas):
    """
    The strongest form of the guarantee, and the one a schema migration
    depends on: after a rebuild the reference graph and every inherited value
    computed from it are the same rows, not merely the same counts.

    `created_at`, `updated_at` and `payload_sha256` are deliberately not
    compared. The first two now follow the upload's own timestamp rather than
    the moment each entry happened to be written, and HDF5 payload bytes are
    not reproducible — nothing reads that digest, and a rebuild rewrites the
    payloads and their digests together.
    """
    precursors = write_sheet(tmp_path, "precursors.xlsx",
                             [{"Experiment": "EA-1", "Supplier": "Aldrich"}])
    ingest_entity_sheet(connection, paths, precursors, "precursor_chemical",
                        "alice", schemas={})

    semiconductors = write_sheet(tmp_path, "semis.xlsx", [
        {"Experiment": "SEMI-1", "Precursor Chemicals": "EA-1",
         "Catalyst material": "SrTiO3", "Synthesis route": "Flux",
         "Synthesis temperature [°C]": 1150}])
    ingest_entity_sheet(
        connection, paths, semiconductors, "finished_semiconductor", "alice",
        reference_instructions=(
            schemas["finished_semiconductor"].reference_instructions()),
        schemas=schemas)

    batches = write_sheet(tmp_path, "batches.xlsx", [
        {"Experiment": "BATCH-1", "Finished Semiconductor": "SEMI-1",
         "Loading method [photodeposition/wet impregnation]": "photodeposition"}])
    ingest_entity_sheet(
        connection, paths, batches, "catalyst_batch", "alice",
        reference_instructions=schemas["catalyst_batch"].reference_instructions(),
        schemas=schemas)

    def structure():
        return {
            "edges": [tuple(row) for row in connection.execute(
                "SELECT * FROM edges ORDER BY source, role, target")],
            "contributions": [tuple(row) for row in connection.execute(
                "SELECT * FROM contributions ORDER BY rowid")],
        }

    before = structure()
    assert before["edges"] and before["contributions"]

    rebuild_index(connection, paths, schemas=schemas)

    assert structure() == before
