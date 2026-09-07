"""
Tests for the database index: ingestion, the reference graph and the registries.

The reference structure is tested against synthetic entities with known
metadata, so every assertion has a ground truth that does not depend on files
which only exist on one machine. The cases that matter are the ones the design
identified as able to fail silently: a forward reference resolved by a later
upload, a diamond, a cycle, a repeated role, and a metadata key that arrives
with two different types.
"""

import json
import sqlite3

import pandas as pd
import pytest

from pyKES.database.database_experiments import Experiment, ExperimentalDataset
from pyKES.database.index_ingest import (
    IngestionError,
    allocate_entity_id,
    apply_index_instructions,
    finalise_entity,
    ingest_entity_sheet,
    ingest_hdf5_upload,
    insert_entity,
    may_edit,
    rebuild_index,
    read_index_instructions,
    update_entity_metadata,
)
from pyKES.database.index_references import (
    ReferenceError,
    extract_references,
    find_cyclic_entities,
    find_dependents,
    read_dangling_references,
    read_reference_type_mismatches,
    reachable_contributors,
    split_reference_cell,
)
from pyKES.database.index_query import (
    Filter,
    read_contributions,
    search_entities,
)
from pyKES.database.index_registry import (
    TYPE_MIXED,
    TYPE_NUMBER,
    coerce_index_value,
    read_metadata_keys,
    register_metadata_keys,
    split_qualified_key,
)
from pyKES.database.index_schema import (
    INDEX_SCHEMA_VERSION,
    IndexPaths,
    MAX_REFERENCE_DEPTH,
    open_index,
    read_index_schema_version,
)

import numpy as np


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


def add_entity(connection, entity_id, entity_type, metadata, owner="tester"):
    """Insert a bare entity without going through an upload."""
    insert_entity(connection, entity_id=entity_id, base_id=entity_id, version=1,
                  entity_type=entity_type, metadata=metadata, results={},
                  owner=owner, upload_id=None)


def inherited(connection, entity_id, key):
    """
    Read what one entity inherits under one type-qualified key.

    A field is named by the kind of entry that owns it and may have several
    contributors, so this is a list — which is the whole point of the change
    these tests cover.
    """
    contributor_type, _, field_name = key.partition("/")

    return [row["number"] if row["number"] is not None else row["value"]
            for row in read_contributions(connection, entity_id)
            if row["contributor_type"] == contributor_type
            and row["key"] == field_name]


def build_chain(connection, reference_instructions):
    """
    Build the worked example: experiment -> batch -> semiconductor.

    Returns the reference instructions used, so a test can re-apply them.
    """
    add_entity(connection, "BC-2", "finished_semiconductor",
               {"Synthesis temperature [degC]": 1150})
    add_entity(connection, "ABC-12", "catalyst_batch",
               {"Photodeposition wavelength [nm]": 360, "Finished Semiconductor": "BC-2"})
    add_entity(connection, "ABC-67", "experiment",
               {"Irradiance [mW/cm2]": 50, "Catalyst Batch": "ABC-12"})

    for entity_id, metadata in [
        ("BC-2", {"Synthesis temperature [degC]": 1150}),
        ("ABC-12", {"Photodeposition wavelength [nm]": 360,
                    "Finished Semiconductor": "BC-2"}),
        ("ABC-67", {"Irradiance [mW/cm2]": 50, "Catalyst Batch": "ABC-12"}),
    ]:
        finalise_entity(connection, entity_id, metadata, reference_instructions)

    connection.commit()


CHAIN_REFERENCES = {
    "Catalyst Batch": {"role": "catalyst_batch"},
    "Finished Semiconductor": {"role": "finished_semiconductor"},
}


# =============================================================================
# Schema
# =============================================================================

def test_index_initialises_and_records_its_schema_version(connection):
    assert read_index_schema_version(connection) == INDEX_SCHEMA_VERSION

    tables = {row["name"] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    assert {"entities", "edges", "uploads", "metadata_keys", "result_keys"} <= tables


def test_unknown_entity_type_is_refused(connection):
    with pytest.raises(IngestionError, match="Unknown entity type"):
        add_entity(connection, "X-1", "not_a_type", {})


# =============================================================================
# Value coercion
# =============================================================================

@pytest.mark.parametrize("value, expected", [
    (np.float64(1.5), 1.5),
    (np.int32(7), 7),
    (np.bool_(True), True),
    (float("nan"), None),
    (float("inf"), None),
    ("text", "text"),
    (None, None),
])
def test_values_are_coerced_to_something_storable(value, expected):
    assert coerce_index_value(value) == expected


def test_coerced_metadata_survives_a_json_round_trip():
    # An empty Excel cell arrives as NaN, which json.dumps writes as the bare
    # token NaN — valid Python, invalid JSON, and unreadable by json_extract.
    coerced = coerce_index_value({"Notes": float("nan"), "T": np.int32(1150)})

    assert json.loads(json.dumps(coerced)) == {"Notes": None, "T": 1150}


# =============================================================================
# Qualified keys
# =============================================================================

def test_a_key_containing_a_slash_survives_being_stored_and_split():
    from pyKES.database.index_registry import coerce_index_mapping

    # Real metadata keys contain slashes. They are escaped on the way in with
    # the same convention the HDF5 layout uses, so splitting a reference path
    # cannot mistake one for a separator.
    stored = coerce_index_mapping({"Catalyst concentration [g/L]": 1.0})
    assert list(stored) == ["Catalyst concentration [g__SLASH__L]"]

    role_path, leaf = split_qualified_key("Catalyst concentration [g__SLASH__L]")
    assert role_path is None
    assert leaf == "Catalyst concentration [g/L]"


def test_an_inherited_key_that_itself_contains_a_slash_round_trips():
    from pyKES.database.index_registry import coerce_index_mapping, qualify_key

    stored = next(iter(coerce_index_mapping({"Irradiance A [mW/cm2]": 44.25})))
    qualified = qualify_key("catalyst_batch", stored)

    assert split_qualified_key(qualified) == ("catalyst_batch",
                                              "Irradiance A [mW/cm2]")


def test_inherited_keys_are_named_after_the_entry_that_owns_them(connection):
    build_chain(connection, CHAIN_REFERENCES)

    # Named by the kind of entry the value belongs to, not by the route that
    # reached it -- so the semiconductor's field has one name whether it was
    # reached in one hop or three.
    assert inherited(connection, "ABC-67",
                     "finished_semiconductor/Synthesis temperature [degC]") == [1150]
    assert inherited(connection, "ABC-67",
                     "catalyst_batch/Photodeposition wavelength [nm]") == [360]


def test_the_routes_that_reached_a_value_are_kept_for_display(connection):
    build_chain(connection, CHAIN_REFERENCES)

    routes = {row["contributor"]: json.loads(row["role_paths"])
              for row in read_contributions(connection, "ABC-67")}

    # Nothing filters on these any more, but which chain led to a value is
    # still worth being able to see on the entry page.
    assert routes["BC-2"] == ["catalyst_batch/finished_semiconductor"]


def test_the_target_query_finds_the_experiment_two_hops_away(connection):
    build_chain(connection, CHAIN_REFERENCES)

    rows, total = search_entities(connection, "experiment", filters=[
        Filter("finished_semiconductor/Synthesis temperature [degC]",
               "between", [1150, 1150]),
        Filter("catalyst_batch/Photodeposition wavelength [nm]",
               "between", [360, 360])], limit=None)

    assert [row["entity_id"] for row in rows] == ["ABC-67"]


def test_own_metadata_is_never_displaced_by_an_inherited_key(connection):
    # Both entries carry their own 'Temperature [degC]'. Qualification is what
    # makes the merge incapable of collision.
    add_entity(connection, "PREC-1", "precursor_chemical", {"Temperature [degC]": 1150})
    add_entity(connection, "EXP-1", "experiment",
               {"Temperature [degC]": 25, "Precursor": "PREC-1"})
    finalise_entity(connection, "EXP-1", {"Temperature [degC]": 25, "Precursor": "PREC-1"},
                    {"Precursor": {"role": "precursor"}})

    own = json.loads(connection.execute(
        "SELECT metadata FROM entities WHERE entity_id = 'EXP-1'").fetchone()["metadata"])

    assert own["Temperature [degC]"] == 25
    assert inherited(connection, "EXP-1",
                     "precursor_chemical/Temperature [degC]") == [1150]


def test_presentation_fields_are_not_inherited(connection):
    add_entity(connection, "PREC-2", "precursor_chemical",
               {"color": "blue", "group": "Reference", "Purity [%]": 99.9})
    add_entity(connection, "EXP-2", "experiment", {"Precursor": "PREC-2"})
    finalise_entity(connection, "EXP-2", {"Precursor": "PREC-2"},
                    {"Precursor": {"role": "precursor"}})

    assert inherited(connection, "EXP-2", "precursor_chemical/color") == []
    assert inherited(connection, "EXP-2", "precursor_chemical/group") == []
    assert inherited(connection, "EXP-2", "precursor_chemical/Purity [%]") == [99.9]


# =============================================================================
# Reference edge cases the design identified as able to fail silently
# =============================================================================

def test_two_fields_may_now_share_a_role(connection):
    # This used to be refused: two targets under one role produced identical
    # qualified keys and the second overwrote the first. Inherited metadata is
    # a set of contributions now, so nothing overwrites anything.
    references = extract_references(
        {"Precursor A": "BC-1", "Precursor B": "BC-2"},
        {"Precursor A": {"role": "precursor"}, "Precursor B": {"role": "precursor"}},
    )

    assert references == [("precursor", "BC-1", 0), ("precursor", "BC-2", 0)]


@pytest.mark.parametrize("cell, expected", [
    ("EA-1; EA-2; EA-3", ["EA-1", "EA-2", "EA-3"]),
    ("EA-1;EA-2", ["EA-1", "EA-2"]),
    ("EA-1\nEA-2", ["EA-1", "EA-2"]),
    ("  EA-1  ", ["EA-1"]),
    ("EA-1; EA-1", ["EA-1"]),
])
def test_one_field_may_name_several_entries(cell, expected):
    # Naming an entry twice in one cell is a typo; recording it twice would say
    # the entry was used twice.
    assert split_reference_cell(cell) == expected


def test_several_entries_of_one_kind_all_contribute(connection):
    for entity_id, supplier in (("EA-1", "Merck"), ("EA-2", "Alfa"), ("EA-3", "Acme")):
        add_entity(connection, entity_id, "precursor_chemical", {"Supplier": supplier})

    add_entity(connection, "SEMI-1", "finished_semiconductor",
               {"Precursor Chemicals": "EA-1; EA-2; EA-3"})
    finalise_entity(connection, "SEMI-1", {"Precursor Chemicals": "EA-1; EA-2; EA-3"},
                    {"Precursor Chemicals": {"role": "precursor_chemical"}})

    assert sorted(inherited(connection, "SEMI-1",
                            "precursor_chemical/Supplier")) == ["Acme", "Alfa", "Merck"]


def test_blank_reference_fields_are_skipped():
    references = extract_references(
        {"Coating": "", "Precursor": "BC-1", "Dopant": None},
        {"Coating": {"role": "coating"}, "Precursor": {"role": "precursor"},
         "Dopant": {"role": "dopant"}},
    )

    assert references == [("precursor", "BC-1", 0)]


def test_a_forward_reference_resolves_when_its_target_arrives(connection):
    # The experiment names a batch that does not exist yet.
    add_entity(connection, "EXP-3", "experiment", {"Catalyst Batch": "LATE-1"})
    finalise_entity(connection, "EXP-3", {"Catalyst Batch": "LATE-1"}, CHAIN_REFERENCES)

    assert len(read_dangling_references(connection)) == 1
    assert inherited(connection, "EXP-3",
                     "catalyst_batch/Photodeposition wavelength [nm]") == []

    # The batch arrives later and the experiment picks its metadata up.
    add_entity(connection, "LATE-1", "catalyst_batch",
               {"Photodeposition wavelength [nm]": 405})
    finalise_entity(connection, "LATE-1", {"Photodeposition wavelength [nm]": 405},
                    CHAIN_REFERENCES)

    assert read_dangling_references(connection) == []
    assert inherited(connection, "EXP-3",
                     "catalyst_batch/Photodeposition wavelength [nm]") == [405]


def test_a_diamond_reaches_the_shared_ancestor_by_both_paths(connection):
    add_entity(connection, "SRC-1", "commercial_chemical", {"Supplier": "Acme"})
    for batch in ("BATCH-A", "BATCH-B"):
        add_entity(connection, batch, "catalyst_batch", {"Precursor": "SRC-1"})
        finalise_entity(connection, batch, {"Precursor": "SRC-1"},
                        {"Precursor": {"role": "precursor"}})

    add_entity(connection, "EXP-D", "experiment",
               {"Batch One": "BATCH-A", "Batch Two": "BATCH-B"})
    finalise_entity(connection, "EXP-D",
                    {"Batch One": "BATCH-A", "Batch Two": "BATCH-B"},
                    {"Batch One": {"role": "batch_one"}, "Batch Two": {"role": "batch_two"}})

    # One contributor, reached twice: the primary key of `contributions` is
    # what stops a diamond in the graph from counting the same entry twice.
    assert inherited(connection, "EXP-D",
                     "commercial_chemical/Supplier") == ["Acme"]

    routes = [json.loads(row["role_paths"]) for row in
              read_contributions(connection, "EXP-D")
              if row["contributor"] == "SRC-1"]

    # Both routes are still recorded, for the entry page to show.
    assert sorted(routes[0]) == ["batch_one/precursor", "batch_two/precursor"]


def test_a_cycle_is_detected_rather_than_hung(connection):
    add_entity(connection, "LOOP-A", "other_entity", {"Next": "LOOP-B", "A": 1})
    add_entity(connection, "LOOP-B", "other_entity", {"Next": "LOOP-A", "B": 2})
    for entity_id, other in (("LOOP-A", "LOOP-B"), ("LOOP-B", "LOOP-A")):
        finalise_entity(connection, entity_id, {"Next": other},
                        {"Next": {"role": "next"}})

    assert inherited(connection, "LOOP-A", "other_entity/B") == [2]
    assert set(find_cyclic_entities(connection)) == {"LOOP-A", "LOOP-B"}


def test_resolution_stops_at_the_depth_cap(connection):
    length = MAX_REFERENCE_DEPTH + 4
    for position in range(length):
        add_entity(connection, f"N-{position}", "other_entity",
                   {"Depth": position, "Next": f"N-{position + 1}"})
    for position in range(length):
        finalise_entity(connection, f"N-{position}",
                        {"Depth": position, "Next": f"N-{position + 1}"},
                        {"Next": {"role": "next"}})

    deepest = max(row["depth"] for row in read_contributions(connection, "N-0"))

    assert deepest <= MAX_REFERENCE_DEPTH


def test_editing_an_ancestor_recomputes_every_descendant(connection):
    build_chain(connection, CHAIN_REFERENCES)

    assert find_dependents(connection, "BC-2") == ["ABC-12", "ABC-67"] or \
           sorted(find_dependents(connection, "BC-2")) == ["ABC-12", "ABC-67"]

    update_entity_metadata(connection, "BC-2",
                           {"Synthesis temperature [degC]": 1200},
                           user="tester", reference_instructions=CHAIN_REFERENCES)

    assert inherited(connection, "ABC-67",
                     "finished_semiconductor/Synthesis temperature [degC]") == [1200]


# =============================================================================
# Permissions
# =============================================================================

def test_only_the_owner_or_an_admin_may_edit(connection):
    add_entity(connection, "OWNED-1", "experiment", {"T": 1}, owner="alice")

    assert may_edit(connection, "OWNED-1", "alice") is True
    assert may_edit(connection, "OWNED-1", "bob") is False
    assert may_edit(connection, "OWNED-1", "bob", is_admin=True) is True
    assert may_edit(connection, "ABSENT", "alice", is_admin=True) is False


def test_a_non_owner_cannot_edit(connection):
    add_entity(connection, "OWNED-2", "experiment", {"T": 1}, owner="alice")

    with pytest.raises(PermissionError, match="only its owner or an admin"):
        update_entity_metadata(connection, "OWNED-2", {"T": 2}, user="bob")


# =============================================================================
# Registries
# =============================================================================

def test_a_key_seen_with_two_types_is_marked_mixed(connection):
    add_entity(connection, "M-1", "experiment", {"Loading [wt%]": 0.1})
    add_entity(connection, "M-2", "experiment", {"Loading [wt%]": "trace"})
    for entity_id, metadata in (("M-1", {"Loading [wt%]": 0.1}),
                                ("M-2", {"Loading [wt%]": "trace"})):
        finalise_entity(connection, entity_id, metadata, {})

    row = connection.execute(
        "SELECT inferred_type FROM metadata_keys WHERE key = 'Loading [wt%]'"
    ).fetchone()

    assert row["inferred_type"] == TYPE_MIXED


def test_a_consistent_key_keeps_its_type_and_counts_occurrences(connection):
    for position in range(3):
        add_entity(connection, f"C-{position}", "experiment", {"Irradiance": 50})
        finalise_entity(connection, f"C-{position}", {"Irradiance": 50}, {})

    row = connection.execute(
        "SELECT inferred_type, occurrences FROM metadata_keys WHERE key = 'Irradiance'"
    ).fetchone()

    assert row["inferred_type"] == TYPE_NUMBER
    assert row["occurrences"] == 3


def test_one_leaf_is_one_key_however_deep_it_was_reached(connection):
    build_chain(connection, CHAIN_REFERENCES)

    keys = read_metadata_keys(connection, leaf_name="Synthesis temperature [degC]")

    # Two keys, not one per depth: bare on the semiconductor that owns it, and
    # `finished_semiconductor/...` on everything that inherits it -- whether
    # that is one hop away or three. Naming the field after the entry that owns
    # it is what collapses the rest.
    assert sorted(row["key"] for row in keys) == [
        "Synthesis temperature [degC]",
        "finished_semiconductor/Synthesis temperature [degC]"]

    inheritors = next(row for row in keys if row["role_path"])
    assert sorted(json.loads(inheritors["entity_types"])) == ["catalyst_batch",
                                                              "experiment"]


# =============================================================================
# Ingestion
# =============================================================================

def make_dataset(names, index_instructions=None, reference_instructions=None):
    """Build a small in-memory dataset with known processed values."""
    dataset = ExperimentalDataset()

    for position, name in enumerate(names):
        dataset.add_experiment(Experiment(
            experiment_name=name,
            raw_data_file=f"{name}.csv",
            color="blue",
            group="Reference",
            metadata={"Experiment": name, "Active": True,
                      "Irradiance [mW/cm2]": 50 + position,
                      "Catalyst Batch": "BATCH-X"},
            raw_data={"time_s": np.arange(500, dtype=float)},
            processed_data={"max_rate_umol_s": 1e-5 * (position + 1)},
        ))

    instruction = {}
    if index_instructions is not None:
        instruction["index_instructions"] = index_instructions
    if reference_instructions is not None:
        instruction["reference_instructions"] = reference_instructions
    dataset.plotting_instruction = instruction

    return dataset


def test_hdf5_upload_is_split_into_payloads_and_indexed(connection, paths, tmp_path):
    source = tmp_path / "batch.h5"
    make_dataset(["EXP-A", "EXP-B"],
                 index_instructions={"Max rate": {"result": "processed_data/max_rate_umol_s"}}
                 ).save_to_hdf5(str(source), verbose=False)

    report = ingest_hdf5_upload(connection, paths, source, "alice", schemas={})

    assert sorted(report.added) == ["EXP-A", "EXP-B"]
    assert (paths.payload_directory / "EXP-A.h5").exists()

    results = json.loads(connection.execute(
        "SELECT results FROM entities WHERE entity_id = 'EXP-A'").fetchone()["results"])
    assert results["Max rate"] == pytest.approx(1e-5)


def test_a_payload_is_independently_a_valid_dataset(connection, paths, tmp_path):
    source = tmp_path / "batch.h5"
    make_dataset(["EXP-A"]).save_to_hdf5(str(source), verbose=False)
    ingest_hdf5_upload(connection, paths, source, "alice", schemas={})

    reloaded = ExperimentalDataset.load_from_hdf5(
        str(paths.payload_directory / "EXP-A.h5"))

    assert list(reloaded.experiments) == ["EXP-A"]
    assert reloaded.experiments["EXP-A"].processed_data["max_rate_umol_s"] == \
        pytest.approx(1e-5)


def test_reingesting_the_same_file_changes_nothing(connection, paths, tmp_path):
    source = tmp_path / "batch.h5"
    make_dataset(["EXP-A"]).save_to_hdf5(str(source), verbose=False)

    ingest_hdf5_upload(connection, paths, source, "alice", schemas={})
    second = ingest_hdf5_upload(connection, paths, source, "alice", schemas={})

    assert second.already_ingested is True
    assert connection.execute(
        "SELECT COUNT(*) AS n FROM entities").fetchone()["n"] == 1


def test_a_name_collision_keeps_both_under_a_version_suffix(connection, paths, tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    make_dataset(["EXP-A"]).save_to_hdf5(str(first), verbose=False)
    # A different file holding the same experiment name.
    make_dataset(["EXP-A", "EXP-C"]).save_to_hdf5(str(second), verbose=False)

    ingest_hdf5_upload(connection, paths, first, "alice", schemas={})
    report = ingest_hdf5_upload(connection, paths, second, "bob", schemas={})

    assert "EXP-A__v2" in report.added
    assert report.versioned == ["EXP-A__v2"]
    stored = {row["entity_id"] for row in connection.execute(
        "SELECT entity_id FROM entities")}
    assert stored == {"EXP-A", "EXP-A__v2", "EXP-C"}


def test_allocate_entity_id_counts_up_across_collisions(connection):
    add_entity(connection, "N-1", "experiment", {})

    assert allocate_entity_id(connection, "N-1") == ("N-1__v2", 2)


def test_results_table_instructions_are_used_when_no_index_mapping_is_declared():
    dataset = make_dataset(["EXP-A"])
    dataset.plotting_instruction = {
        "results_table_instructions": {"Max rate": {"result": "processed_data/max_rate_umol_s"}}
    }

    assert read_index_instructions(dataset) == {
        "Max rate": {"result": "processed_data/max_rate_umol_s"}}


def test_unresolvable_result_paths_are_simply_absent():
    dataset = make_dataset(["EXP-A"])
    experiment = dataset.experiments["EXP-A"]

    results = apply_index_instructions(experiment, {
        "Max rate": {"result": "processed_data/max_rate_umol_s"},
        "Gas phase rate": {"result": "processed_data/not_measured_here"},
    })

    assert set(results) == {"Max rate"}


def test_entity_sheet_ingestion_links_the_chain(connection, paths, tmp_path):
    sheet = tmp_path / "batches.xlsx"
    pd.DataFrame([
        {"Experiment": "BATCH-X", "Finished Semiconductor": "SEMI-1",
         "Photodeposition wavelength [nm]": 365},
    ]).to_excel(sheet, index=False)

    source = tmp_path / "batch.h5"
    make_dataset(["EXP-A"],
                 reference_instructions={"Catalyst Batch": {"role": "catalyst_batch"}}
                 ).save_to_hdf5(str(source), verbose=False)

    ingest_hdf5_upload(connection, paths, source, "alice", schemas={})
    ingest_entity_sheet(connection, paths, sheet, "catalyst_batch", "bob",
                        schemas={})

    assert inherited(connection, "EXP-A",
                     "catalyst_batch/Photodeposition wavelength [nm]") == [365]


def test_a_sheet_without_the_identifier_column_is_refused(connection, paths, tmp_path):
    sheet = tmp_path / "bad.xlsx"
    pd.DataFrame([{"Name": "BATCH-X"}]).to_excel(sheet, index=False)

    with pytest.raises(IngestionError, match="has no 'Experiment' column"):
        ingest_entity_sheet(connection, paths, sheet, "catalyst_batch", "bob",
                        schemas={})


def test_an_empty_hdf5_upload_is_refused(connection, paths, tmp_path):
    source = tmp_path / "empty.h5"
    ExperimentalDataset().save_to_hdf5(str(source), verbose=False)

    with pytest.raises(IngestionError, match="holds no experiments"):
        ingest_hdf5_upload(connection, paths, source, "alice", schemas={})


def test_new_metadata_columns_are_absorbed_without_migration(connection, paths, tmp_path):
    # The whole point of the JSON metadata column: next year's experiments carry
    # fields this year's do not, and nothing has to be migrated.
    first = tmp_path / "old.h5"
    make_dataset(["OLD-1"]).save_to_hdf5(str(first), verbose=False)
    ingest_hdf5_upload(connection, paths, first, "alice", schemas={})

    newer = make_dataset(["NEW-1"])
    newer.experiments["NEW-1"].metadata["Sacrificial agent"] = "methanol"
    second = tmp_path / "new.h5"
    newer.save_to_hdf5(str(second), verbose=False)
    ingest_hdf5_upload(connection, paths, second, "alice", schemas={})

    keys = {row["key"] for row in read_metadata_keys(connection)}
    assert "Sacrificial agent" in keys

    found = connection.execute(
        """SELECT entity_id FROM entities
           WHERE json_extract(metadata, '$."Sacrificial agent"') = 'methanol'"""
    ).fetchall()
    assert [row["entity_id"] for row in found] == ["NEW-1"]


# =============================================================================
# Rebuilding
# =============================================================================

def test_the_index_can_be_rebuilt_from_the_retained_uploads(connection, paths, tmp_path):
    source = tmp_path / "batch.h5"
    make_dataset(["EXP-A", "EXP-B"],
                 index_instructions={"Max rate": {"result": "processed_data/max_rate_umol_s"}}
                 ).save_to_hdf5(str(source), verbose=False)
    ingest_hdf5_upload(connection, paths, source, "alice", schemas={})

    before = connection.execute(
        "SELECT entity_id, results FROM entities ORDER BY entity_id").fetchall()

    rebuild_index(connection, paths)

    after = connection.execute(
        "SELECT entity_id, results FROM entities ORDER BY entity_id").fetchall()

    assert [tuple(row) for row in before] == [tuple(row) for row in after]


# =============================================================================
# One role, several kinds of entry
# =============================================================================

def test_a_role_carries_metadata_from_whatever_kind_it_reaches(connection):
    add_entity(connection, "SEMI-1", "finished_semiconductor",
               {"Synthesis temperature [°C]": 1150})
    add_entity(connection, "ABC-1", "catalyst_batch",
               {"Finished Semiconductor": "SEMI-1",
                "Photodeposition wavelength [nm]": 365})
    finalise_entity(connection, "ABC-1",
                    {"Finished Semiconductor": "SEMI-1"},
                    {"Finished Semiconductor": {"role": "finished_semiconductor"}})

    add_entity(connection, "MOD-1", "modified_catalyst_batch",
               {"Original Catalyst Batch": "ABC-1", "Coating thickness [nm]": 3.5})
    finalise_entity(connection, "MOD-1", {"Original Catalyst Batch": "ABC-1"},
                    {"Original Catalyst Batch": {"role": "catalyst_batch"}})

    add_entity(connection, "EXP-1", "experiment",
               {"Catalyst Batch [experiment no.]": "MOD-1"})
    recomputed = finalise_entity(
        connection, "EXP-1", {"Catalyst Batch [experiment no.]": "MOD-1"},
        {"Catalyst Batch [experiment no.]": {"role": "catalyst_batch"}})

    # An edge records no target type, so the same role reaches a different kind
    # of entry and the chain simply continues through it -- and the
    # semiconductor's field is named the same whichever route reached it.
    assert inherited(connection, "EXP-1",
                     "modified_catalyst_batch/Coating thickness [nm]") == [3.5]
    assert inherited(connection, "EXP-1",
                     "finished_semiconductor/Synthesis temperature [°C]") == [1150]
    assert "EXP-1" in recomputed


def test_a_reference_to_the_wrong_kind_is_reported_not_refused(connection):
    add_entity(connection, "SEMI-1", "finished_semiconductor", {})
    add_entity(connection, "MOD-1", "modified_catalyst_batch",
               {"Original Catalyst Batch": "SEMI-1", "Coating thickness [nm]": 3.5})
    finalise_entity(connection, "MOD-1", {"Original Catalyst Batch": "SEMI-1"},
                    {"Original Catalyst Batch": {"role": "catalyst_batch"}})

    mismatches = read_reference_type_mismatches(
        connection, {("modified_catalyst_batch", "catalyst_batch"): ["catalyst_batch"]})

    # The entry is written and its metadata merges: a wrong-kind reference is a
    # labelling mistake, and discarding it would hide the mistake rather than
    # show it.
    assert len(mismatches) == 1
    assert mismatches[0]["target_type"] == "finished_semiconductor"
    assert connection.execute(
        "SELECT 1 FROM entities WHERE entity_id = 'MOD-1'").fetchone() is not None


def test_a_role_nobody_declared_is_not_checked(connection):
    add_entity(connection, "BC-1", "commercial_chemical", {})
    add_entity(connection, "SEMI-1", "finished_semiconductor",
               {"Precursor Chemical A": "BC-1"})
    finalise_entity(connection, "SEMI-1", {"Precursor Chemical A": "BC-1"},
                    {"Precursor Chemical A": {"role": "precursor_chemical_a"}})

    # An upload may declare references of its own; inventing a constraint for
    # one would refuse data the group deliberately sent.
    assert read_reference_type_mismatches(connection, {}) == []


# =============================================================================
# Mapping-valued metadata through the graph
# =============================================================================

def test_a_mapping_survives_the_inheritance_merge(connection):
    add_entity(connection, "SEMI-1", "finished_semiconductor",
               {"Dopants [mol%]": {"Ir": 0.02, "Ru": 0.02}})
    add_entity(connection, "ABC-1", "catalyst_batch",
               {"Finished Semiconductor": "SEMI-1"})
    finalise_entity(connection, "ABC-1", {"Finished Semiconductor": "SEMI-1"},
                    {"Finished Semiconductor": {"role": "finished_semiconductor"}})

    dopants = {row["sub_key"]: row["number"] for row in
               read_contributions(connection, "ABC-1")
               if row["key"] == "Dopants [mol%]"}

    # Expanded to one row per name, which is what lets a filter on one dopant
    # use an index instead of reading every mapping.
    assert dopants == {"Ir": 0.02, "Ru": 0.02}


def test_the_registry_records_the_names_a_mapping_carries(connection):
    add_entity(connection, "SEMI-1", "finished_semiconductor",
               {"Dopants [mol%]": {"Ir": 0.02}})
    add_entity(connection, "SEMI-2", "finished_semiconductor",
               {"Dopants [mol%]": {"Cr": 0.03, "Ir": 0.05}})
    for entity_id in ("SEMI-1", "SEMI-2"):
        add_entity(connection, f"ABC-{entity_id[-1]}", "catalyst_batch",
                   {"Finished Semiconductor": entity_id})
        finalise_entity(connection, f"ABC-{entity_id[-1]}",
                        {"Finished Semiconductor": entity_id},
                        {"Finished Semiconductor": {"role": "finished_semiconductor"}})

    row = read_metadata_keys(connection, "Dopants [mol%]")[0]

    # Every entry declares its own set, so the filter's dropdown is the union.
    assert row["inferred_type"] == "mapping"
    assert json.loads(row["sub_keys"]) == ["Cr", "Ir"]


def test_a_column_added_after_the_index_was_built_is_added_to_it(paths):
    import sqlite3

    connection = open_index(paths)
    connection.execute("ALTER TABLE metadata_keys DROP COLUMN sub_keys")
    connection.commit()
    connection.close()

    # Reopening must repair it: CREATE TABLE IF NOT EXISTS leaves an existing
    # table alone, so a new column never reaches a database that already exists.
    reopened = open_index(paths)
    columns = {row["name"] for row in
               reopened.execute("PRAGMA table_info(metadata_keys)")}
    reopened.close()

    assert "sub_keys" in columns
