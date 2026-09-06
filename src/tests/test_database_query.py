"""
Tests for the query layer the database application sits on.

The cases worth pinning are the ones a browser would only reveal by breaking:
that a filter on a field two references away works at all, that metadata keys
containing quotes or slashes cannot corrupt the SQL they end up in, and that
facets are generated from the metadata actually present rather than configured.
"""

import json

import pytest

from pyKES.database.index_ingest import finalise_entity, insert_entity
from pyKES.database.index_query import (
    Filter,
    RESULT_PREFIX,
    build_facets,
    build_predicate,
    database_statistics,
    json_path,
    list_axis_options,
    property_map_data,
    read_neighbours,
    rows_to_frame,
    search_entities,
)
from pyKES.database.index_registry import register_result_keys
from pyKES.database.index_schema import IndexPaths, open_index


# =============================================================================
# Fixtures
# =============================================================================

CHAIN_REFERENCES = {
    "Catalyst Batch": {"role": "catalyst_batch"},
    "Finished Semiconductor": {"role": "finished_semiconductor"},
}


@pytest.fixture
def connection(tmp_path):
    """An index holding a three-deep chain with known values."""
    handle = open_index(IndexPaths(root=tmp_path))

    add(handle, "SEMI-1", "finished_semiconductor",
        {"Synthesis temperature [degC]": 1150})
    add(handle, "SEMI-2", "finished_semiconductor",
        {"Synthesis temperature [degC]": 1000})

    for index, semiconductor in enumerate(("SEMI-1", "SEMI-2"), start=1):
        add(handle, f"BATCH-{index}", "catalyst_batch",
            {"Photodeposition wavelength [nm]": 365 if index == 1 else 455,
             "Finished Semiconductor": semiconductor})

    for index in range(1, 7):
        batch = f"BATCH-{1 if index <= 3 else 2}"
        add(handle, f"EXP-{index}", "experiment",
            {"Irradiance A [mW/cm2]": 10.0 * index,
             "Catalyst concentration [g/L]": 1.0,
             "Measured Analyte [O2 or H2]": "O2" if index % 2 else "H2",
             "Catalyst Batch": batch},
            results={"Max. rate (umol/s)": 1e-5 * index})

    # Ingestion registers the labels it maps; the fixture writes results
    # directly, so it has to do the same for them to appear as axis options.
    register_result_keys(handle, {"Max. rate (umol/s)": {
        "result": "processed_data/max_rate_umol_s"}})

    handle.commit()
    yield handle
    handle.close()


def add(connection, entity_id, entity_type, metadata, results=None):
    """Insert an entity and settle its references."""
    insert_entity(connection, entity_id=entity_id, base_id=entity_id, version=1,
                  entity_type=entity_type, metadata=metadata,
                  results=results or {}, owner="tester", upload_id=None)
    finalise_entity(connection, entity_id, metadata, CHAIN_REFERENCES)


# =============================================================================
# Predicate construction
# =============================================================================

def test_metadata_keys_are_bound_never_interpolated():
    predicate, parameters = build_predicate(
        Filter("Notes'; DROP TABLE entities; --", "equals", "x"))

    # The key reaches SQLite as a bound JSON path, so nothing in it is parsed
    # as SQL.
    assert "DROP TABLE" not in predicate
    assert parameters[0] == json_path("Notes'; DROP TABLE entities; --")


def test_a_key_containing_a_quote_is_escaped_into_the_path():
    assert json_path('Odd "key"') == '$."Odd \\"key\\""'


def test_an_empty_multiselect_means_no_constraint():
    predicate, parameters = build_predicate(Filter("Cocatalyst", "in", []))

    assert predicate == "1 = 1"
    assert parameters == []


def test_an_unknown_operator_is_refused():
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        build_predicate(Filter("x", "regex", "y"))


# =============================================================================
# Search
# =============================================================================

def test_searching_one_kind_of_entry(connection):
    rows, total = search_entities(connection, entity_type="experiment")

    assert total == 6
    assert len(rows) == 6


def test_filtering_on_the_entitys_own_metadata(connection):
    rows, total = search_entities(
        connection, "experiment",
        filters=[Filter("Irradiance A [mW/cm2]", "between", [25.0, 45.0])])

    assert {row["entity_id"] for row in rows} == {"EXP-3", "EXP-4"}


def test_filtering_one_reference_away(connection):
    rows, _ = search_entities(
        connection, "experiment",
        filters=[Filter("catalyst_batch::Photodeposition wavelength [nm]",
                        "between", [360, 370])])

    assert {row["entity_id"] for row in rows} == {"EXP-1", "EXP-2", "EXP-3"}


def test_filtering_two_references_away(connection):
    # The question the whole design exists to answer.
    rows, _ = search_entities(
        connection, "experiment",
        filters=[Filter("catalyst_batch::finished_semiconductor::"
                        "Synthesis temperature [degC]", "between", [1100, 1200])])

    assert {row["entity_id"] for row in rows} == {"EXP-1", "EXP-2", "EXP-3"}


def test_combining_an_inherited_filter_with_an_own_one(connection):
    rows, _ = search_entities(
        connection, "experiment",
        filters=[
            Filter("catalyst_batch::finished_semiconductor::"
                   "Synthesis temperature [degC]", "between", [1100, 1200]),
            Filter("Measured Analyte [O2 or H2]", "in", ["O2"]),
        ])

    assert {row["entity_id"] for row in rows} == {"EXP-1", "EXP-3"}


def test_filtering_on_a_mapped_result(connection):
    rows, _ = search_entities(
        connection, "experiment",
        filters=[Filter(f"{RESULT_PREFIX}Max. rate (umol/s)",
                        "between", [4.5e-5, 1e-4])])

    assert {row["entity_id"] for row in rows} == {"EXP-5", "EXP-6"}


def test_free_text_reaches_into_the_metadata(connection):
    rows, _ = search_entities(connection, "experiment", text="BATCH-2")

    assert {row["entity_id"] for row in rows} == {"EXP-4", "EXP-5", "EXP-6"}


def test_pagination_splits_the_result_without_losing_the_total(connection):
    first, total = search_entities(connection, "experiment", limit=4, offset=0)
    second, _ = search_entities(connection, "experiment", limit=4, offset=4)

    assert total == 6
    assert len(first) == 4 and len(second) == 2
    assert not {row["entity_id"] for row in first} & {row["entity_id"] for row in second}


def test_latest_only_hides_superseded_versions(connection):
    insert_entity(connection, entity_id="EXP-1__v2", base_id="EXP-1", version=2,
                  entity_type="experiment", metadata={"Irradiance A [mW/cm2]": 99.0},
                  results={}, owner="other", upload_id=None)
    connection.commit()

    latest, latest_total = search_entities(connection, "experiment")
    everything, all_total = search_entities(connection, "experiment",
                                            latest_only=False)

    assert latest_total == 6 and all_total == 7
    assert "EXP-1" not in {row["entity_id"] for row in latest}
    assert "EXP-1__v2" in {row["entity_id"] for row in latest}


def test_an_unsortable_column_falls_back_rather_than_reaching_sql(connection):
    rows, _ = search_entities(connection, "experiment",
                              order_by="metadata; DROP TABLE entities")

    assert len(rows) == 6


# =============================================================================
# Facets
# =============================================================================

def test_facets_are_generated_from_the_metadata_present(connection):
    facets = build_facets(connection, "experiment")
    by_label = {facet.label: facet for facet in facets}

    assert by_label["Irradiance A [mW/cm2]"].kind == "range"
    assert by_label["Irradiance A [mW/cm2]"].bounds == (10.0, 60.0)
    assert by_label["Measured Analyte [O2 or H2]"].kind == "select"
    assert set(by_label["Measured Analyte [O2 or H2]"].options) == {"O2", "H2"}


def test_inherited_facets_carry_the_path_they_came_through(connection):
    facets = {facet.label: facet for facet in build_facets(connection, "experiment")}

    assert facets["Photodeposition wavelength [nm]"].role_path == "catalyst_batch"
    assert (facets["Synthesis temperature [degC]"].role_path
            == "catalyst_batch::finished_semiconductor")


def test_a_constant_numeric_key_gets_no_slider(connection):
    # Every experiment has the same catalyst concentration, so a range slider
    # would have zero width and filter nothing.
    labels = [facet.label for facet in build_facets(connection, "experiment")]

    assert "Catalyst concentration [g/L]" not in labels


# =============================================================================
# Cross-entity views
# =============================================================================

def test_the_property_map_pairs_an_inherited_axis_with_a_result(connection):
    frame = property_map_data(
        connection,
        x_key="catalyst_batch::finished_semiconductor::Synthesis temperature [degC]",
        y_key=f"{RESULT_PREFIX}Max. rate (umol/s)",
        entity_type="experiment")

    assert len(frame) == 6
    assert set(frame["x"]) == {1150, 1000}


def test_rows_lacking_an_axis_are_dropped(connection):
    frame = property_map_data(connection, x_key="Irradiance A [mW/cm2]",
                              y_key="Not measured anywhere",
                              entity_type="experiment")

    assert frame.empty


def test_axis_options_offer_results_before_metadata(connection):
    options = list_axis_options(connection, "experiment")

    assert options[0].startswith(RESULT_PREFIX)
    assert any("Synthesis temperature" in option for option in options)


# =============================================================================
# The graph around one entry
# =============================================================================

def test_neighbours_show_both_directions(connection):
    experiment = read_neighbours(connection, "EXP-1")
    semiconductor = read_neighbours(connection, "SEMI-1")

    assert [edge["entity_id"] for edge in experiment["references"]] == ["BATCH-1"]
    assert experiment["referenced_by"] == []
    assert [edge["entity_id"] for edge in semiconductor["referenced_by"]] == ["BATCH-1"]


def test_statistics_count_each_kind(connection):
    statistics = database_statistics(connection)

    assert statistics["by_type"] == {"experiment": 6, "catalyst_batch": 2,
                                     "finished_semiconductor": 2}
    assert statistics["entities"] == 10


# =============================================================================
# Table rendering
# =============================================================================

def test_the_results_table_carries_the_requested_columns(connection):
    rows, _ = search_entities(connection, "experiment", limit=2)
    frame = rows_to_frame(rows, ["Irradiance A [mW/cm2]",
                                 f"{RESULT_PREFIX}Max. rate (umol/s)"])

    assert list(frame.columns) == ["entity_id", "entity_type", "group", "owner",
                                   "Irradiance A [mW/cm2]", "Max. rate (umol/s)"]
    assert frame["Max. rate (umol/s)"].notna().all()
