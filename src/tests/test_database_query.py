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
    stored_metadata_key,
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
from pyKES.database.index_registry import read_metadata_keys, register_result_keys
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
        filters=[Filter(stored_metadata_key("Irradiance A [mW/cm2]"),
                        "between", [25.0, 45.0])])

    assert {row["entity_id"] for row in rows} == {"EXP-3", "EXP-4"}


def test_a_hand_written_key_is_converted_to_the_stored_form():
    assert stored_metadata_key("Irradiance A [mW/cm2]") == \
        "Irradiance A [mW__SLASH__cm2]"


def test_filtering_one_reference_away(connection):
    rows, _ = search_entities(
        connection, "experiment",
        filters=[Filter("catalyst_batch/Photodeposition wavelength [nm]",
                        "between", [360, 370])])

    assert {row["entity_id"] for row in rows} == {"EXP-1", "EXP-2", "EXP-3"}


def test_filtering_two_references_away(connection):
    # The question the whole design exists to answer.
    rows, _ = search_entities(
        connection, "experiment",
        filters=[Filter("catalyst_batch/finished_semiconductor/"
                        "Synthesis temperature [degC]", "between", [1100, 1200])])

    assert {row["entity_id"] for row in rows} == {"EXP-1", "EXP-2", "EXP-3"}


def test_combining_an_inherited_filter_with_an_own_one(connection):
    rows, _ = search_entities(
        connection, "experiment",
        filters=[
            Filter("catalyst_batch/finished_semiconductor/"
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
            == "catalyst_batch/finished_semiconductor")


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
        x_key="catalyst_batch/finished_semiconductor/Synthesis temperature [degC]",
        y_key=f"{RESULT_PREFIX}Max. rate (umol/s)",
        entity_type="experiment")

    assert len(frame) == 6
    assert set(frame["x"]) == {1150, 1000}


def test_rows_lacking_an_axis_are_dropped(connection):
    frame = property_map_data(connection,
                              x_key=stored_metadata_key("Irradiance A [mW/cm2]"),
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
    stored = stored_metadata_key("Irradiance A [mW/cm2]")
    frame = rows_to_frame(rows, [stored, f"{RESULT_PREFIX}Max. rate (umol/s)"])

    # Looked up by the stored key, but headed by the name a person wrote: the
    # escaping is storage detail and never reaches the screen. The chosen
    # columns come before the fixed ones, or a wide table hides them.
    assert list(frame.columns) == ["Entity ID", "Irradiance A [mW/cm2]",
                                   "Max. rate (umol/s)", "Kind", "Group", "Owner"]
    assert frame["Max. rate (umol/s)"].notna().all()


# =============================================================================
# Display names
# =============================================================================

def test_entity_types_read_as_names_not_identifiers():
    from pyKES.database.index_query import display_entity_type, display_role_path

    assert display_entity_type("finished_semiconductor") == "Finished semiconductor"
    assert display_role_path("catalyst_batch/finished_semiconductor") == \
        "Catalyst batch › Finished semiconductor"
    assert display_role_path(None) == ""


def test_facets_are_ordered_by_reference_depth(connection):
    from pyKES.database.index_query import reference_depth

    depths = [reference_depth(facet) for facet in build_facets(connection, "experiment")]

    # Own fields first, then one reference away, then two: the order somebody
    # narrowing a search thinks in.
    assert depths == sorted(depths)
    assert depths[0] == 0 and max(depths) == 2


def test_the_fields_of_one_referenced_entity_stay_together(connection):
    paths = [facet.role_path or "" for facet in build_facets(connection, "experiment")]

    # The sidebar puts each group under the name of the entity it describes, so
    # a path may not reappear once another has started.
    seen = []
    for role_path in paths:
        if not seen or seen[-1] != role_path:
            assert role_path not in seen
            seen.append(role_path)


def test_a_filter_group_is_named_after_its_entity(connection):
    from pyKES.database.index_query import facet_group_label

    facets = build_facets(connection, "experiment")
    labels = {facet_group_label(facet, "experiment") for facet in facets}

    # 'Finished semiconductor' says what the fields describe; 'two references
    # away' only says how far off it is.
    assert "Experiment" in labels
    assert "Finished semiconductor" in labels


# =============================================================================
# Column headers
# =============================================================================

def test_a_column_header_is_the_field_name_and_the_chain_is_the_tooltip():
    from pyKES.database.index_query import column_labels

    labels = column_labels(["Operator",
                            "catalyst_batch/finished_semiconductor/Synthesis route",
                            f"{RESULT_PREFIX}Max. rate (umol/s)"])

    assert labels["Operator"] == ("Operator", "This entry")
    assert labels["catalyst_batch/finished_semiconductor/Synthesis route"] == (
        "Synthesis route", "Catalyst batch › Finished semiconductor")
    assert labels[f"{RESULT_PREFIX}Max. rate (umol/s)"] == (
        "Max. rate (umol/s)", "Result")


def test_columns_sharing_a_field_name_are_told_apart():
    from pyKES.database.index_query import column_labels

    headers = [header for header, _ in column_labels(
        ["finished_semiconductor/precursor_chemical_a/Supplier",
         "finished_semiconductor/precursor_chemical_b/Supplier"]).values()]

    # A duplicate header would silently drop one of the two columns, which is
    # exactly the failure the user sees as 'my column did not appear'.
    assert len(set(headers)) == 2
    assert all("Supplier" in header for header in headers)


def test_a_chosen_column_never_overwrites_another(connection):
    rows, _ = search_entities(connection, "experiment", limit=2)
    chosen = ["Operator", "catalyst_batch/Photodeposition wavelength [nm]",
              f"{RESULT_PREFIX}Max. rate (umol/s)"]

    frame = rows_to_frame(rows, chosen)

    assert len(frame.columns) == len(chosen) + 4


def test_the_results_table_survives_arrow_serialisation(connection):
    import pyarrow

    rows, _ = search_entities(connection, "experiment", limit=5)
    keys = [row["key"] for row in
            read_metadata_keys(connection, entity_type="experiment")]

    # Streamlit hands every table to Arrow. A column mixing text and numbers
    # made that fail loudly on the console and then silently repair itself.
    pyarrow.Table.from_pandas(rows_to_frame(rows, keys))


def test_a_value_of_any_shape_renders_as_text():
    from pyKES.database.index_query import MISSING_PLACEHOLDER, format_value

    assert format_value(None) == MISSING_PLACEHOLDER
    assert format_value(True) == "Yes"
    assert format_value(44.25) == "44.25"
    assert format_value({"Ir": 0.02}) == '{"Ir": 0.02}'


def test_a_numeric_column_keeps_its_type_and_a_mixed_one_does_not():
    from pyKES.database.index_query import arrow_safe_frame

    frame = arrow_safe_frame([{"Result": "Max. rate", "Value": 0.031884829},
                              {"Result": "Yield", "Value": 3.98560370}])

    # Rendering these as text would print every digit pandas holds instead of
    # letting Streamlit format the column.
    assert frame["Value"].dtype == float

    mixed = arrow_safe_frame([{"Field": "Analyte", "Value": "H2"},
                              {"Field": "Irradiance", "Value": 80.0},
                              {"Field": "Active", "Value": True},
                              {"Field": "Notes", "Value": None}])

    assert list(mixed["Value"]) == ["H2", "80.0", "Yes", "—"]


# =============================================================================
# Type-ahead
# =============================================================================

def test_prefix_matches_are_offered_before_containments(connection):
    from pyKES.database.index_query import search_entity_ids

    matches = search_entity_ids(connection, "EXP-")

    assert matches[:3] == ["EXP-1", "EXP-2", "EXP-3"]
    assert len(matches) == 6


def test_an_empty_query_offers_nothing(connection):
    from pyKES.database.index_query import search_entity_ids

    assert search_entity_ids(connection, "   ") == []


def test_the_type_ahead_can_be_scoped_to_one_kind(connection):
    from pyKES.database.index_query import search_entity_ids

    assert search_entity_ids(connection, "SEMI", entity_type="experiment") == []
    assert search_entity_ids(connection, "SEMI",
                             entity_type="finished_semiconductor") == \
        ["SEMI-1", "SEMI-2"]


# =============================================================================
# One field reached by several paths
# =============================================================================

def test_paths_to_one_field_become_one_filter():
    from pyKES.database.index_query import merge_key_paths

    rows = [{"key": "catalyst_batch/finished_semiconductor/Synthesis route",
             "leaf_name": "Synthesis route",
             "role_path": "catalyst_batch/finished_semiconductor",
             "inferred_type": "text", "distinct_sample": '["Osterloh"]',
             "sub_keys": None},
            {"key": "catalyst_batch/catalyst_batch/finished_semiconductor/Synthesis route",
             "leaf_name": "Synthesis route",
             "role_path": "catalyst_batch/catalyst_batch/finished_semiconductor",
             "inferred_type": "text", "distinct_sample": '["Lercher"]',
             "sub_keys": None}]

    groups = merge_key_paths(rows)

    # A modified batch adds a hop, so the semiconductor sits at two depths. One
    # filter per depth would answer for half the experiments each, and neither
    # would say so.
    assert len(groups) == 1
    assert len(groups[0]["keys"]) == 2
    assert groups[0]["keys"][0].count("/") == 2, "the shortest path leads"
    assert sorted(groups[0]["sample"]) == ["Lercher", "Osterloh"]


def test_the_same_name_on_two_entities_stays_two_filters():
    from pyKES.database.index_query import merge_key_paths

    rows = [{"key": "catalyst_batch/Notes", "leaf_name": "Notes",
             "role_path": "catalyst_batch", "inferred_type": "text",
             "distinct_sample": "[]", "sub_keys": None},
            {"key": "catalyst_batch/finished_semiconductor/Notes",
             "leaf_name": "Notes",
             "role_path": "catalyst_batch/finished_semiconductor",
             "inferred_type": "text", "distinct_sample": "[]", "sub_keys": None}]

    # A batch's notes and a semiconductor's notes are different fields; sharing
    # a name is not enough to merge them.
    assert len(merge_key_paths(rows)) == 2


def test_a_filter_over_several_paths_reads_whichever_one_exists():
    from pyKES.database.index_query import build_expression

    expression, paths = build_expression(
        "catalyst_batch/X", None, ["catalyst_batch/catalyst_batch/X"])

    assert expression.startswith("COALESCE(")
    assert len(paths) == 2


# =============================================================================
# Mapping-valued metadata
# =============================================================================

def test_a_dopant_is_addressed_one_level_deeper():
    assert json_path("Dopants [mol%]", "Ir") == '$."Dopants [mol%]"."Ir"'


def test_a_dopant_filter_binds_its_path_rather_than_interpolating_it():
    predicate, parameters = build_predicate(
        Filter("Dopants [mol%]", "between", [0.01, 0.03], sub_key='Ir"; DROP--'))

    assert '"; DROP' not in predicate
    assert any('DROP' in str(parameter) for parameter in parameters)
