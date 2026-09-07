"""
Tests for the per-entity-type metadata schemas.

Two properties matter more than the rest and are easy to get backwards. A
declared field that is filled in wrongly must be an error, or the schema buys
nothing. A field nobody declared must *not* be an error, or the schema defeats
the thing the whole database is built around — absorbing metadata that did not
exist when it was written.
"""

from pathlib import Path

import pandas as pd
import pytest

from pyKES.database.entity_schema import (
    DEFAULT_SCHEMA_DIRECTORY,
    EntitySchema,
    FieldSchema,
    coerce_boolean,
    load_entity_schema,
    load_entity_schemas,
    parse_mapping,
    prepare_metadata,
    template_frames,
    validate_entries,
    validate_metadata,
    write_template,
)
from pyKES.database.index_ingest import IngestionError, check_against_schema
from pyKES.database.index_schema import ENTITY_TYPES


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def schema():
    """A small schema exercising every kind of check."""
    return EntitySchema(
        entity_type="experiment",
        label="Experiment",
        identifier_field="Experiment",
        fields=[
            FieldSchema(name="Analyte", type="select", options=["O2", "H2"],
                        required=True),
            FieldSchema(name="Irradiance [mW/cm2]", type="number", required=True),
            FieldSchema(name="Cocatalyst", type="multiselect",
                        options=["Rh", "Cr", "Pt"]),
            FieldSchema(name="Active", type="boolean"),
            FieldSchema(name="Catalyst Batch", type="reference",
                        role="catalyst_batch"),
            FieldSchema(name="Notes", type="text"),
        ],
    )


# =============================================================================
# The shipped schemas
# =============================================================================

def test_every_entity_type_has_a_schema():
    schemas = load_entity_schemas()

    # A kind of entry with no schema is silently unchecked, which is worse than
    # a schema that declares nothing.
    assert set(schemas) == set(ENTITY_TYPES)


def test_the_shipped_schemas_all_parse():
    for path in DEFAULT_SCHEMA_DIRECTORY.glob("*.yaml"):
        loaded = load_entity_schema(path)

        assert loaded.entity_type and loaded.label
        assert loaded.fields, f"{path.name} declares no fields"


def test_reference_fields_supply_the_chain(schema):
    schemas = load_entity_schemas()

    # The reference declaration is derived from the schemas rather than written
    # twice, so this is what wires the group's chain together.
    assert schemas["experiment"].reference_instructions() == {
        "Catalyst Batch [experiment no.]": {
            "role": "catalyst_batch",
            "accepts": ["catalyst_batch", "modified_catalyst_batch"]}}

    precursors = ["precursor_chemical", "commercial_chemical"]
    assert schemas["finished_semiconductor"].reference_instructions() == {
        "Precursor Chemical A": {"role": "precursor_chemical_a",
                                 "accepts": precursors},
        "Precursor Chemical B": {"role": "precursor_chemical_b",
                                 "accepts": precursors}}


# =============================================================================
# Malformed schema files
# =============================================================================

def test_an_unknown_field_type_is_refused():
    with pytest.raises(ValueError, match="unknown type"):
        FieldSchema(name="X", type="colour")


def test_a_choice_field_without_options_is_refused():
    with pytest.raises(ValueError, match="declares no options"):
        FieldSchema(name="Analyte", type="select")


def test_a_reference_field_without_a_role_is_refused():
    with pytest.raises(ValueError, match="declares no role"):
        FieldSchema(name="Catalyst Batch", type="reference")


def test_a_file_without_an_entity_type_is_refused(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("label: Nothing\nfields: []\n")

    with pytest.raises(ValueError, match="declares no entity_type"):
        load_entity_schema(path)


def test_a_missing_schema_directory_simply_checks_nothing(tmp_path):
    assert load_entity_schemas(tmp_path / "absent") == {}


# =============================================================================
# Validation
# =============================================================================

def test_a_complete_entry_passes(schema):
    report = validate_metadata(schema, {"Analyte": "O2",
                                        "Irradiance [mW/cm2]": 44.25})

    assert report.errors == []


def test_a_missing_required_field_is_an_error(schema):
    report = validate_metadata(schema, {"Analyte": "O2"}, "EXP-1")

    assert report.errors == ["EXP-1: 'Irradiance [mW/cm2]' is required."]


def test_a_blank_required_field_is_an_error(schema):
    report = validate_metadata(schema, {"Analyte": "  ",
                                        "Irradiance [mW/cm2]": 1.0})

    assert any("'Analyte' is required" in error for error in report.errors)


def test_a_value_outside_the_allowed_options_is_an_error(schema):
    report = validate_metadata(schema, {"Analyte": "CO2",
                                        "Irradiance [mW/cm2]": 1.0})

    assert any("expects one of" in error and "CO2" in error
               for error in report.errors)


def test_a_non_numeric_number_is_an_error(schema):
    report = validate_metadata(schema, {"Analyte": "O2",
                                        "Irradiance [mW/cm2]": "quite bright"})

    assert any("expects a number" in error for error in report.errors)


def test_a_multiselect_reports_only_the_unexpected_entries(schema):
    report = validate_metadata(schema, {"Analyte": "O2",
                                        "Irradiance [mW/cm2]": 1.0,
                                        "Cocatalyst": "Rh, Cr, Unobtainium"})

    assert len(report.errors) == 1
    assert "Unobtainium" in report.errors[0]
    assert "Rh" not in report.errors[0].split("found")[1]


@pytest.mark.parametrize("value, expected", [
    (True, True), ("True", True), ("yes", True), ("1", True),
    (False, False), ("FALSE", False), ("no", False), ("0", False),
    ("maybe", None), ("", None),
])
def test_booleans_are_read_however_a_spreadsheet_wrote_them(value, expected):
    assert coerce_boolean(value) is expected


def test_an_optional_field_left_blank_is_fine(schema):
    report = validate_metadata(schema, {"Analyte": "H2",
                                        "Irradiance [mW/cm2]": 12.0,
                                        "Notes": ""})

    assert report.errors == []


def test_an_undeclared_field_is_reported_but_never_an_error(schema):
    # The database exists to absorb fields that did not exist when the schema
    # was written. A schema that rejected them would defeat it.
    report = validate_metadata(schema, {"Analyte": "O2",
                                        "Irradiance [mW/cm2]": 1.0,
                                        "Sacrificial agent": "methanol"})

    assert report.errors == []
    assert report.undeclared == ["Sacrificial agent"]


def test_a_batch_reports_every_offending_entry(schema):
    report = validate_entries(schema, {
        "EXP-1": {"Analyte": "O2", "Irradiance [mW/cm2]": 1.0},
        "EXP-2": {"Analyte": "CO2", "Irradiance [mW/cm2]": 1.0},
        "EXP-3": {"Analyte": "H2"},
    })

    assert len(report.errors) == 2
    assert any("EXP-2" in error for error in report.errors)
    assert any("EXP-3" in error for error in report.errors)


# =============================================================================
# The ingestion hook
# =============================================================================

def test_ingestion_refuses_a_batch_that_violates_the_schema():
    with pytest.raises(IngestionError, match="do not match"):
        check_against_schema("experiment", {"EXP-1": {"Measured Analyte [O2 or H2]": "N2"}})


def test_ingestion_of_an_unschemad_kind_checks_nothing():
    assert check_against_schema("experiment", {"EXP-1": {}}, schemas={}) == []


def test_ingestion_returns_the_undeclared_fields_rather_than_refusing(schema):
    undeclared = check_against_schema(
        "experiment",
        {"EXP-1": {"Analyte": "O2", "Irradiance [mW/cm2]": 1.0,
                   "Something new": 3}},
        schemas={"experiment": schema})

    assert undeclared == ["Something new"]


# =============================================================================
# Templates
# =============================================================================

def test_the_template_carries_the_identifier_and_every_field(schema):
    entries, guide = template_frames(schema)

    assert list(entries.columns) == ["Experiment", "Analyte",
                                     "Irradiance [mW/cm2]", "Cocatalyst",
                                     "Active", "Catalyst Batch", "Notes"]
    assert entries.empty
    assert len(guide) == len(schema.fields)


def test_the_template_guide_says_what_each_field_expects(schema):
    _, guide = template_frames(schema)
    analyte = guide[guide["Field"] == "Analyte"].iloc[0]

    assert analyte["Required"] == "Yes"
    assert analyte["Allowed values"] == "O2, H2"


def test_a_written_template_reads_back_with_both_sheets(schema, tmp_path):
    path = write_template(schema, tmp_path / "template.xlsx")

    assert set(pd.ExcelFile(path).sheet_names) == {"Sheet1", "Field guide"}
    assert list(pd.read_excel(path).columns)[0] == "Experiment"


def test_a_filled_in_template_passes_its_own_schema(schema, tmp_path):
    # The template a person downloads must be fillable into something the
    # upload accepts; otherwise the two have drifted apart.
    path = write_template(schema, tmp_path / "template.xlsx")
    frame = pd.read_excel(path)
    frame.loc[0] = ["ABC-1", "O2", 44.25, "Rh", True, "BATCH-1", ""]

    report = validate_entries(schema, {
        row["Experiment"]: {key: value for key, value in row.items()
                            if key != "Experiment"}
        for row in frame.to_dict(orient="records")})

    assert report.errors == []


# =============================================================================
# Mapping fields
# =============================================================================

@pytest.mark.parametrize("cell, expected", [
    ("Ir=0.02; Ru=0.02", {"Ir": 0.02, "Ru": 0.02}),
    ("900:0.5; 1000 = 2", {"900": 0.5, "1000": 2.0}),
    ("Ir=0.02\nRu=0.02", {"Ir": 0.02, "Ru": 0.02}),
    ("  Ir = 0.02  ", {"Ir": 0.02}),
    ("", {}),
    (None, {}),
    (float("nan"), {}),
    ({"Ir": 0.02}, {"Ir": 0.02}),
])
def test_a_mapping_cell_is_read_however_it_was_typed(cell, expected):
    # An empty Excel cell arrives as NaN, not as an empty string: reading it as
    # the text 'nan' reports a filled-in field nobody filled in.
    assert parse_mapping(cell)[0] == expected


def test_a_pair_that_cannot_be_read_is_reported_not_dropped():
    mapping, problems = parse_mapping("Ir 0.02; Ru=0.02")

    # For a composition, silently dropping one is the difference between
    # "no dopant" and "a dopant we lost".
    assert mapping == {"Ru": 0.02}
    assert problems == ["Ir 0.02"]


def test_an_unreadable_pair_fails_validation():
    field_schema = FieldSchema(name="Dopants [mol%]", type="mapping")
    report = validate_metadata(EntitySchema("x", "X", fields=[field_schema]),
                               {"Dopants [mol%]": "Ir 0.02"})

    assert any("could not read" in error for error in report.errors)


def test_a_dopant_nobody_declared_is_accepted():
    field_schema = FieldSchema(name="Dopants [mol%]", type="mapping",
                               key_options=["Ir", "Ru"])
    report = validate_metadata(EntitySchema("x", "X", fields=[field_schema]),
                               {"Dopants [mol%]": "Ir=0.02; Unobtainium=0.5"})

    # key_options says what is expected, like every other option list here.
    assert report.errors == []


def test_a_profile_derives_the_scalars_its_schema_declares():
    field_schema = FieldSchema(
        name="Temperature steps [°C and h]", type="mapping",
        derived=[{"name": "Peak temperature [°C]", "of": "max_key"},
                 {"name": "Time at peak temperature [h]", "of": "value_at_max_key"}])
    schema = EntitySchema("x", "X", fields=[field_schema])

    prepared = prepare_metadata(
        schema, {"Temperature steps [°C and h]": "900=0.5; 1150=10; 1000=2"})

    # The peak is the largest temperature, not the last one written.
    assert prepared["Peak temperature [°C]"] == 1150.0
    assert prepared["Time at peak temperature [h]"] == 10.0
    assert prepared["Temperature steps [°C and h]"] == {"900": 0.5, "1000": 2.0,
                                                        "1150": 10.0}


def test_a_derived_scalar_is_not_reported_as_undeclared():
    field_schema = FieldSchema(
        name="Steps", type="mapping",
        derived=[{"name": "Peak temperature [°C]", "of": "max_key"}])
    schema = EntitySchema("x", "X", fields=[field_schema])

    report = validate_metadata(schema, prepare_metadata(
        schema, {"Steps": "900=1; 1150=4"}))

    assert report.errors == [] and report.undeclared == []


def test_dopant_names_derive_nothing():
    field_schema = FieldSchema(
        name="Dopants [mol%]", type="mapping",
        derived=[{"name": "Peak", "of": "max_key"}])

    prepared = prepare_metadata(EntitySchema("x", "X", fields=[field_schema]),
                                {"Dopants [mol%]": "Ir=0.02; Ru=0.02"})

    # No key is a number, so there is no peak to record — as opposed to a peak
    # of zero, which would be a measurement nobody made.
    assert "Peak" not in prepared


def test_an_unknown_derived_function_is_refused():
    with pytest.raises(ValueError, match="unknown function"):
        FieldSchema(name="Steps", type="mapping",
                    derived=[{"name": "Peak", "of": "average_of_everything"}])


def test_the_template_shows_how_a_mapping_is_written():
    schema = load_entity_schemas()["finished_semiconductor"]
    _, guide = template_frames(schema)
    dopants = guide[guide["Field"] == "Dopants [mol%]"].iloc[0]

    # The one column whose format cannot be guessed from its name, explained
    # where somebody filling the template is looking.
    assert "=" in dopants["Example"]


# =============================================================================
# References that accept more than one kind of entry
# =============================================================================

def test_a_reference_accepts_what_its_schema_says():
    schemas = load_entity_schemas()

    assert schemas["experiment"].field_named(
        "Catalyst Batch [experiment no.]").accepts == [
            "catalyst_batch", "modified_catalyst_batch"]


def test_an_undeclared_reference_is_left_unchecked():
    # Deriving the accepted kind from the role would read 'precursor_chemical_a'
    # as a type and flag every correct reference in the group's own chain.
    assert FieldSchema(name="X", type="reference", role="precursor_chemical_a").accepts == []


def test_accepts_on_something_that_is_not_a_reference_is_refused():
    with pytest.raises(ValueError, match="not a reference"):
        FieldSchema(name="Operator", type="text", accepts=["experiment"])


def test_the_accepted_types_index_spans_every_schema():
    from pyKES.database.entity_schema import accepted_types

    accepted = accepted_types(load_entity_schemas())

    assert accepted[("experiment", "catalyst_batch")] == [
        "catalyst_batch", "modified_catalyst_batch"]
    assert accepted[("modified_catalyst_batch", "catalyst_batch")] == [
        "catalyst_batch"]
