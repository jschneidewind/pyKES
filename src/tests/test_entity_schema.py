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
        "Catalyst Batch [experiment no.]": {"role": "catalyst_batch"}}
    assert schemas["finished_semiconductor"].reference_instructions() == {
        "Precursor Chemical A": {"role": "precursor_chemical_a"},
        "Precursor Chemical B": {"role": "precursor_chemical_b"}}


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
