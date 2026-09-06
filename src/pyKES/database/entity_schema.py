"""
What each kind of entry is expected to carry.

One YAML file per entity type, in `entity_schemas/`, listing the metadata
fields the group expects, their types, and the options a choice field allows.
The files are meant to be edited by hand — they are the group's own vocabulary,
not application code — so they carry comments and are read at runtime rather
than imported.

The same file drives four things, which is the point of having it:

* the **contribution form**, whose widgets follow from the field types;
* the **Excel template** offered for download;
* the **validation** an upload is checked against;
* the **reference declarations**, since a field of type ``reference`` is simply
  metadata whose value is another entry's identifier.

A schema is a statement of what is *expected*, not a closed list. A metadata
column nobody has declared is reported and then accepted: the database exists to
absorb fields that did not exist when it was built, and a schema that rejected
them would defeat it. Only a declared field can be got *wrong*.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# =============================================================================
# Field types
# =============================================================================

# The widget a field gets on the contribution form, and how it is checked.
TYPE_TEXT = "text"
TYPE_NUMBER = "number"
TYPE_INTEGER = "integer"
TYPE_BOOLEAN = "boolean"
TYPE_SELECT = "select"
TYPE_MULTISELECT = "multiselect"
TYPE_DATE = "date"
TYPE_REFERENCE = "reference"

FIELD_TYPES = (TYPE_TEXT, TYPE_NUMBER, TYPE_INTEGER, TYPE_BOOLEAN, TYPE_SELECT,
               TYPE_MULTISELECT, TYPE_DATE, TYPE_REFERENCE)

# Types whose value must appear in the field's `options` list.
CHOICE_TYPES = (TYPE_SELECT, TYPE_MULTISELECT)

# Types that must parse as a number.
NUMERIC_TYPES = (TYPE_NUMBER, TYPE_INTEGER)

# Directory the shipped schemas live in. A deployment editing its own copy
# points `DatabaseAppConfig.schema_directory` somewhere else.
DEFAULT_SCHEMA_DIRECTORY = Path(__file__).parent / "entity_schemas"

# Strings accepted for a boolean field, because a value that has been through
# Excel and HDF5 arrives as text.
TRUE_STRINGS = ("true", "yes", "1")
FALSE_STRINGS = ("false", "no", "0")

# How many offending values one error message names before it stops listing.
MAX_REPORTED_VALUES = 5


# =============================================================================
# Schema objects
# =============================================================================

@dataclass
class FieldSchema:
    """
    One expected metadata field.

    Parameters
    ----------
    name : str
        Column heading, exactly as it appears in a sheet or in an experiment's
        metadata.
    type : str
        One of `FIELD_TYPES`.
    required : bool
        Whether an entry without it is rejected.
    options : list
        Allowed values, for a choice field.
    unit : str, optional
        Unit shown beside the field; not parsed.
    help : str, optional
        One line shown under the widget.
    default : Any, optional
        Value the form starts on.
    role : str, optional
        For a reference field, the role its edge is recorded under.

    Raises
    ------
    ValueError
        If the type is unknown, or a choice field declares no options — both
        are mistakes in a hand-edited file, and both are worth catching when the
        file is read rather than when somebody fills in a form.
    """

    name: str
    type: str = TYPE_TEXT
    required: bool = False
    options: List[Any] = field(default_factory=list)
    unit: Optional[str] = None
    help: Optional[str] = None
    default: Any = None
    role: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in FIELD_TYPES:
            raise ValueError(
                f"Field '{self.name}' has unknown type '{self.type}'; "
                f"expected one of {FIELD_TYPES}."
            )

        if self.type in CHOICE_TYPES and not self.options:
            raise ValueError(
                f"Field '{self.name}' is a {self.type} but declares no options."
            )

        if self.type == TYPE_REFERENCE and not self.role:
            # Deriving it from the field name would produce a role like
            # 'catalyst_batch_experiment_no', which then names every inherited
            # key. Worth stating explicitly.
            raise ValueError(
                f"Reference field '{self.name}' declares no role."
            )


@dataclass
class EntitySchema:
    """
    What one kind of entry is expected to carry.

    Parameters
    ----------
    entity_type : str
        Stored entity type, matching one of `ENTITY_TYPES`.
    label : str
        Name shown on screen.
    identifier_field : str
        Column holding the entry's identifier.
    description : str
        One line explaining what this kind of entry is.
    fields : list of FieldSchema
        The expected metadata fields, in the order a form and a template should
        present them.
    """

    entity_type: str
    label: str
    identifier_field: str = "Experiment"
    description: str = ""
    fields: List[FieldSchema] = field(default_factory=list)

    def field_named(self, name: str) -> Optional[FieldSchema]:
        """
        Find one declared field by name.

        Parameters
        ----------
        name : str
            Field name to look for.

        Returns
        -------
        field_schema : FieldSchema or None
            The field, or None when it is not declared.
        """
        return next((entry for entry in self.fields if entry.name == name), None)

    def reference_instructions(self) -> Dict[str, Dict[str, str]]:
        """
        Derive the reference declaration from the fields of type ``reference``.

        A reference is metadata whose value is another entry's identifier, so
        the schema already says which columns those are. Deriving it here keeps
        one statement of the group's chain rather than two that can disagree.

        Returns
        -------
        instructions : dict
            ``{column: {'role': role}}`` for every reference field.
        """
        return {entry.name: {"role": entry.role}
                for entry in self.fields if entry.type == TYPE_REFERENCE}


# =============================================================================
# Loading
# =============================================================================

def load_entity_schema(path: Path) -> EntitySchema:
    """
    Read one schema file.

    Parameters
    ----------
    path : Path
        YAML file to read.

    Returns
    -------
    schema : EntitySchema
        The parsed schema.

    Raises
    ------
    ValueError
        If the file declares no entity type, or a field is malformed.
    """
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    if not document.get("entity_type"):
        raise ValueError(f"{path.name} declares no entity_type.")

    return EntitySchema(
        entity_type=document["entity_type"],
        label=document.get("label", document["entity_type"]),
        identifier_field=document.get("identifier_field", "Experiment"),
        description=document.get("description", ""),
        fields=[FieldSchema(**entry) for entry in document.get("fields", [])],
    )


def load_entity_schemas(directory: Optional[Path] = None) -> Dict[str, EntitySchema]:
    """
    Read every schema in a directory.

    Parameters
    ----------
    directory : Path, optional
        Directory of ``.yaml`` files. Defaults to the shipped schemas.

    Returns
    -------
    schemas : dict
        Mapping of entity type to its schema. Empty when the directory does not
        exist, which simply means nothing is validated.
    """
    directory = Path(directory or DEFAULT_SCHEMA_DIRECTORY)

    if not directory.is_dir():
        return {}

    schemas = {}
    for path in sorted(directory.glob("*.yaml")):
        schema = load_entity_schema(path)
        schemas[schema.entity_type] = schema

    return schemas


# =============================================================================
# Validation
# =============================================================================

@dataclass
class ValidationReport:
    """
    What checking one upload against a schema found.

    Parameters
    ----------
    errors : list of str
        Violations of the declared schema: a missing required field, a value
        outside the allowed options, a number that is not one. These block the
        upload.
    undeclared : list of str
        Fields present in the data but not in the schema. Reported so a typo is
        visible, never an error: absorbing fields that did not exist when the
        database was built is the point of the design.
    """

    errors: List[str] = field(default_factory=list)
    undeclared: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Whether anything at all was found."""
        return bool(self.errors or self.undeclared)


def coerce_boolean(value: Any) -> Optional[bool]:
    """
    Interpret a value that should be a boolean.

    Parameters
    ----------
    value : Any
        Raw value, possibly the string a spreadsheet produced.

    Returns
    -------
    flag : bool or None
        The boolean, or None when the value does not express one.
    """
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False

    return None


def check_value(field_schema: FieldSchema, value: Any) -> Optional[str]:
    """
    Check one value against the field that declares it.

    Parameters
    ----------
    field_schema : FieldSchema
        Declared field.
    value : Any
        Value found in the data.

    Returns
    -------
    error : str or None
        What is wrong with the value, or None when it is acceptable. A blank
        value is acceptable here; whether it may be blank is the required check,
        which is made separately.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None

    if field_schema.type in NUMERIC_TYPES:
        try:
            float(value)
        except (TypeError, ValueError):
            return f"'{field_schema.name}' expects a number, found '{value}'."
        return None

    if field_schema.type == TYPE_BOOLEAN and coerce_boolean(value) is None:
        return (f"'{field_schema.name}' expects true or false, "
                f"found '{value}'.")

    if field_schema.type == TYPE_SELECT:
        allowed = [str(option) for option in field_schema.options]
        if str(value) not in allowed:
            return (f"'{field_schema.name}' expects one of {allowed}, "
                    f"found '{value}'.")

    if field_schema.type == TYPE_MULTISELECT:
        allowed = {str(option) for option in field_schema.options}
        chosen = value if isinstance(value, (list, tuple)) else \
            re.split(r"[;,]", str(value))
        unexpected = [str(item).strip() for item in chosen
                      if str(item).strip() and str(item).strip() not in allowed]
        if unexpected:
            return (f"'{field_schema.name}' allows {sorted(allowed)}, "
                    f"found {unexpected}.")

    return None


def validate_metadata(schema: EntitySchema,
                      metadata: Dict[str, Any],
                      entry_name: str = "") -> ValidationReport:
    """
    Check one entry's metadata against its schema.

    Parameters
    ----------
    schema : EntitySchema
        Schema for this kind of entry.
    metadata : dict
        The entry's metadata.
    entry_name : str, optional
        Identifier used to name the entry in messages.

    Returns
    -------
    report : ValidationReport
        Errors and undeclared fields.
    """
    prefix = f"{entry_name}: " if entry_name else ""
    report = ValidationReport()

    for field_schema in schema.fields:
        value = metadata.get(field_schema.name)
        blank = value is None or (isinstance(value, str) and not value.strip())

        if field_schema.required and blank:
            report.errors.append(f"{prefix}'{field_schema.name}' is required.")
            continue

        problem = check_value(field_schema, value)
        if problem:
            report.errors.append(f"{prefix}{problem}")

    declared = {field_schema.name for field_schema in schema.fields}
    declared.add(schema.identifier_field)
    report.undeclared = [name for name in metadata if name not in declared]

    return report


def validate_entries(schema: EntitySchema,
                     entries: Dict[str, Dict[str, Any]]) -> ValidationReport:
    """
    Check a whole upload before any of it is written.

    Validating the batch up front rather than row by row keeps an ingestion
    all-or-nothing: a file with one bad row is rejected whole, instead of
    leaving half its experiments in the database.

    Parameters
    ----------
    schema : EntitySchema
        Schema for this kind of entry.
    entries : dict
        Mapping of entry name to its metadata.

    Returns
    -------
    report : ValidationReport
        Errors across every entry, and the union of the undeclared fields.
    """
    combined = ValidationReport()
    undeclared = set()

    for name, metadata in entries.items():
        report = validate_metadata(schema, metadata, name)
        combined.errors.extend(report.errors)
        undeclared.update(report.undeclared)

    combined.undeclared = sorted(undeclared)

    return combined


# =============================================================================
# Template generation
# =============================================================================

def template_frames(schema: EntitySchema) -> tuple:
    """
    Build the sheets of the Excel template for one kind of entry.

    Parameters
    ----------
    schema : EntitySchema
        Schema to build a template for.

    Returns
    -------
    entries : pandas.DataFrame
        Empty frame whose columns are the identifier and every declared field —
        the sheet a person fills in.
    guide : pandas.DataFrame
        One row per field, saying what it expects. A template whose columns are
        unexplained gets filled in wrongly.
    """
    import pandas as pd

    columns = [schema.identifier_field] + [entry.name for entry in schema.fields]

    guide = pd.DataFrame([{
        "Field": entry.name,
        "Type": entry.type,
        "Required": "Yes" if entry.required else "No",
        "Unit": entry.unit or "",
        "Allowed values": ", ".join(str(option) for option in entry.options),
        "Notes": entry.help or "",
    } for entry in schema.fields])

    return pd.DataFrame(columns=columns), guide


def write_template(schema: EntitySchema, path: Path) -> Path:
    """
    Write the Excel template for one kind of entry.

    Parameters
    ----------
    schema : EntitySchema
        Schema to build a template for.
    path : Path
        File to write.

    Returns
    -------
    path : Path
        The written file.
    """
    import pandas as pd

    entries, guide = template_frames(schema)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        entries.to_excel(writer, sheet_name="Sheet1", index=False)
        guide.to_excel(writer, sheet_name="Field guide", index=False)

    return path
