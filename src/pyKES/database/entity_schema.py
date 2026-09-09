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

import io
import math
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
TYPE_MAPPING = "mapping"

FIELD_TYPES = (TYPE_TEXT, TYPE_NUMBER, TYPE_INTEGER, TYPE_BOOLEAN, TYPE_SELECT,
               TYPE_MULTISELECT, TYPE_DATE, TYPE_REFERENCE, TYPE_MAPPING)

# Types whose value must appear in the field's `options` list.
CHOICE_TYPES = (TYPE_SELECT, TYPE_MULTISELECT)

# Types that must parse as a number.
NUMERIC_TYPES = (TYPE_NUMBER, TYPE_INTEGER)


# =============================================================================
# Mapping fields
# =============================================================================

# How a mapping is written in one spreadsheet cell: `Ir=0.02; Ru=0.02; Cr=0.03`.
# Pairs are separated by a semicolon or a newline — not a comma, which would be
# ambiguous wherever Excel writes a decimal comma. Both `=` and `:` separate a
# name from its value, because people write both.
MAPPING_PAIR_SEPARATORS = ";\n"
MAPPING_VALUE_SEPARATORS = "=:"

# Shown in the template's field guide so the format is visible where it is
# filled in rather than only in the documentation.
MAPPING_EXAMPLE = "Ir=0.02; Ru=0.02; Cr=0.03"

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
# Scalars derived from a mapping
# =============================================================================

def numeric_keys(mapping: Dict[str, Any]) -> List[float]:
    """
    Read the keys of a mapping that are numbers.

    Parameters
    ----------
    mapping : dict
        Parsed mapping value.

    Returns
    -------
    keys : list of float
        The numeric keys, ascending. Non-numeric keys are skipped, so a mapping
        of dopant names simply has none and derives nothing.
    """
    values = []

    for key in mapping:
        try:
            values.append(float(key))
        except (TypeError, ValueError):
            continue

    return sorted(values)


def peak_key(mapping: Dict[str, Any]) -> Optional[float]:
    """
    The largest numeric key of a mapping — the peak of a profile.

    Parameters
    ----------
    mapping : dict
        Parsed mapping value.

    Returns
    -------
    peak : float or None
        The largest key, or None when none of them is a number.
    """
    keys = numeric_keys(mapping)

    return keys[-1] if keys else None


def value_at_peak_key(mapping: Dict[str, Any]) -> Optional[Any]:
    """
    The value held at a mapping's largest numeric key.

    For a temperature profile that is the time spent at the peak temperature,
    which is the quantity a synthesis is usually compared on.

    Parameters
    ----------
    mapping : dict
        Parsed mapping value.

    Returns
    -------
    value : Any or None
        The value at the peak, or None when no key is a number.
    """
    peak = peak_key(mapping)

    if peak is None:
        return None

    # The mapping's keys are strings, since that is what JSON holds, so the
    # numeric peak has to be matched back against them numerically.
    return next(value for key, value in mapping.items()
                if _is_number(key) and float(key) == peak)


def _is_number(value: Any) -> bool:
    """
    Whether a value parses as a number.

    Parameters
    ----------
    value : Any
        Value to test.

    Returns
    -------
    numeric : bool
        True when ``float`` accepts it.
    """
    try:
        float(value)
    except (TypeError, ValueError):
        return False

    return True


# What a mapping field may derive. Each entry turns the whole mapping into one
# number, which is then stored as ordinary metadata and so gets a filter, a
# column and inheritance without any further work.
DERIVED_FUNCTIONS = {
    "max_key": peak_key,
    "value_at_max_key": value_at_peak_key,
}


# =============================================================================
# Schema objects
# =============================================================================

@dataclass
class DerivedScalar:
    """
    A single number a mapping field derives.

    A temperature profile is a sequence, and the questions asked of it are
    aggregates — the peak reached, the time held there — which no per-key slider
    expresses. Deriving them at ingestion turns each into an ordinary metadata
    field, so it gets a slider, a table column and inheritance for free.

    Parameters
    ----------
    name : str
        Metadata field the derived value is stored under.
    of : str
        One of `DERIVED_FUNCTIONS`.
    unit : str, optional
        Unit shown beside it; not parsed.

    Raises
    ------
    ValueError
        If the function is unknown — a mistake in a hand-edited file, worth
        catching when the file is read rather than at ingestion.
    """

    name: str
    of: str
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if self.of not in DERIVED_FUNCTIONS:
            raise ValueError(
                f"Derived field '{self.name}' asks for unknown function "
                f"'{self.of}'; expected one of {sorted(DERIVED_FUNCTIONS)}."
            )


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
    accepts : list of str, optional
        For a reference field, the kinds of entry it may point at. Left empty
        the reference is simply not checked, which is deliberately not the same
        as deriving it from the role: a role names the *relationship*, so
        `precursor_chemical_a` is filled by a `precursor_chemical`, and reading
        the role as a type would flag every correct reference in the group's
        own chain. It is guidance for the form and a diagnostic for the admin
        page, never a storage constraint — an edge records no target type, and
        that is what lets a new kind of entry join an existing chain without a
        migration.
    key_label, value_label : str, optional
        For a mapping field, what its names and its numbers are called —
        ``Element`` and ``mol%`` for a dopant field. Used to label the filter.
    multiple : bool, optional
        For a reference field, whether it may name several entries in one cell —
        the precursor chemicals a semiconductor was made from, written
        ``EA-1; EA-2; EA-3``. Their metadata is then contributed by all of them,
        and a filter on it asks whether any one satisfies it.
    key_options : list, optional
        For a mapping field, the names expected. Like every other option list
        this says what is *expected*: a name nobody declared is reported and
        then accepted.
    derived : list, optional
        For a mapping field, the scalars computed from it at ingestion.

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
    accepts: List[str] = field(default_factory=list)
    multiple: bool = False
    key_label: Optional[str] = None
    value_label: Optional[str] = None
    key_options: List[Any] = field(default_factory=list)
    derived: List[Any] = field(default_factory=list)

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

        if self.accepts and self.type != TYPE_REFERENCE:
            raise ValueError(
                f"Field '{self.name}' declares 'accepts' but is a "
                f"{self.type}, not a reference."
            )

        if self.multiple and self.type != TYPE_REFERENCE:
            raise ValueError(
                f"Field '{self.name}' declares 'multiple' but is a "
                f"{self.type}, not a reference."
            )

        if self.derived and self.type != TYPE_MAPPING:
            raise ValueError(
                f"Field '{self.name}' declares derived scalars but is a "
                f"{self.type}, not a mapping."
            )

        self.derived = [entry if isinstance(entry, DerivedScalar)
                        else DerivedScalar(**entry) for entry in self.derived]

    def derived_names(self) -> List[str]:
        """
        Name the metadata fields this one derives.

        Returns
        -------
        names : list of str
            Field names, empty for anything but a mapping that declares them.
        """
        return [entry.name for entry in self.derived]


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
            ``{column: {'role': role, 'accepts': [entity_type, ...]}}`` for
            every reference field. ``extract_references`` reads only the role;
            the accepted kinds travel with it so the form and the admin page do
            not have to reach back into the schema.
        """
        return {entry.name: {"role": entry.role, "accepts": list(entry.accepts),
                             "multiple": entry.multiple}
                for entry in self.fields if entry.type == TYPE_REFERENCE}

    def mapping_fields(self) -> List[FieldSchema]:
        """
        List the fields whose value is a mapping.

        Returns
        -------
        fields : list of FieldSchema
            Mapping fields, in declaration order.
        """
        return [entry for entry in self.fields if entry.type == TYPE_MAPPING]

    def declared_names(self) -> set:
        """
        Name everything the schema accounts for.

        Returns
        -------
        names : set of str
            Declared fields, the identifier, and the scalars mapping fields
            derive — which are written by ingestion and so must not be reported
            back as fields nobody declared.
        """
        names = {entry.name for entry in self.fields}
        names.add(self.identifier_field)

        for entry in self.fields:
            names.update(entry.derived_names())

        return names


def accepted_types(schemas: Dict[str, EntitySchema]) -> Dict[tuple, List[str]]:
    """
    Index which kinds of entry each role may point at.

    Parameters
    ----------
    schemas : dict
        Mapping of entity type to its schema.

    Returns
    -------
    accepted : dict
        ``{(source_entity_type, role): [entity_type, ...]}``.
    """
    return {(entity_type, entry.role): list(entry.accepts)
            for entity_type, schema in schemas.items()
            for entry in schema.fields if entry.type == TYPE_REFERENCE}


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


def parse_mapping(value: Any) -> tuple:
    """
    Read a mapping written in one cell into a dictionary.

    ``Ir=0.02; Ru=0.02; Cr=0.03`` becomes ``{'Ir': 0.02, 'Ru': 0.02,
    'Cr': 0.03}``. That form was chosen over JSON in a cell because it is
    typable in Excel without quoting, and over one column per name because the
    number of names is not known in advance.

    Parameters
    ----------
    value : Any
        Cell contents, or a mapping that has already been parsed — an HDF5
        upload can carry one directly.

    Returns
    -------
    mapping : dict
        Name to value, values numeric where they parse as numbers.
    problems : list of str
        Pairs that could not be read. A pair that does not parse is reported
        rather than skipped: for a composition, silently dropping one is the
        difference between "no dopant" and "a dopant we lost".
    """
    if is_blank(value):
        return {}, []

    if isinstance(value, dict):
        return {str(key): _as_number(item) for key, item in value.items()}, []

    mapping, problems = {}, []

    for pair in re.split(f"[{MAPPING_PAIR_SEPARATORS}]", str(value)):
        if not pair.strip():
            continue

        parts = re.split(f"[{MAPPING_VALUE_SEPARATORS}]", pair, maxsplit=1)

        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            problems.append(pair.strip())
            continue

        mapping[parts[0].strip()] = _as_number(parts[1].strip())

    return mapping, problems


def is_blank(value: Any) -> bool:
    """
    Whether a value is empty.

    An empty Excel cell reaches here as a float NaN rather than as an empty
    string, so a check for one alone reads it as the text 'nan' and reports a
    filled-in field nobody filled in.

    Parameters
    ----------
    value : Any
        Value from a sheet, a form or an HDF5 file.

    Returns
    -------
    blank : bool
        True for None, whitespace, and NaN.
    """
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return isinstance(value, str) and not value.strip()


def _as_number(value: Any) -> Any:
    """
    Read a value as a number where it is one.

    Parameters
    ----------
    value : Any
        Value from one side of a mapping pair.

    Returns
    -------
    parsed : Any
        A float where the text parses as one, otherwise the value unchanged. A
        mapping of names to text is unusual but not wrong, and rejecting it
        here would be the schema deciding what the group may record.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def prepare_metadata(schema: Optional[EntitySchema],
                     metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the values a schema describes and add the scalars it derives.

    Applied before an entry is validated *and* before it is written, from the
    same function, so what is checked is exactly what is stored.

    Parameters
    ----------
    schema : EntitySchema or None
        Schema for this kind of entry. None leaves the metadata untouched,
        which is what an unschema'd kind of entry should do.
    metadata : dict
        The entry's raw metadata.

    Returns
    -------
    prepared : dict
        Metadata with every mapping field parsed into a dictionary and every
        declared derived scalar added beside it.
    """
    if schema is None:
        return dict(metadata)

    prepared = dict(metadata)

    for field_schema in schema.mapping_fields():
        mapping, _ = parse_mapping(prepared.get(field_schema.name))

        if not mapping:
            continue

        prepared[field_schema.name] = mapping

        for derived in field_schema.derived:
            value = DERIVED_FUNCTIONS[derived.of](mapping)
            if value is not None:
                prepared[derived.name] = value

    return prepared


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
    if is_blank(value):
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

    if field_schema.type == TYPE_MAPPING:
        return check_mapping(field_schema, value)

    return None


def check_mapping(field_schema: FieldSchema, value: Any) -> Optional[str]:
    """
    Check one mapping cell.

    Parameters
    ----------
    field_schema : FieldSchema
        Declared mapping field.
    value : Any
        Cell contents.

    Returns
    -------
    error : str or None
        What is wrong, or None. A pair that cannot be read is an error, since
        it means a value was written and not recorded. A name nobody declared
        is not: `key_options` says what is *expected*, like every other option
        list here.
    """
    mapping, problems = parse_mapping(value)

    if problems:
        return (f"'{field_schema.name}' expects pairs like "
                f"'{MAPPING_EXAMPLE}'; could not read "
                f"{problems[:MAX_REPORTED_VALUES]}.")

    unreadable = [name for name, item in mapping.items()
                  if not isinstance(item, (int, float))]
    if unreadable:
        return (f"'{field_schema.name}' expects a number for each name; "
                f"{unreadable[:MAX_REPORTED_VALUES]} carry text.")

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
        blank = is_blank(value)

        if field_schema.required and blank:
            report.errors.append(f"{prefix}'{field_schema.name}' is required.")
            continue

        problem = check_value(field_schema, value)
        if problem:
            report.errors.append(f"{prefix}{problem}")

    report.undeclared = [name for name in metadata
                         if name not in schema.declared_names()]

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
        "Allowed values": ", ".join(str(option) for option in
                                    (entry.options or entry.key_options)),
        "Example": field_example(entry),
        "Notes": entry.help or "",
    } for entry in schema.fields])

    return pd.DataFrame(columns=columns), guide


def field_example(field_schema: FieldSchema) -> str:
    """
    Show how a field is written, where the format is not obvious.

    A mapping column is the one that cannot be guessed from its name, and the
    template is where somebody is looking when they need to know.

    Parameters
    ----------
    field_schema : FieldSchema
        Field to describe.

    Returns
    -------
    example : str
        An example cell, empty where the field needs none.
    """
    if field_schema.multiple:
        return "EA-1; EA-2; EA-3"

    if field_schema.type != TYPE_MAPPING:
        return ""

    if not field_schema.key_options:
        return MAPPING_EXAMPLE

    return "; ".join(f"{option}=0.02"
                     for option in field_schema.key_options[:3])


def template_bytes(schema: EntitySchema) -> bytes:
    """
    Build the Excel template for one kind of entry in memory.

    The application offers this as a download, which needs the bytes rather
    than a file — writing one out per render left a temporary directory behind
    on every keystroke.

    Parameters
    ----------
    schema : EntitySchema
        Schema to build a template for.

    Returns
    -------
    content : bytes
        The ``.xlsx`` file.
    """
    import pandas as pd

    entries, guide = template_frames(schema)
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        entries.to_excel(writer, sheet_name="Sheet1", index=False)
        guide.to_excel(writer, sheet_name="Field guide", index=False)

    return buffer.getvalue()


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
    Path(path).write_bytes(template_bytes(schema))

    return Path(path)
