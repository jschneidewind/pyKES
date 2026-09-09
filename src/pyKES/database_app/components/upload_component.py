"""
Contributing data: HDF5 batches from the processing app, and entity sheets.

Anyone signed in may upload. Both routes attribute what they add to the
authenticated user, so the owner-or-admin rule has something to work with.

Uploading never overwrites: a name already in the database is kept alongside the
newcomer under a version suffix. Correcting a mistake is a deliberate edit on
the entry page, not a side effect of uploading a file twice.
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from pyKES.database.entity_schema import (
    EntitySchema,
    FieldSchema,
    TYPE_BOOLEAN,
    TYPE_DATE,
    TYPE_INTEGER,
    TYPE_MAPPING,
    TYPE_MULTISELECT,
    TYPE_NUMBER,
    TYPE_REFERENCE,
    TYPE_SELECT,
    load_entity_schemas,
    template_bytes,
)
from pyKES.database.index_ingest import (
    IngestionError,
    ingest_entity_sheet,
    ingest_hdf5_upload,
)
from pyKES.database.index_references import ReferenceError, read_dangling_references
from pyKES.database.index_query import display_entity_type
from pyKES.database.index_schema import ENTITY_TYPES, analyse_index
from pyKES.database_app.config import DEFAULT_CONFIG, DatabaseAppConfig
from pyKES.database_app.session import index_paths, open_shared_index, read_identity


# =============================================================================
# Display constants
# =============================================================================

# Extensions each uploader accepts.
HDF5_EXTENSIONS = ["h5", "hdf5"]
SHEET_EXTENSIONS = ["xlsx", "csv"]

# Kinds of entry that arrive as measured batches rather than as sheets.
PAYLOAD_ENTITY_TYPES = ("experiment",)

# Entries offered in a reference field's picker. Enough to cover what somebody
# is likely to be linking to, bounded so the form does not load the database.
REFERENCE_OPTION_LIMIT = 200

# How several identifiers are written into one cell, matching what the sheet
# uploader reads back out of one.
REFERENCE_JOIN = "; "

# Session-state key holding this session's staging directory for uploads.
STAGING_DIRECTORY_KEY = "upload_staging_directory"


def stage_upload(uploaded_file) -> Path:
    """
    Write a Streamlit upload to a temporary file ingestion can open.

    Parameters
    ----------
    uploaded_file : UploadedFile
        File as delivered by Streamlit, held in memory.

    Returns
    -------
    staged : Path
        Path to the staged copy, inside this session's staging directory. The
        ingestion stores its own verbatim copy under the file's hash, so the
        staged file is deleted as soon as it has been read.

    Raises
    ------
    IngestionError
        If the upload carries no usable file name.
    """
    # Reduced to its last component before it is joined. The name is whatever
    # the browser sent, and joining it unchanged makes the upload a write to
    # any path the process can reach — `../../data/index.sqlite` among them,
    # since the data root is writable by exactly this user.
    name = Path(uploaded_file.name).name

    if not name or name in (".", ".."):
        raise IngestionError(
            f"Upload has no usable file name ({uploaded_file.name!r}).")

    staged = session_staging_directory() / name
    staged.write_bytes(uploaded_file.getbuffer())

    return staged


def session_staging_directory() -> Path:
    """
    Find, or create, this session's directory for staged uploads.

    One per session rather than one per upload. `mkdtemp` is not reclaimed by
    anything — the operating system does not clean it up, whatever its name
    suggests — so with a 1000 MB upload ceiling an afternoon of staging large
    batches left gigabytes behind. Filling the disk stops SQLite writing and
    breaks the nightly backup, and it reads as a database fault rather than as
    a full disk. Holding the handle in session state means the directory is
    removed when the session ends.

    Returns
    -------
    directory : Path
        Directory to stage this session's uploads in.
    """
    if STAGING_DIRECTORY_KEY not in st.session_state:
        st.session_state[STAGING_DIRECTORY_KEY] = tempfile.TemporaryDirectory(
            prefix="photocat-upload-")

    return Path(st.session_state[STAGING_DIRECTORY_KEY].name)


def report_result(report, connection) -> None:
    """
    Show what an ingestion did, including the things that need attention.

    Parameters
    ----------
    report : IngestionReport
        Result of the ingestion.
    connection : sqlite3.Connection
        Open connection, used to look up references left dangling.

    Returns
    -------
    None : None
    """
    if report.already_ingested:
        st.info(report.summary())
        return

    st.success(report.summary())

    if report.versioned:
        st.warning(
            f"{len(report.versioned)} names were already in the database and "
            f"were kept alongside the existing entries: "
            f"{', '.join(report.versioned[:10])}"
            + ("…" if len(report.versioned) > 10 else "")
            + ". Nothing was overwritten."
        )

    if report.result_conflicts:
        st.error(
            f"These result labels were declared with a different path than an "
            f"earlier upload used: {', '.join(report.result_conflicts)}. Both "
            f"definitions are kept and the labels are flagged on the Admin page."
        )

    dangling = read_dangling_references(connection)
    if dangling:
        st.warning(
            f"{len(dangling)} references point at entries that are not in the "
            f"database yet — for example "
            f"{dangling[0]['source']} → {dangling[0]['target']}. They will "
            f"resolve by themselves when those entries are uploaded."
        )

    if report.undeclared_fields:
        st.info(
            f"{len(report.undeclared_fields)} fields are not in this kind of "
            f"entry's schema: {', '.join(report.undeclared_fields[:12])}"
            + ("…" if len(report.undeclared_fields) > 12 else "")
            + ". They were accepted and are searchable — the schema says what "
              "is expected, not what is allowed — but check them for typos."
        )

    if report.added:
        with st.expander(f"Entries added ({len(report.added)})"):
            st.write(report.added)


# =============================================================================
# The two upload routes
# =============================================================================

def render_hdf5_upload(connection, config: DatabaseAppConfig, identity,
                       schemas: dict) -> None:
    """
    Upload a batch produced by the processing app.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    config : DatabaseAppConfig
        Deployment settings.
    identity : Identity
        Authenticated user the entries are attributed to.
    schemas : dict
        Schemas the uploaded metadata is checked against.

    Returns
    -------
    None : None
    """
    st.subheader("1. Measured Batch (HDF5)")
    st.caption("The file the processing app produces: experiments with their "
               "metadata, raw data and processed data.")

    uploaded = st.file_uploader("HDF5 batch", type=HDF5_EXTENSIONS,
                                key="hdf5_uploader")
    entity_type = st.selectbox("These entries are", PAYLOAD_ENTITY_TYPES,
                               key="hdf5_type", format_func=display_entity_type)

    if uploaded is None or not st.button("Ingest Batch", type="primary"):
        return

    staged = stage_upload(uploaded)

    with st.spinner(f"Ingesting {uploaded.name}…"):
        try:
            report = ingest_hdf5_upload(connection, index_paths(config), staged,
                                        identity.name, entity_type,
                                        schemas=schemas)
        finally:
            # Ingestion has already stored its own verbatim copy under the
            # file's hash, so the staged one is dead weight either way.
            staged.unlink(missing_ok=True)
        analyse_index(connection)

    report_result(report, connection)
    st.cache_data.clear()


def render_sheet_upload(connection, config: DatabaseAppConfig, identity,
                        schemas: dict) -> None:
    """
    Upload a sheet of entries that carry metadata but no measurements.

    This is the route for catalyst batches, semiconductors and precursors — the
    people who make them have no raw traces, and their metadata is what the
    search depends on.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    config : DatabaseAppConfig
        Deployment settings supplying the default reference declaration.
    identity : Identity
        Authenticated user the entries are attributed to.
    schemas : dict
        Schemas the uploaded metadata is checked against.

    Returns
    -------
    None : None
    """
    st.subheader("2. Metadata Sheet (Excel or CSV)")
    st.caption("One row per entry, an id column, and whatever metadata columns "
               "the sheet has. New columns need no migration.")

    uploaded = st.file_uploader("Metadata sheet", type=SHEET_EXTENSIONS,
                                key="sheet_uploader")
    entity_type = st.selectbox("These entries are",
                               [kind for kind in ENTITY_TYPES
                                if kind not in PAYLOAD_ENTITY_TYPES] +
                               list(PAYLOAD_ENTITY_TYPES),
                               key="sheet_type", format_func=display_entity_type)
    identifier_column = st.text_input("ID column", value="Experiment")

    schema = schemas.get(entity_type)
    declared = (schema.reference_instructions() if schema
                else config.reference_instructions_by_type.get(entity_type, {}))
    if declared:
        st.caption("Reference columns declared for this kind of entry: "
                   + ", ".join(f"`{column}` → {instruction['role']}"
                               for column, instruction in declared.items()))
    else:
        st.caption("No reference columns are declared for this kind of entry, "
                   "so it will not link to anything.")

    if uploaded is None or not st.button("Ingest Sheet", type="primary"):
        return

    staged = stage_upload(uploaded)

    with st.spinner(f"Ingesting {uploaded.name}…"):
        try:
            report = ingest_entity_sheet(connection, index_paths(config), staged,
                                         entity_type, identity.name, declared,
                                         identifier_column, schemas=schemas)
        finally:
            staged.unlink(missing_ok=True)
        analyse_index(connection)

    report_result(report, connection)
    st.cache_data.clear()


# =============================================================================
# Entry point
# =============================================================================

def render_upload(config: DatabaseAppConfig = DEFAULT_CONFIG) -> None:
    """
    Render the upload page.

    Parameters
    ----------
    config : DatabaseAppConfig, optional
        Deployment settings.

    Returns
    -------
    None : None
    """
    connection = open_shared_index(str(config.data_root))
    identity = read_identity(config)
    schemas = load_entity_schemas(config.schema_directory)

    st.title("Contribute")
    st.caption(f"Everything you add is attributed to **{identity.name}**, and "
               f"only you or an admin can correct it afterwards.")

    with st.sidebar:
        render_template_downloads(schemas)

    # Ingestion failures are shown rather than swallowed: a rejected upload has
    # a reason the person uploading needs to read, and a schema violation names
    # the entries and fields at fault.
    try:
        render_hdf5_upload(connection, config, identity, schemas)
        st.divider()
        render_sheet_upload(connection, config, identity, schemas)
        st.divider()
        render_entry_form(connection, config, identity, schemas)
    except (IngestionError, ReferenceError) as error:
        st.error(str(error))


# =============================================================================
# Adding one entry through a form
# =============================================================================

def render_field(field_schema: FieldSchema, connection=None):
    """
    Draw the widget one declared field calls for.

    Parameters
    ----------
    field_schema : FieldSchema
        Field to draw, whose type decides the widget and whose options fill it.
    connection : sqlite3.Connection, optional
        Open connection, used only to offer the entries a reference may point
        at. Without one a reference falls back to a typed identifier, which is
        still valid — an entry may name a target that has not been uploaded yet.

    Returns
    -------
    value : Any
        What the user entered, or None where they left it blank.
    """
    label = field_schema.name + (f"  [{field_schema.unit}]" if field_schema.unit
                                 and field_schema.unit not in field_schema.name
                                 else "")
    if field_schema.required:
        label = f"{label} *"

    arguments = {"help": field_schema.help, "key": f"form_{field_schema.name}"}

    if field_schema.type == TYPE_BOOLEAN:
        return st.checkbox(label, value=bool(field_schema.default), **arguments)

    if field_schema.type == TYPE_SELECT:
        return st.selectbox(label, field_schema.options, index=None,
                            placeholder="Select…", **arguments)

    if field_schema.type == TYPE_MULTISELECT:
        return st.multiselect(label, field_schema.options, **arguments)

    if field_schema.type in (TYPE_NUMBER, TYPE_INTEGER):
        # Left as None rather than 0 so an untouched optional number stays
        # empty instead of recording a measurement nobody made.
        return st.number_input(label, value=None,
                               step=1 if field_schema.type == TYPE_INTEGER else None,
                               **arguments)

    if field_schema.type == TYPE_DATE:
        return st.date_input(label, value=None, **arguments)

    if field_schema.type == TYPE_MAPPING:
        return render_mapping_field(field_schema, label, arguments["key"])

    if field_schema.type == TYPE_REFERENCE:
        return render_reference_field(field_schema, label, arguments, connection)

    return st.text_input(label, **arguments)


def render_reference_field(field_schema: FieldSchema, label: str,
                           arguments: dict, connection):
    """
    Offer the entries a reference may point at, and accept one that is not there.

    Which kinds are on offer comes from the field's ``accepts``; a field that
    declares none offers every kind, since inventing a constraint from the role
    name would exclude the group's own chain — a ``precursor_chemical_a`` role
    is filled by a ``precursor_chemical``.

    A field declared ``multiple`` offers several at once and writes them into
    one cell the way a sheet does, so a form entry and an uploaded row are
    indistinguishable afterwards.

    A free-text box sits beside the list because a reference to an entry that
    has not been uploaded yet is legitimate and routine: it is recorded
    unresolved and promoted when the target arrives.

    Parameters
    ----------
    field_schema : FieldSchema
        Reference field being drawn.
    label : str
        Label for the widget.
    arguments : dict
        Shared widget arguments, including its key.
    connection : sqlite3.Connection or None
        Open connection to the index database.

    Returns
    -------
    value : str
        The chosen or typed identifier.
    """
    if connection is None:
        return st.text_input(label, placeholder="Identifier of the linked entry",
                             **arguments)

    options = reference_options(connection, field_schema)

    if field_schema.multiple:
        chosen = REFERENCE_JOIN.join(
            st.multiselect(label, options, **arguments))
    else:
        chosen = st.selectbox(label, options, index=None,
                              placeholder="Select an entry…", **arguments)

    typed = st.text_input(f"…or type identifiers for {field_schema.name}",
                          placeholder="For entries not uploaded yet"
                          if field_schema.multiple
                          else "For an entry not uploaded yet",
                          key=f"{arguments['key']}_typed")

    if chosen and typed:
        return f"{chosen}{REFERENCE_JOIN}{typed}"

    return chosen or typed


def reference_options(connection, field_schema: FieldSchema) -> list:
    """
    List the entries one reference field may point at.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    field_schema : FieldSchema
        Reference field, whose ``accepts`` narrows the list.

    Returns
    -------
    entity_ids : list of str
        Identifiers of entries of an accepted kind, newest first.
    """
    if not field_schema.accepts:
        rows = connection.execute(
            "SELECT entity_id FROM entities ORDER BY updated_at DESC LIMIT ?",
            (REFERENCE_OPTION_LIMIT,))
    else:
        placeholders = ", ".join("?" for _ in field_schema.accepts)
        rows = connection.execute(
            f"""SELECT entity_id FROM entities WHERE entity_type IN ({placeholders})
                ORDER BY updated_at DESC LIMIT ?""",
            list(field_schema.accepts) + [REFERENCE_OPTION_LIMIT])

    return [row["entity_id"] for row in rows]


def render_mapping_field(field_schema: FieldSchema, label: str, widget_key: str):
    """
    Draw a field holding any number of named numbers.

    An editable table is the Streamlit-native way to type a set whose size is
    not known in advance, and it is written back as the same ``name=value``
    text a sheet carries — so a form entry and an uploaded row produce
    identical metadata and neither is a special case afterwards.

    Parameters
    ----------
    field_schema : FieldSchema
        Mapping field being drawn.
    label : str
        Label shown above the table.
    widget_key : str
        Key for the editor.

    Returns
    -------
    value : str
        The pairs, written the way a sheet writes them.
    """
    import pandas as pd

    st.markdown(f"**{label}**")
    if field_schema.help:
        st.caption(field_schema.help)

    names = field_schema.key_label or "Name"
    numbers = field_schema.value_label or "Value"

    edited = st.data_editor(
        pd.DataFrame({names: pd.Series(dtype="str"),
                      numbers: pd.Series(dtype="float")}),
        num_rows="dynamic", width="stretch", hide_index=True,
        key=widget_key,
        column_config={names: st.column_config.SelectboxColumn(
            options=field_schema.key_options)} if field_schema.key_options else None)

    pairs = [f"{row[names]}={row[numbers]}" for row in
             edited.to_dict(orient="records")
             if row[names] and row[numbers] is not None
             and str(row[numbers]).lower() != "nan"]

    return "; ".join(pairs)


def form_row(schema: EntitySchema, identifier: str, values: dict) -> dict:
    """
    Turn the filled-in form into the row an entity sheet would have held.

    Parameters
    ----------
    schema : EntitySchema
        Schema the form was drawn from.
    identifier : str
        Identifier the user gave the new entry.
    values : dict
        Raw widget values, keyed by field name.

    Returns
    -------
    row : dict
        One record, blanks dropped so an untouched optional field is absent
        rather than empty.
    """
    row = {schema.identifier_field: identifier}

    for field_schema in schema.fields:
        value = values.get(field_schema.name)

        if value is None or value == [] or value == "":
            continue

        # Multiselects are stored the way a spreadsheet would hold them, so a
        # form entry and an uploaded row are indistinguishable afterwards.
        row[field_schema.name] = (", ".join(str(item) for item in value)
                                  if isinstance(value, list) else value)

    return row


def render_entry_form(connection, config: DatabaseAppConfig, identity,
                      schemas: dict) -> None:
    """
    Offer a form for adding one entry, built from its schema.

    The submitted form is written as a one-row sheet and ingested through the
    ordinary sheet route. That is not a detour: it means a form entry is stored,
    validated, versioned and rebuilt exactly like an uploaded one, and that
    `rebuild_index` can reconstruct it, which it could not if the form wrote
    straight to the database.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    config : DatabaseAppConfig
        Deployment settings.
    identity : Identity
        Authenticated user the entry is attributed to.
    schemas : dict
        Loaded schemas, keyed by entity type.

    Returns
    -------
    None : None
    """
    st.subheader("3. Add a Single Entry")
    st.caption("For one catalyst batch, semiconductor or chemical, without "
               "making a spreadsheet for it.")

    entity_type = st.selectbox("Kind of entry", sorted(schemas),
                               format_func=lambda kind: schemas[kind].label,
                               key="form_entity_type")
    schema = schemas[entity_type]

    if schema.description:
        st.caption(schema.description)

    with st.form("new_entry", clear_on_submit=False):
        identifier = st.text_input(f"{schema.identifier_field} *",
                                   placeholder="e.g. ABC-014")

        values = {field_schema.name: render_field(field_schema, connection)
                  for field_schema in schema.fields}

        submitted = st.form_submit_button("Add Entry", type="primary")

    if not submitted:
        return

    if not identifier.strip():
        st.error(f"{schema.identifier_field} is required.")
        return

    sheet = session_staging_directory() / f"{identifier.strip()}.xlsx"
    pd.DataFrame([form_row(schema, identifier.strip(), values)]).to_excel(
        sheet, index=False)

    try:
        report = ingest_entity_sheet(connection, index_paths(config), sheet,
                                     entity_type, identity.name,
                                     schema.reference_instructions(),
                                     schema.identifier_field, schemas=schemas)
    finally:
        sheet.unlink(missing_ok=True)
    analyse_index(connection)

    report_result(report, connection)
    st.cache_data.clear()


# =============================================================================
# Templates
# =============================================================================

def render_template_downloads(schemas: dict) -> None:
    """
    Offer the Excel template for each kind of entry.

    The template is generated from the schema, so it cannot drift from what the
    upload will be checked against. Its second sheet says what each column
    expects, since a template of bare headings gets filled in wrongly.

    Parameters
    ----------
    schemas : dict
        Loaded schemas, keyed by entity type.

    Returns
    -------
    None : None
    """
    st.subheader("Excel Templates")
    st.caption("Generated from the same schemas the upload is checked against.")

    entity_type = st.selectbox("Kind of entry", sorted(schemas),
                               format_func=lambda kind: schemas[kind].label,
                               key="template_entity_type")
    schema = schemas[entity_type]

    st.download_button(f"Download {schema.label} Template",
                       data=template_bytes(schema),
                       file_name=f"{entity_type}_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet")

    required = [entry.name for entry in schema.fields if entry.required]
    st.caption(f"{len(schema.fields)} fields, {len(required)} required"
               + (f": {', '.join(required)}." if required else "."))
