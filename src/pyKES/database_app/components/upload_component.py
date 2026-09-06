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

import streamlit as st

from pyKES.database.index_ingest import (
    IngestionError,
    ingest_entity_sheet,
    ingest_hdf5_upload,
)
from pyKES.database.index_references import ReferenceError, read_dangling_references
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
        Path to the staged copy. It lives in a temporary directory that the
        operating system reclaims; the ingestion stores its own verbatim copy
        under the file's hash.
    """
    staged_directory = Path(tempfile.mkdtemp())
    staged = staged_directory / uploaded_file.name
    staged.write_bytes(uploaded_file.getbuffer())

    return staged


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

    if report.added:
        with st.expander(f"Entries added ({len(report.added)})"):
            st.write(report.added)


# =============================================================================
# The two upload routes
# =============================================================================

def render_hdf5_upload(connection, config: DatabaseAppConfig, identity) -> None:
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

    Returns
    -------
    None : None
    """
    st.subheader("1. Measured batch (HDF5)")
    st.caption("The file the processing app produces: experiments with their "
               "metadata, raw data and processed data.")

    uploaded = st.file_uploader("HDF5 batch", type=HDF5_EXTENSIONS,
                                key="hdf5_uploader")
    entity_type = st.selectbox("These entries are", PAYLOAD_ENTITY_TYPES,
                               key="hdf5_type")

    if uploaded is None or not st.button("Ingest batch", type="primary"):
        return

    with st.spinner(f"Ingesting {uploaded.name}…"):
        report = ingest_hdf5_upload(connection, index_paths(config),
                                    stage_upload(uploaded), identity.name,
                                    entity_type)
        analyse_index(connection)

    report_result(report, connection)
    st.cache_data.clear()


def render_sheet_upload(connection, config: DatabaseAppConfig, identity) -> None:
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

    Returns
    -------
    None : None
    """
    st.subheader("2. Metadata sheet (Excel or CSV)")
    st.caption("One row per entry, an id column, and whatever metadata columns "
               "the sheet has. New columns need no migration.")

    uploaded = st.file_uploader("Metadata sheet", type=SHEET_EXTENSIONS,
                                key="sheet_uploader")
    entity_type = st.selectbox("These entries are",
                               [kind for kind in ENTITY_TYPES
                                if kind not in PAYLOAD_ENTITY_TYPES] +
                               list(PAYLOAD_ENTITY_TYPES),
                               key="sheet_type")
    identifier_column = st.text_input("Id column", value="Experiment")

    declared = config.reference_instructions_by_type.get(entity_type, {})
    if declared:
        st.caption("Reference columns declared for this kind of entry: "
                   + ", ".join(f"`{column}` → {instruction['role']}"
                               for column, instruction in declared.items()))
    else:
        st.caption("No reference columns are declared for this kind of entry, "
                   "so it will not link to anything.")

    if uploaded is None or not st.button("Ingest sheet", type="primary"):
        return

    with st.spinner(f"Ingesting {uploaded.name}…"):
        report = ingest_entity_sheet(connection, index_paths(config),
                                     stage_upload(uploaded), entity_type,
                                     identity.name, declared, identifier_column)
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

    st.title("Contribute")
    st.caption(f"Everything you add is attributed to **{identity.name}**, and "
               f"only you or an admin can correct it afterwards.")

    # Ingestion failures are shown rather than swallowed: a rejected upload has
    # a reason the person uploading needs to read.
    try:
        render_hdf5_upload(connection, config, identity)
        st.divider()
        render_sheet_upload(connection, config, identity)
    except (IngestionError, ReferenceError) as error:
        st.error(str(error))
