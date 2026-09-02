"""
Reusable Streamlit data-upload page for pyKES applications.

The page is structured around two independent uploaders:

* **Metadata Excel** — always rendered. Uploaded sheets are merged into
  ``dataset.overview_df`` by experiment name (existing rows updated, new rows
  appended).
* **Raw-data uploaders** — one per ``FileUploadHandler`` declared in the
  ``DataUploadConfig``. Each handler runs in either ``overview_df_based_processing``
  mode (looking up filenames in columns named by ``file_name_field``) or
  file-list mode (iterating the uploaded files directly).

A per-handler progress bar reports ingestion progress in real time. Ingestion
and reprocessing are not run in one go: both are handed to
`pyKES.streamlit_app.chunked_processing`, which advances them one experiment
per rerun. That is what keeps the bar visible in the stlite browser build,
where a blocking loop would occupy the only event loop and let nothing reach
the screen until it finished.

The page also offers **reprocessing**: rerunning a handler's processing
function against the metadata and raw data already stored in the dataset. That
rebuilds ``processed_data`` — after an algorithm change, for instance — without
needing the original raw-data files, and updates the dataset's version
information.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.database.data_processing import (ingest_experiment,
                                            reprocess_experiment_by_name,
                                            resolve_external_version,
                                            select_experiments_to_reprocess,
                                            select_unprocessed_experiments)
from pyKES.streamlit_app.chunked_processing import (active_job, any_active_job,
                                                    collect_job_results, render_chunked_job,
                                                    start_chunked_job)
from pyKES.streamlit_app.config_interface import DataUploadConfig, FileUploadHandler
from pyKES.database.database_experiments import import_overview_excel
from pyKES.utilities.version_information import describe_version_information


# Session-state key of the experiment multiselect on the reprocessing form
REPROCESS_SELECTION_KEY = "reprocess_selected_experiments"

# Session-state keys of the two chunked processing jobs. The ingestion key is
# suffixed with the handler's storage key, since a page can carry several
# uploaders.
INGESTION_JOB_KEY_TEMPLATE = "ingestion_job_{file_storage_key}"
REPROCESS_JOB_KEY = "reprocess_job"


def _page_job_keys(config: DataUploadConfig) -> list:
    """
    Session-state keys of every chunked job this page can start.

    Parameters
    ----------
    config : DataUploadConfig
        Configuration listing the file handlers.

    Returns
    -------
    list of str
        One ingestion job key per handler, plus the reprocessing job key.
    """

    return [INGESTION_JOB_KEY_TEMPLATE.format(file_storage_key=handler.file_storage_key)
            for handler in config.file_handlers] + [REPROCESS_JOB_KEY]


def render_data_upload(config: DataUploadConfig) -> None:
    """
    Render the data upload page.

    Parameters
    ----------
    config : DataUploadConfig
        External-repo configuration listing file handlers, fresh-dataset
        defaults, and download / auto-save behavior.
    """

     # Title and filename display
    col_title, col_filename = st.columns([3, 1])
    with col_title:
        st.title(config.page_title)
    with col_filename:
        if st.session_state.hdf5_filename:
            st.markdown(f"<p style='text-align: right; font-size: 0.8em; color: gray; margin-top: 1.5em;'>{st.session_state.hdf5_filename}</p>", 
                        unsafe_allow_html=True)

    if config.page_description:
        st.markdown(config.page_description)
    st.divider()

    # No dataset in session state yet -> only offer the "Start Fresh" path.
    if st.session_state.get("experimental_dataset") is None:
        _render_dataset_init(config)
        return

    dataset = st.session_state.experimental_dataset

    st.subheader("1. 📋 Upload Metadata (Excel)")
    _render_metadata_uploader(config, dataset)
    st.divider()

    st.subheader("2. 📥 Upload Raw Data")
    # Render the different raw data uploaders
    for handler in config.file_handlers:
        _render_raw_data_uploaders(handler, dataset, config.external_version)
    st.divider()

    st.subheader("3. ♻️ Reprocess Existing Experiments")
    _render_reprocessing_section(config, dataset)
    st.divider()

    # A running job reruns the page once per experiment. The sections below
    # rewrite the whole HDF5 file and re-derive the statistics on every run,
    # which would dwarf the processing itself, so they wait for it to finish.
    if any_active_job(_page_job_keys(config)):
        return

    st.subheader("4. 📦 Merge HDF5 Files")
    _render_HDF5_merging(config, dataset)
    st.divider()

    st.subheader("5. 💾 Download Dataset")
    _render_download_section(config, dataset)
    st.divider()

    st.subheader("📊 Dataset Overview")
    st.dataframe(st.session_state.experimental_dataset.overview_df)
    st.divider()

    st.subheader("📊 Dataset Statistics")
    _render_dataset_statistics(dataset)
    st.divider()

    st.subheader("🧾 Dataset Provenance")
    _render_version_information(dataset)
    st.divider()


# ---------------------------------------------------------------------------
# Dataset initialization
# ---------------------------------------------------------------------------

def _render_dataset_init(config: DataUploadConfig) -> None:
    """
    Show the empty-state prompt and create a fresh dataset on demand.

    Parameters
    ----------
    config : DataUploadConfig
        Provides the schema (overview columns, mappings, parameters) used to
        seed an empty ``ExperimentalDataset``.
    """
    st.warning(
        "⚠️ No dataset loaded. Upload an HDF5 file on the Home page first, "
        "or start a fresh dataset below."
    )

    if st.button("Start Fresh Dataset"):
        dataset = ExperimentalDataset(
            overview_df=pd.DataFrame(),
            group_mapping=config.group_mapping,
            plotting_instruction=config.plotting_instruction,
            processing_parameters=config.processing_parameters,
        )
        dataset.stamp_version(external_version=config.external_version)

        st.session_state.experimental_dataset = dataset
        st.success("Fresh dataset created.")
        st.rerun()

    st.info("Alternatively, load an existing HDF5 file on the Home page to extend it.")


# ---------------------------------------------------------------------------
# Metadata Excel uploader
# ---------------------------------------------------------------------------

def _render_metadata_uploader(
    config: DataUploadConfig, dataset: ExperimentalDataset
) -> None:
    """
    Render the always-on metadata uploader and merge submissions into
    ``dataset.overview_df``.

    Parameters
    ----------
    config : DataUploadConfig
    dataset : ExperimentalDataset
        Mutated in place — ``overview_df`` is replaced by the merged frame.
    """

    uploaded = st.file_uploader(
        label = '📋 Upload Metadata (Excel)',
        type = ['xlsx', 'xls'],
        help = 'Excel sheet listing experiments. Uploading merges into the dataset overview by experiment name.',
        accept_multiple_files=False,
        key="metadata_excel_uploader",
        )
    
    if uploaded is None:
        return

    incoming_df = import_overview_excel(uploaded, 
                config.metadata_excel_sheet_name)
    
    dataset.update_overview_df(incoming_df, 
                               config.metadata_excel_experiment_column) 

    st.success(
        f"✅ Metadata merged successfully")


# ---------------------------------------------------------------------------
# Raw-data uploaders
# ---------------------------------------------------------------------------

def _stage_uploaded_files(uploaded_files: list) -> str:
    """
    Write the uploaded files to a directory the raw-data reader can read from.

    Parameters
    ----------
    uploaded_files : list of UploadedFile
        Files submitted through the uploader.

    Returns
    -------
    str
        Path of the staging directory.
    """

    # Not a TemporaryDirectory context: the directory has to outlive this
    # script run, since the ingestion is spread over one rerun per experiment.
    # `finish_chunked_job` removes it. Navigating away mid-job leaves it
    # behind — in the browser that is Pyodide's in-memory FS, which goes away
    # with the tab.
    staging_directory = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:
        (Path(staging_directory) / uploaded_file.name).write_bytes(uploaded_file.getbuffer())

    return staging_directory


def _start_ingestion_job(job_key: str,
                         config: FileUploadHandler,
                         dataset: ExperimentalDataset,
                         uploaded_files: list,
                         external_version: Optional[dict]) -> None:
    """
    Stage the uploaded files and register the ingestion run.

    Parameters
    ----------
    job_key : str
        Session-state key of this handler's job.
    config : FileUploadHandler
        Handler defining the processing pipeline.
    dataset : ExperimentalDataset
        Dataset the new experiments are added to.
    uploaded_files : list of UploadedFile
        Files submitted through the uploader.
    external_version : dict, optional
        Provenance of the external app.

    Returns
    -------
    None : None
    """

    experiment_names = select_unprocessed_experiments(dataset, config.overview_df_experiment_column)

    if not experiment_names:
        st.info("No new experiments to process")
        return

    staging_directory = _stage_uploaded_files(uploaded_files)

    start_chunked_job(
        job_key = job_key,
        experiment_names = experiment_names,
        context = {
            'database': dataset,
            'metadata_retrival_function': config.metadata_retrival_function,
            'raw_data_reading_function': config.raw_data_reading_function,
            'processing_function': config.processing_function,
            'overview_df_experiment_column': config.overview_df_experiment_column,
            'directory': Path(staging_directory),
            'external_version': resolve_external_version(dataset, external_version),
        },
        staging_directory = staging_directory,
    )

    st.rerun()


def _render_raw_data_uploaders(config: FileUploadHandler,
                               dataset: ExperimentalDataset,
                               external_version: Optional[dict] = None) -> None:
    """
    Render one handler's uploader and ingest its files on submit.

    The ingestion itself is not run here: it is handed to
    `pyKES.streamlit_app.chunked_processing`, which advances it one experiment
    per rerun so the progress bar is painted while the run is in progress.

    Parameters
    ----------
    config : FileUploadHandler
        Handler defining the uploader and its processing pipeline.
    dataset : ExperimentalDataset
        Dataset the new experiments are added to; mutated in place.
    external_version : dict, optional
        Provenance of the external app, stamped onto the processed experiments.
    """
    job_key = INGESTION_JOB_KEY_TEMPLATE.format(file_storage_key=config.file_storage_key)

    with st.form(key=f"upload_form_{config.file_storage_key}", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            label = config.label,
            type = config.file_type,
            help = config.help_text,
            key = config.file_storage_key,
            accept_multiple_files = True,
        )
        submitted = st.form_submit_button("🚀 Process data", width="stretch")

    if active_job(job_key) is not None:
        render_chunked_job(job_key, ingest_experiment)
        return

    results = collect_job_results(job_key)

    if results is not None:
        _report_processing_results(results, "Processed")
        return

    if not submitted or not uploaded_files:
        return

    _start_ingestion_job(job_key, config, dataset, uploaded_files, external_version)


def _report_processing_results(results: list, verb: str) -> None:
    """
    Summarize the outcome of an ingestion or reprocessing run.

    Parameters
    ----------
    results : list of dict
        Result dicts with ``'success'`` and, on failure, ``'file'`` and
        ``'error'``.
    verb : str
        Past-tense verb used in the success message, e.g. ``"Processed"``.

    Returns
    -------
    None : None
        Messages are written to the current Streamlit container.
    """
    successes = [result for result in results if result["success"]]
    failures = [result for result in results if not result["success"]]

    if successes:
        st.success(f"✓ {verb} {len(successes)} experiment(s) successfully")

    if failures:
        st.error(f"✗ Failed on {len(failures)} experiment(s):")
        for failure in failures:
            st.error(f"**{failure['file']}**: {failure['error']}")


# ---------------------------------------------------------------------------
# Reprocessing of experiments already in the dataset
# ---------------------------------------------------------------------------

def _processing_enabled_handlers(config: DataUploadConfig) -> list:
    """
    Collect the file handlers that define a processing function.

    Parameters
    ----------
    config : DataUploadConfig
        Configuration listing the file handlers.

    Returns
    -------
    handlers : list of FileUploadHandler
        Handlers usable as a reprocessing pipeline.
    """
    return [handler for handler in config.file_handlers
            if handler.processing_function is not None]


def _render_reprocessing_section(config: DataUploadConfig, dataset: ExperimentalDataset) -> None:
    """
    Render the reprocessing form and rerun the processing step on submit.

    Reprocessing reuses the metadata and raw data already held by the dataset,
    so an updated processing function (or an edited overview sheet) can be
    applied to an existing HDF5 file without the original raw-data files.

    Parameters
    ----------
    config : DataUploadConfig
        Supplies the candidate processing pipelines and the external version.
    dataset : ExperimentalDataset
        Dataset whose experiments are reprocessed; mutated in place.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """
    if not dataset.experiments:
        st.info("No experiments in the dataset yet — upload raw data first.")
        return

    handlers = _processing_enabled_handlers(config)

    if not handlers:
        st.info("None of the configured file handlers defines a processing function.")
        return

    st.markdown(
        "Rerun the processing step on the metadata and raw data already stored in "
        "the dataset. The raw-data files are not needed — only `processed_data` is "
        "rebuilt, and the dataset version information is updated accordingly."
    )

    with st.form(key="reprocess_experiments_form", clear_on_submit=False):
        handler_label = st.selectbox(
            "Processing pipeline",
            options=[handler.label for handler in handlers],
            help="Processing function used to rebuild the processed data.",
        )
        selected_experiments = st.multiselect(
            "Experiments to reprocess (all experiments when left empty)",
            options=sorted(dataset.experiments.keys()),
            key=REPROCESS_SELECTION_KEY,
        )
        refresh_metadata = st.checkbox(
            "Refresh metadata from the overview table",
            value=True,
            help="Rerun the metadata retrieval function so edits to the uploaded "
                 "overview sheet take effect. Unchecked, the metadata stored in the "
                 "file is reused unchanged.",
        )
        submitted = st.form_submit_button("♻️ Reprocess", width="stretch")

    if active_job(REPROCESS_JOB_KEY) is not None:
        render_chunked_job(REPROCESS_JOB_KEY, reprocess_experiment_by_name)
        return

    results = collect_job_results(REPROCESS_JOB_KEY)

    if results is not None:
        _report_processing_results(results, "Reprocessed")
        st.info("Download the dataset below to persist the reprocessed results.")
        return

    if not submitted:
        return

    handler = next(candidate for candidate in handlers if candidate.label == handler_label)

    if refresh_metadata and handler.metadata_retrival_function is None:
        st.error(
            f"Handler '{handler.label}' defines no metadata retrieval function; "
            "uncheck 'Refresh metadata' to reprocess with the stored metadata."
        )
        return

    start_chunked_job(
        job_key = REPROCESS_JOB_KEY,
        experiment_names = select_experiments_to_reprocess(dataset, selected_experiments or None),
        context = {
            'database': dataset,
            'processing_function': handler.processing_function,
            'metadata_retrival_function': handler.metadata_retrival_function if refresh_metadata else None,
            'external_version': resolve_external_version(dataset, config.external_version),
        },
    )

    st.rerun()


def _render_HDF5_merging(config: DataUploadConfig, dataset: ExperimentalDataset) -> None:
    '''
    Render the HDF5 merging uploader and merge files into the dataset.
    '''

    with st.form(key = "uploading_HDF5_files_to_merge", clear_on_submit = False):
        uploaded_files = st.file_uploader(
            label = "📦 Upload HDF5 Files to Merge",
            type = ['h5', 'hdf5'],
            help = "Upload one or more HDF5 files containing experiments to merge into the current dataset.",
            key = "merge_hdf5_uploader",
            accept_multiple_files = True,
        )
        submitted = st.form_submit_button("🚀 Merge HDF5 Files", width="stretch")
    
    if not submitted or not uploaded_files:
        return
    
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Save the CURRENT dataset as the first file in the merge
        current_dataset_path = str(Path(tmp_dir) / "_current_dataset.h5")
        dataset.save_to_hdf5(current_dataset_path)

        all_files = [current_dataset_path]

        for uploaded_file in uploaded_files:
            file_path = Path(tmp_dir) / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            all_files.append(str(file_path))

        merged_dataset = ExperimentalDataset.merge_hdf5_files(all_files)

    st.session_state.experimental_dataset = merged_dataset
    st.success(f"✅ Merged {len(uploaded_files)} file(s) into current dataset successfully")
            
# ---------------------------------------------------------------------------
# Results / dataset views
# ---------------------------------------------------------------------------

def _render_download_section(
    config: DataUploadConfig, dataset: ExperimentalDataset
    ) -> None:
    """
    Render the HDF5 download button and a short format note.

    Parameters
    ----------
    config : DataUploadConfig
    dataset : ExperimentalDataset
    """
    st.subheader("💾 Download Dataset")

    if not dataset.experiments:
        st.info("No experiments in dataset. Upload data first to enable downloads.")
        return

    col1, col2 = st.columns([2, 1])
    col1.markdown(
        f"Download your complete dataset as an HDF5 file. "
        f"Contains {len(dataset.experiments)} experiment(s)."
    )

    # Stage HDF5 inside a TemporaryDirectory so the bytes can be read back
    # cross-platform (NamedTemporaryFile reopen behavior differs on Windows).
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, config.output_hdf5_name)
        dataset.save_to_hdf5(tmp_path)
        with open(tmp_path, "rb") as f:
            hdf5_bytes = f.read()

    col2.download_button(
        label="📥 Download HDF5",
        data=hdf5_bytes,
        file_name=config.output_hdf5_name,
        mime="application/x-hdf",
        width="stretch",
    )

    with st.expander("ℹ️ About HDF5 Format"):
        st.markdown(
            "**HDF5** (Hierarchical Data Format 5) is an open binary format for "
            "storing large numerical datasets and metadata together. Files written "
            "by pyKES can be loaded back with ``ExperimentalDataset.load_from_hdf5``."
        )

def _render_version_information(dataset: ExperimentalDataset) -> None:
    """
    Show which code produced the dataset and when it was last touched.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset whose ``version`` dictionary is displayed.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """
    st.markdown(f"`{describe_version_information(dataset.version)}`")

    with st.expander("ℹ️ Full version information"):
        st.json(dataset.version or {})

        experiment_versions = pd.DataFrame([
            {
                "Experiment": exp_name,
                "pyKES version": experiment.version.get("pykes_version", "—"),
                "Last processed": experiment.version.get("last_processed", "—"),
            }
            for exp_name, experiment in sorted(dataset.experiments.items())
        ])

        if not experiment_versions.empty:
            st.markdown("**Per-experiment processing provenance:**")
            st.dataframe(experiment_versions, width="stretch")


def _render_dataset_statistics(dataset: ExperimentalDataset) -> None:
    """
    Show experiment counts, overview row count, and a per-group breakdown.

    Parameters
    ----------
    dataset : ExperimentalDataset
    """
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Experiments Loaded", len(dataset.experiments))
    col2.metric(
        "Overview Records",
        len(dataset.overview_df) if not dataset.overview_df.empty else "—",
    )
    groups = {exp.group for exp in dataset.experiments.values()}
    col3.metric("Experiment Groups", len(groups))

    if dataset.experiments:
        st.markdown("**Experiments by Group:**")
        for group in sorted(groups):
            count = sum(1 for e in dataset.experiments.values() if e.group == group)
            st.markdown(f"- **{group}**: {count} experiments")
