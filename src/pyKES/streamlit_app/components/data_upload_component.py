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

from pyKES.database.database_experiments import (ExperimentalDataset,
                                                 describe_experiment_names,
                                                 describe_metadata_divergences,
                                                 describe_skipped_experiments)
from pyKES.database.data_processing import (ingest_experiment,
                                            reprocess_experiment_by_name,
                                            resolve_external_version,
                                            select_experiments_needing_reprocessing,
                                            select_experiments_to_reprocess,
                                            select_unprocessed_experiments)
from pyKES.database.metadata_editing import columns_invalidating_processing
from pyKES.streamlit_app.chunked_processing import (active_job, any_active_job,
                                                    collect_job_results, render_chunked_job,
                                                    start_chunked_job)
from pyKES.streamlit_app.components.metadata_editor import (discard_metadata_editor_state,
                                                             render_metadata_editor)
from pyKES.streamlit_app.config_interface import DataUploadConfig, FileUploadHandler
from pyKES.database.database_experiments import import_overview_excel
from pyKES.utilities.version_information import describe_version_information


# Session-state key holding the file_id of the metadata workbook already
# merged into overview_df. `st.file_uploader` keeps its file for the whole
# session, so without this the sheet was re-read and re-merged on *every*
# rerun — and since an uploaded sheet takes precedence over stored values,
# that silently reverted every metadata edit one rerun after it was applied,
# while leaving the experiment's own metadata holding the edit.
MERGED_METADATA_FILE_KEY = "merged_metadata_excel_file_id"

# Session-state key of the experiment multiselect on the reprocessing form
REPROCESS_SELECTION_KEY = "reprocess_selected_experiments"

# Session-state key of the checkbox restricting reprocessing to the
# experiments whose metadata has changed since they were processed
REPROCESS_STALE_ONLY_KEY = "reprocess_only_stale_experiments"

# How the dataset already loaded is named in a merge report. The merge itself
# runs on temporary files, so without a label of its own the report would name
# the dataset after a path the user never saw.
CURRENT_DATASET_MERGE_LABEL = "the dataset already loaded"

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

    # The editor comes before the reprocessing section, and that ordering is
    # load-bearing: applying an edit makes the reprocessing warning appear, and
    # anything appearing *above* the grid pushes the grid down by its own
    # height. Below it, the warning costs the editor nothing.
    job_running = any_active_job(_page_job_keys(config))

    st.subheader("3. ✏️ Edit Metadata")
    _render_metadata_section(config, dataset, job_running)
    st.divider()

    st.subheader("4. ♻️ Reprocess Existing Experiments")
    _render_reprocessing_section(config, dataset)
    st.divider()

    # A running job reruns the page once per experiment. The sections below
    # rewrite the whole HDF5 file and re-derive the statistics on every run,
    # which would dwarf the processing itself, so they wait for it to finish.
    if job_running:
        return

    st.subheader("5. 📦 Merge HDF5 Files")
    _render_HDF5_merging(config, dataset)
    st.divider()

    st.subheader("6. 💾 Download Dataset")
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


def _render_metadata_section(config: DataUploadConfig,
                             dataset: ExperimentalDataset,
                             job_running: bool) -> None:
    """
    Render the metadata editor, or say why it is not editable right now.

    Parameters
    ----------
    config : DataUploadConfig
        Page configuration, handed on to the editor.
    dataset : ExperimentalDataset
        Dataset whose metadata is edited.
    job_running : bool
        Whether an ingestion or reprocessing run is in progress.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.

    Notes
    -----
    The editor sits above the reprocessing section, which is before the
    active-job guard, so unlike the sections after that guard it would
    otherwise render while a run is stepping through the experiments. A grid
    and a processing run writing to the same overview table at the same time is
    worth refusing rather than resolving.
    """

    if job_running:
        st.info("Metadata editing pauses while experiments are being processed — "
                "the grid and the run would be writing to the same overview table.")
        return

    render_metadata_editor(config, dataset)


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
        help = 'Excel sheet listing experiments. Uploading merges into the dataset overview by '
               'experiment name, and takes precedence over metadata edited in section 3.',
        accept_multiple_files=False,
        key="metadata_excel_uploader",
        )

    if uploaded is None:
        # Clearing the widget lets the same workbook be uploaded again.
        st.session_state.pop(MERGED_METADATA_FILE_KEY, None)
        return

    if st.session_state.get(MERGED_METADATA_FILE_KEY) == uploaded.file_id:
        st.success("✅ Metadata merged successfully")
        return

    incoming_df = import_overview_excel(uploaded, 
                config.metadata_excel_sheet_name)
    
    # Without the declared columns every difference would clear the processed
    # flag, so a corrected comment would cost a reprocessing run.
    dataset.update_overview_df(incoming_df,
                               config.metadata_excel_experiment_column,
                               columns_invalidating_processing(dataset))

    st.session_state[MERGED_METADATA_FILE_KEY] = uploaded.file_id

    # The sheet has just overridden whatever was edited, so the grid must not
    # replay its client-side edits back over it.
    discard_metadata_editor_state()

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


def _handler_experiment_column(handler: FileUploadHandler, config: DataUploadConfig) -> str:
    """
    Resolve which overview column names the experiments for a handler.

    Parameters
    ----------
    handler : FileUploadHandler
        Handler whose pipeline is about to run.
    config : DataUploadConfig
        Page configuration, supplying the fallback used when the handler
        leaves the column unset.

    Returns
    -------
    str
        Column of ``overview_df`` holding the experiment names.
    """

    return handler.overview_df_experiment_column or config.metadata_excel_experiment_column


def _render_stale_experiment_warning(stale_experiments: list) -> None:
    """
    Warn about experiments whose results no longer match their metadata.

    Rendered above the reprocessing form — and above the active-job guard, so
    it stays visible while a job runs — because this is the control that
    clears it.

    Parameters
    ----------
    stale_experiments : list of str
        Experiments flagged as needing reprocessing.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """

    if not stale_experiments:
        return

    st.warning(
        f"⚠️ {len(stale_experiments)} experiment(s) need reprocessing after a "
        "metadata change: " + ", ".join(stale_experiments)
    )


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

    stale_experiments = select_experiments_needing_reprocessing(
        dataset, config.metadata_excel_experiment_column)

    _render_stale_experiment_warning(stale_experiments)

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
        only_stale_experiments = st.checkbox(
            f"Only experiments needing reprocessing ({len(stale_experiments)})",
            value=False,
            key=REPROCESS_STALE_ONLY_KEY,
            disabled=not stale_experiments,
            help="Reprocess exactly the experiments whose metadata changed since "
                 "they were last processed, overriding the selection above.",
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

    # The shortcut stays checked once its list empties, and the widget is only
    # disabled, not reset — so an empty list has to fall back to the
    # multiselect rather than through it to "all experiments".
    requested_experiments = (stale_experiments
                             if only_stale_experiments and stale_experiments
                             else selected_experiments)

    start_chunked_job(
        job_key = REPROCESS_JOB_KEY,
        experiment_names = select_experiments_to_reprocess(dataset, requested_experiments or None),
        context = {
            'database': dataset,
            'processing_function': handler.processing_function,
            'metadata_retrival_function': handler.metadata_retrival_function if refresh_metadata else None,
            'external_version': resolve_external_version(dataset, config.external_version),
            'overview_df_experiment_column': _handler_experiment_column(handler, config),
        },
    )

    st.rerun()


def _render_HDF5_merging(config: DataUploadConfig, dataset: ExperimentalDataset) -> None:
    """
    Render the HDF5 merging uploader and merge submitted files into the dataset.

    Parameters
    ----------
    config : DataUploadConfig
        Page configuration. Unused here, kept for the uniform section signature.
    dataset : ExperimentalDataset
        Dataset the uploaded files are merged into.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """

    # Cleared on submit: the files have been absorbed into the dataset, and
    # pressing the button again with them still listed would merge them a
    # second time, only to report every experiment as a skipped duplicate.
    with st.form(key = "uploading_HDF5_files_to_merge", clear_on_submit = True):
        uploaded_files = st.file_uploader(
            label = "📦 Upload HDF5 Files to Merge",
            type = ['h5', 'hdf5'],
            help = "Upload one or more HDF5 files containing experiments to merge into the current dataset.",
            key = "merge_hdf5_uploader",
            accept_multiple_files = True,
        )
        submitted = st.form_submit_button("🚀 Merge HDF5 Files", width="stretch")

    if submitted and uploaded_files:
        _merge_uploaded_hdf5_files(dataset, uploaded_files)

    _render_merge_report(dataset)


def _merge_uploaded_hdf5_files(dataset: ExperimentalDataset, uploaded_files: list) -> None:
    """
    Merge the uploaded HDF5 files into the loaded dataset, and restart the page.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset already loaded. It goes into the merge as the first file, which
        is what gives it precedence: an experiment it already holds is kept,
        and the uploaded file's copy is reported as skipped.
    uploaded_files : list
        Files submitted through the merge uploader.

    Returns
    -------
    None : None
        Replaces ``st.session_state.experimental_dataset`` and reruns the page.

    Notes
    -----
    Merging produces a *new* dataset object, so every section rendered after
    this one would still be describing the dataset this run started with — the
    download button included, which is how a merge could be followed by
    downloading the unmerged file. The rerun is what makes the whole page
    describe the merged dataset instead.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Staged for the merge, not delivered: refusing this file because an
        # experiment awaits reprocessing would block merging altogether, and
        # it is read back and discarded within this run.
        current_dataset_path = str(Path(tmp_dir) / "_current_dataset.h5")
        dataset.save_to_hdf5(current_dataset_path, verbose=False,
                             allow_stale_processed_data=True)

        merge_paths = [current_dataset_path]

        # Staged under numbered names: two files can arrive with the same one,
        # and writing both to that name merged one of them twice and dropped
        # the other, reporting the difference only as skipped duplicates.
        for index, uploaded_file in enumerate(uploaded_files):
            file_path = Path(tmp_dir) / f"{index}_{uploaded_file.name}"
            file_path.write_bytes(uploaded_file.getbuffer())
            merge_paths.append(str(file_path))

        # Labelled by what the user recognises: the merge runs on temporary
        # paths, and those are what the provenance would otherwise record.
        merged_dataset = ExperimentalDataset.merge_hdf5_files(
            merge_paths,
            source_labels = [CURRENT_DATASET_MERGE_LABEL]
                            + [uploaded_file.name for uploaded_file in uploaded_files])

    st.session_state.experimental_dataset = merged_dataset

    # The merged sheet is a different table with different rows in different
    # places, and the grid addresses its rows by position — a delta it still
    # holds client-side would land on another experiment's row.
    discard_metadata_editor_state()

    st.rerun()


def _render_merge_report(dataset: ExperimentalDataset) -> None:
    """
    Report what the merge that produced this dataset decided.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset carrying a ``merge_report``; nothing is rendered for a dataset
        that is not a merge.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.

    Notes
    -----
    Read from the dataset rather than parked in session state, so that it
    survives the rerun the merge ends with — and keeps describing the dataset
    in memory for as long as it is the merged one. The skipped duplicates in
    particular were only ever printed to a console the user of a deployed app
    never sees, so a merge silently dropped the experiments it could not take.
    """

    report = dataset.merge_report

    if not report:
        return

    st.success(f"✅ Merged: {', '.join(report['sources'])} → "
               f"{len(dataset.experiments)} experiment(s), "
               f"{len(dataset.overview_df)} overview row(s)")

    skipped_experiments = report['skipped_experiments']

    if skipped_experiments:
        st.warning(
            f"⚠️ {len(skipped_experiments)} experiment(s) were skipped, because the dataset "
            "already held an experiment of the same name and nothing already loaded is "
            "overwritten: "
            + describe_skipped_experiments(skipped_experiments))

    metadata_corrected = report['metadata_corrected']

    if metadata_corrected:
        st.warning(
            f"⚠️ {len(metadata_corrected)} experiment(s) held metadata that disagreed with "
            "the merged overview table. The overview values have been adopted, and are what "
            "the analysis pages now use.")

        with st.expander("What was corrected"):
            st.code(describe_metadata_divergences(metadata_corrected,
                                                  maximum_reported=len(metadata_corrected)))

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

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.

    Notes
    -----
    The stale-results check has to happen here rather than being left to
    `ExperimentalDataset.save_to_hdf5`: the file is serialized while the page
    renders, because `st.download_button` needs the bytes up front, so the
    refusal would reach the user as a traceback covering the rest of the page.
    """
    st.subheader("💾 Download Dataset")

    if not dataset.experiments:
        st.info("No experiments in dataset. Upload data first to enable downloads.")
        return

    stale_experiments = dataset.stale_processed_experiments()

    if stale_experiments:
        _render_stale_results_refusal(stale_experiments)
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


def _render_stale_results_refusal(stale_experiments: list) -> None:
    """
    Say why the dataset cannot be downloaded, and what clears the way.

    Parameters
    ----------
    stale_experiments : list of str
        Experiments whose stored results no longer follow from their metadata.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.

    Notes
    -----
    The download is withheld rather than offered with a warning next to it,
    because the file would look exactly like a correct one: nothing downstream
    can tell results computed from metadata the file no longer contains from
    results that follow from it.
    """

    st.warning(
        f"⚠️ Download withheld: {len(stale_experiments)} experiment(s) hold results that no "
        "longer follow from their metadata, because the metadata changed after they were "
        f"processed — {describe_experiment_names(stale_experiments)}. The file would store "
        "results derived from values it does not contain, and nothing reading it later could "
        "tell. Reprocess them in section 4 above; its **Only experiments needing reprocessing** "
        "shortcut selects exactly these."
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
