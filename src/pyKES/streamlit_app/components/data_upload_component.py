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

A per-handler progress bar reports ingestion progress in real time.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.database.data_processing import read_in_experiments_single_threaded
from pyKES.streamlit_app.config_interface import DataUploadConfig, FileUploadHandler
from pyKES.database.database_experiments import import_overview_excel


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
        _render_raw_data_uploaders(handler, dataset)
    st.divider()

    st.subheader("3. 📦 Merge HDF5 Files")
    _render_HDF5_merging(config, dataset)
    st.divider()

    st.subheader("4. 💾 Download Dataset")
    _render_download_section(config, dataset)
    st.divider()

    st.subheader("📊 Dataset Overview")
    st.dataframe(st.session_state.experimental_dataset.overview_df)
    st.divider()

    st.subheader("📊 Dataset Statistics")
    _render_dataset_statistics(dataset)
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
        st.session_state.experimental_dataset = ExperimentalDataset(
            overview_df=pd.DataFrame(),
            group_mapping=config.group_mapping,
            plotting_instruction=config.plotting_instruction,
            processing_parameters=config.processing_parameters,
        )
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

def _render_raw_data_uploaders(config: FileUploadHandler, dataset: ExperimentalDataset) -> None:
    with st.form(key=f"upload_form_{config.file_storage_key}", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            label = config.label,
            type = config.file_type,
            help = config.help_text,
            key = config.file_storage_key,
            accept_multiple_files = True,
        )
        submitted = st.form_submit_button("🚀 Process data", use_container_width=True)

    if not submitted or not uploaded_files:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            file_path = Path(tmp_dir) / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())

        results = read_in_experiments_single_threaded(
            database = dataset,
            metadata_retrival_function = config.metadata_retrival_function,
            raw_data_reading_function = config.raw_data_reading_function,
            processing_function = config.processing_function,
            directory = Path(tmp_dir),
            overview_df_experiment_column = config.overview_df_experiment_column,
        )

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    if successes:
        st.success(f"✓ Processed {len(successes)} experiment(s) successfully")
    
    if failures:
        st.error(f"✗ Failed to process {len(failures)} experiment(s):")
        for failure in failures:
            st.error(f"**{failure['file']}**: {failure['error']}")

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
        submitted = st.form_submit_button("🚀 Merge HDF5 Files", use_container_width=True)
    
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
