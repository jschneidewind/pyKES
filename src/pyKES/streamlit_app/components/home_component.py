"""
Home page component for pyKES Streamlit applications.

This module provides a reusable and configurable Streamlit home page that can
be imported by external repositories. The page supports loading an HDF5 dataset,
showing the loaded dataset overview, and customizing the visible title and labels
through HomeConfig.
"""

import os
import tempfile
import streamlit as st
from typing import Optional

from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.streamlit_app.config_interface import HomeConfig

def render_home(config: Optional[HomeConfig] = None) -> None:
    """
    Render the configurable Home page.

    Parameters
    ----------
    config : HomeConfig, optional
        Configuration controlling branding, labels, and intro text.
        If None, defaults are used.
    """
    if config is None:
        config = HomeConfig()

    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(config.main_title)
    st.sidebar.success(config.sidebar_success_message)

    # Initialize session state for shared data
    if 'experimental_dataset' not in st.session_state:
        st.session_state.experimental_dataset = None
    if 'hdf5_filename' not in st.session_state:
        st.session_state.hdf5_filename = None

    uploaded_file = st.file_uploader(
        config.upload_label,
        type=['h5', 'hdf5'],
        help=config.upload_help_text,
    )

    if uploaded_file is not None:
        st.session_state.hdf5_filename = uploaded_file.name

        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                # Write the uploaded file's content to the temporary file
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Load the dataset from the temporary file and store it in session state
                st.session_state.experimental_dataset = ExperimentalDataset.load_from_hdf5(tmp_file_path)
                st.success(
                    "File uploaded successfully! You can now navigate to other pages to analyze this data."
                )
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_file_path)
                except OSError:
                    pass

        except Exception as e:
            st.error(f"Error loading HDF5 file: {str(e)}")
            st.info(
                "Please ensure the HDF5 file has the correct structure with experimental data and metadata."
            )
    else:
        st.info(config.empty_state_message)

    if st.session_state.experimental_dataset is not None:
        st.title(config.loaded_dataset_title)
        if st.session_state.hdf5_filename:
            st.caption(f"File: {st.session_state.hdf5_filename}")
        st.dataframe(st.session_state.experimental_dataset.overview_df)

    st.markdown(config.intro_markdown)

