"""
Configuration of the example app: the one object the pyKES pages are given.

This is the whole extension surface. The pages under `pages/` are one-line
delegations to the reusable pyKES components; everything domain-specific is a
field of a dataclass here, and nothing in `pyKES/streamlit_app/` is forked or
edited.

Two handlers are declared because an app usually owns more than one
instrument setup, and each needs its own file readers and processing
function. They are alternatives, not layers: an experiment is ingested by
whichever pipeline matches it, and both then appear in the reprocessing
pipeline selector on the Data Upload page. The example data in
``examples/example_data`` is a liquid-phase series, so the liquid-phase
handler is the one to upload it to.
"""

from pyKES.streamlit_app.config_interface import (
    DataUploadConfig,
    FileUploadHandler,
    HomeConfig,
    PyKESStreamlitConfig,
)
from pyKES.utilities.version_information import get_project_version

from metadata_functions import metadata_retrival_function
from parameters import GROUP_MAPPING, PLOTTING_INSTRUCTIONS, PROCESSING_PARAMETERS
from processing_functions import process_gas_phase, process_liquid_phase
from raw_data_functions import read_gas_phase_raw_data, read_liquid_phase_raw_data


# -----------------------------------------------------------------------------
# Provenance of this app, stored in every dataset it processes
# -----------------------------------------------------------------------------

# Recorded in dataset.version['external_version'], so processed data can be
# traced back to the code that produced it. get_project_version reads the
# version declared in the nearest pyproject.toml above this file — for a real
# external repository that is the app's own, and bumping it when the
# processing behaviour changes is what makes the stamp mean something.
EXTERNAL_VERSION = {
    "app": "external_repo_example",
    "version": get_project_version(__file__),
}

# -----------------------------------------------------------------------------
# Upload handlers, one per instrument setup
# -----------------------------------------------------------------------------

liquid_phase_handler = FileUploadHandler(
    label="💧 Liquid phase — upload raw sensor files",
    file_type=["csv", "txt"],
    help_text="UniAmp H2 (.csv) and FireStingO2 (.txt) exports of a liquid-phase "
              "run. Upload every file the metadata sheet names, in one go.",
    overview_df_experiment_column="Experiment",
    metadata_retrival_function=metadata_retrival_function,
    raw_data_reading_function=read_liquid_phase_raw_data,
    processing_function=process_liquid_phase,
)

gas_phase_handler = FileUploadHandler(
    label="💨 Gas phase — upload raw sensor files",
    file_type=["csv", "txt"],
    help_text="Same instruments in gas-phase mode, where the sensors report a "
              "partial pressure rather than a dissolved concentration.",
    overview_df_experiment_column="Experiment",
    metadata_retrival_function=metadata_retrival_function,
    raw_data_reading_function=read_gas_phase_raw_data,
    processing_function=process_gas_phase,
)

# -----------------------------------------------------------------------------
# Data Upload and Home configuration
# -----------------------------------------------------------------------------

DATA_UPLOAD_CONFIG = DataUploadConfig(

    file_handlers=[liquid_phase_handler, gas_phase_handler],

    page_title="Data Upload & Processing",
    page_description=(
        "Upload the metadata sheet, then the raw sensor files it references. "
        "Already-processed experiments are skipped, metadata can be corrected "
        "in place, and reprocessing rebuilds the results without the raw files."
    ),
    output_hdf5_name="example_dataset.h5",

    metadata_excel_sheet_name="Sheet1",
    metadata_excel_experiment_column="Experiment",

    group_mapping=GROUP_MAPPING,
    plotting_instruction=PLOTTING_INSTRUCTIONS,
    processing_parameters=PROCESSING_PARAMETERS,
    external_version=EXTERNAL_VERSION,
)

HOME_CONFIG = HomeConfig(
    page_title="pyKES Example App",
    main_title="pyKES Example App",
    upload_label="Load an existing HDF5 dataset",
    upload_help_text="Try examples/example_data/example_dataset.h5 for a finished dataset.",
    loaded_dataset_title="Loaded dataset",
    intro_markdown=(
        "A fully configured pyKES app over a small photocatalytic "
        "water-splitting series.\n\n"
        "* **Data Upload** — build a dataset from a metadata sheet and raw sensor "
        "files, correct metadata in place, and reprocess\n"
        "* **Analysis Results** — scalar results against experiments or against a "
        "metadata parameter\n"
        "* **Time Series** — raw, smoothed and rate curves per experiment\n"
        "* **Results Table** — every result side by side, sortable and exportable\n\n"
        "See `examples/README.md` for a walkthrough that exercises all of it."
    ),
)

# -----------------------------------------------------------------------------
# Top-level app configuration
# -----------------------------------------------------------------------------

PYKES_CONFIG = PyKESStreamlitConfig(
    home_config=HOME_CONFIG,
    data_upload_config=DATA_UPLOAD_CONFIG,
    app_title="pyKES Example App",
    app_icon=":test_tube:",
)
