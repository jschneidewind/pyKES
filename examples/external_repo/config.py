"""pyKES example configuration for an external Streamlit app."""

from pyKES.streamlit_app.config_interface import (
    DataUploadConfig,
    FileUploadHandler,
    HomeConfig,
    PyKESStreamlitConfig,
)

from tests.data.processing_parameters import PROCESSING_PARAMETERS, PLOTTING_INSTRUCTIONS, GROUP_MAPPING
from tests.data.processing_functions_overview_df import metadata_retrival_function, raw_data_reading_function, processing_function


# -----------------------------------------------------------------------------
# File Handler Config
# -----------------------------------------------------------------------------

file_handler_A =  FileUploadHandler(
                        label = "📊 Upload Raw Data - A",
                        file_type = ["csv", "txt"],
                        help_text = "Raw data files referenced from the metadata sheet.",
                        overview_df_experiment_column = "Experiment",
                        metadata_retrival_function = metadata_retrival_function,
                        raw_data_reading_function = raw_data_reading_function,
                        processing_function = processing_function,
                        )

file_handler_B =  FileUploadHandler(
                        label = "📊 Upload Raw Data - B",
                        file_type = ["csv", "txt"],
                        help_text = "Raw data files referenced from the metadata sheet.",
                        overview_df_experiment_column = "Experiment",
                        metadata_retrival_function = metadata_retrival_function,
                        raw_data_reading_function = raw_data_reading_function,
                        processing_function = processing_function,
                        )

# -----------------------------------------------------------------------------
# Data Upload and Home Configuration
# -----------------------------------------------------------------------------

DATA_UPLOAD_CONFIG = DataUploadConfig(

    file_handlers = [file_handler_A, file_handler_B],

    page_title="Data Upload & Processing",
    page_description=(
        "Upload a metadata Excel sheet, then upload the raw CSV files it "
        "references. Already-processed experiments are skipped automatically."
        ),

    metadata_excel_experiment_column="Experiment",
    group_mapping=GROUP_MAPPING,
    plotting_instruction=PLOTTING_INSTRUCTIONS,
    processing_parameters=PROCESSING_PARAMETERS,
)

HOME_CONFIG = HomeConfig()

# -----------------------------------------------------------------------------
# Top-level app configuration
# -----------------------------------------------------------------------------

PYKES_CONFIG = PyKESStreamlitConfig(
    home_config = HOME_CONFIG,
    data_upload_config = DATA_UPLOAD_CONFIG,
    app_title = "Photocatalysis Data Analysis System",
    app_icon = ":test_tube:",
)
