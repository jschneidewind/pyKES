"""
Configuration Interface for pyKES Streamlit Application

This module defines the configuration schema and dataclasses that external repositories
should use to customize the Data Upload page in pyKES Streamlit applications.

External repos provide configurations for:
- Data processing functions (metadata retrieval, raw data reading, processing)
- File upload handlers (what file types to accept and how to handle them)
- Page branding and customization

Author: pyKES Development Team
Date: 2026
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any


@dataclass
class FileUploadHandler:
    """
    Configuration for one file uploader on the Data Upload page.

    Each handler defines a Streamlit ``file_uploader`` and, optionally, the
    processing pipeline that turns its uploads into ``Experiment`` objects.

    Parameters
    ----------
    label : str
        Label shown above the uploader widget.
    file_type : str or List[str]
        Accepted extension(s), without leading dots (e.g. ``"csv"`` or
        ``["csv", "txt"]``).
    help_text : str, optional
        Tooltip displayed under the uploader.
    file_storage_key : str, optional
        Stable key used to identify this handler's uploads. Auto-generated
        from ``label`` when not provided.
    metadata_retrival_function : callable, optional
        ``(metadata_arg, overview_df) -> metadata_dict``. ``metadata_arg`` is
        the experiment name in overview_df-based mode and a file path
        otherwise.
    raw_data_reading_function : callable, optional
        ``(reader_arg, metadata_dict) -> raw_data_dict``. ``reader_arg`` is a
        ``{column: absolute_path}`` dict when ``file_name_field`` is set,
        otherwise a single path string.
    processing_function : callable, optional
        ``(raw_data_dict, metadata_dict) -> processed_data_dict``.

    Notes
    -----
    A handler is "processing-enabled" only when all three callables are
    provided. Handlers without the full pipeline still upload files but do
    not produce experiments.
    """

    label: str
    file_type: str | List[str]
    help_text: Optional[str] = None
    file_storage_key: Optional[str] = None
    overview_df_experiment_column: Optional[str] = None
    metadata_retrival_function: Optional[Callable[[Any, Any], Dict[str, Any]]] = None
    raw_data_reading_function: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None
    processing_function: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        # Derive a stable session-state key from the label, e.g.
        # "Upload Data (CSV)" -> "upload_data_csv".
        if self.file_storage_key is None:
            self.file_storage_key = (
                self.label.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
            )

@dataclass
class DataUploadConfig:
    """
    Configuration for the Data Upload page.

    Parameters
    ----------
    file_handlers : List[FileUploadHandler]
        At least one handler. Each becomes a Streamlit uploader.

    page_title : str, optional
        Title rendered at the top of the page.

    page_description : str, optional
        Markdown blurb shown under the title.

    output_hdf5_name : str, optional
        Filename used for both the download button and auto-save.

    metadata_excel_sheet_name : str, optional
        Sheet name used by the metadata Excel uploader.    
    
    metadata_excel_experiment_column : str, optional
        Column name used when merging the uploaded sheet into
        ``dataset.overview_df`` (existing rows updated, new rows appended).

    fresh_dataset_overview_columns : List[str], optional
        Columns used to seed an empty ``overview_df`` for the
        "Start Fresh Dataset" action.

    fresh_dataset_group_mapping : Dict[str, Any], optional
    fresh_dataset_plotting_instruction : Dict[str, Any], optional
    fresh_dataset_processing_parameters : Dict[str, Any], optional
        Defaults stored on a freshly created ``ExperimentalDataset``.
    """

    file_handlers: List[FileUploadHandler]

    page_title: str = "Data Upload"
    page_description: str = (
        "Upload metadata and raw data files to build or extend your dataset. "
        "The uploaded data will be processed and added to your HDF5 database."
    )
    output_hdf5_name: str = "experiment_data.h5"

    # Metadata Excel uploader (always rendered, single source of overview_df).
    metadata_excel_sheet_name: str = "Sheet1"
    metadata_excel_experiment_column: str = "Experiment"

    # Fresh-dataset defaults used by the "Start Fresh Dataset" action.
    group_mapping: Dict[str, Any] = field(default_factory=dict)
    plotting_instruction: Dict[str, Any] = field(default_factory=dict)
    processing_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HomeConfig:
    """
    Configuration for the Home page in pyKES Streamlit applications.

    This dataclass controls the page branding and the default file-loading
    experience shown on the application home page.

    Parameters
    ----------
    page_title : str, optional
        Browser tab title shown by Streamlit.
        Default: "Kinetic Data Visualization"

    page_icon : str, optional
        Browser tab icon shown by Streamlit.
        Default: ":microscope:"

    main_title : str, optional
        Large title shown in the page body.
        Default: "Kinetic Data Visualization"

    upload_label : str, optional
        Label shown above the HDF5 file uploader.
        Default: "Upload HDF5 File"

    upload_help_text : str, optional
        Help text shown below the HDF5 file uploader.
        Default: "Upload a saved pyKES HDF5 dataset to continue working with it."

    sidebar_success_message : str, optional
        Sidebar message shown on the home page.
        Default: "Select a page above."

    empty_state_message : str, optional
        Message shown when no dataset has been loaded yet.
        Default: "Please upload an HDF5 file to visualize experimental data."

    loaded_dataset_title : str, optional
        Section title shown when a dataset is loaded.
        Default: "Loaded Dataset"

    intro_markdown : str, optional
        Markdown shown at the bottom of the home page.
        Default: application overview text.

    Examples
    --------
    >>> config = HomeConfig(
    ...     page_title="Photocatalysis Dashboard",
    ...     main_title="Photocatalysis Data Portal",
    ...     upload_label="Load Existing HDF5 Dataset",
    ... )
    """

    page_title: str = "Kinetic Data Visualization"
    page_icon: str = ":microscope:"
    main_title: str = "Kinetic Data Visualization"
    upload_label: str = "Upload HDF5 File"
    upload_help_text: str = "Upload a saved pyKES HDF5 dataset to continue working with it."
    sidebar_success_message: str = "Select a page above."
    empty_state_message: str = "Please upload an HDF5 file to visualize experimental data."
    loaded_dataset_title: str = "Loaded Dataset"
    intro_markdown: str = (
        "This is the home page of our data analysis application.\n"
        "Choose a page from the sidebar to begin your analysis:\n"
        "* **Data Upload and Download** - Adding new data and downloading dataset\n"
        "* **📈 Data Analysis** - Basic statistical analysis\n"
        "* **📊 Visualization** - Data visualization tools"
    )



@dataclass
class PyKESStreamlitConfig:
    """
    Complete configuration for a pyKES-based Streamlit application.
    
    This is the top-level config that external repos can use to configure
    their entire Streamlit application with pyKES components.
    
    Parameters
    ----------
    home_config : HomeConfig
        Configuration for the Home page.

    data_upload_config : DataUploadConfig
        Configuration for the Data Upload page.
        
    app_title : str, optional
        Title of the Streamlit app.
        Default: "pyKES Data Analysis"
    
    app_icon : str, optional
        Emoji icon for the app (shown in browser tab).
        Default: ":microscope:"
    
    Examples
    --------
    >>> full_config = PyKESStreamlitConfig(
    ...     data_upload_config=data_upload_cfg,
    ...     app_title="Photocatalysis Lab Data System",
    ...     app_icon=":microscope:"
    ... )
    """
    data_upload_config: DataUploadConfig
    home_config: HomeConfig = field(default_factory=HomeConfig)
    app_title: str = "pyKES Data Analysis"
    app_icon: str = ":microscope:"
