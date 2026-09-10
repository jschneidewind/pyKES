"""
Tests for the metadata editor as it is wired into the Data Upload page.

`AppTest` runs a Streamlit script in-process, so the grid, its form and the
rerun that follows an applied edit can be driven without a browser. The edits
themselves are injected the way the frontend sends them — as the
``edited_rows`` delta `st.data_editor` keeps in session state — which is also
what makes the revision counter in the widget key worth pinning: a stale delta
would otherwise be replayed over the applied values.
"""

from streamlit.testing.v1 import AppTest

from pyKES.streamlit_app.components.metadata_editor import (METADATA_EDITOR_REVISION_KEY,
                                                            METADATA_LOADING_KEY,
                                                            METADATA_PROCESSING_KEY)


# Streamlit script rendering the real Data Upload page over a dataset that
# declares both column lists. `{declarations}` is filled in per test.
DATA_UPLOAD_PAGE_APP = '''
import pandas as pd
import streamlit as st

from pyKES.database.database_experiments import Experiment, ExperimentalDataset
from pyKES.streamlit_app.components.data_upload_component import render_data_upload
from pyKES.streamlit_app.config_interface import DataUploadConfig, FileUploadHandler

EXPERIMENT_NAMES = ["Exp_001", "Exp_002"]


def retrieve_metadata(experiment_name, overview_df):
    return {{"experiment_name": experiment_name}}


def read_raw_data(directory, metadata_dict):
    return {{"signal": [1.0, 2.0]}}


def process_raw_data(raw_data_dict, metadata_dict):
    return {{"maximum": max(raw_data_dict["signal"])}}


CONFIG = DataUploadConfig(file_handlers=[FileUploadHandler(
    label="Raw Data",
    file_type="csv",
    overview_df_experiment_column="Experiment",
    metadata_retrival_function=retrieve_metadata,
    raw_data_reading_function=read_raw_data,
    processing_function=process_raw_data,
)])

st.session_state.setdefault("hdf5_filename", None)

if "experimental_dataset" not in st.session_state:
    dataset = ExperimentalDataset(
        overview_df=pd.DataFrame({{"Experiment": EXPERIMENT_NAMES,
                                  "File name O2": ["one.csv", "two.csv"],
                                  "Irradiance [mW/cm2]": [40.0, 80.0],
                                  "Comment": ["first", "second"],
                                  "Processed": ["True", "True"]}}),
        processing_parameters={declarations})

    for experiment_name in EXPERIMENT_NAMES:
        dataset.add_experiment(Experiment(
            experiment_name=experiment_name,
            raw_data_file=experiment_name,
            color="black",
            group="default",
            metadata={{"experiment_name": experiment_name, "Irradiance [mW/cm2]": 40.0}},
            raw_data={{"signal": [1.0, 2.0]}},
            processed_data={{"maximum": 2.0}}))

    st.session_state.experimental_dataset = dataset

render_data_upload(CONFIG)
'''

DECLARATIONS = ('{"' + METADATA_LOADING_KEY + '": ["Experiment", "File name O2"], '
                '"' + METADATA_PROCESSING_KEY + '": ["Irradiance [mW/cm2]"]}')


def run_page(declarations=DECLARATIONS):
    """Run the Data Upload page over a dataset with the given declarations."""
    app = AppTest.from_string(DATA_UPLOAD_PAGE_APP.format(declarations=declarations))
    app.run(timeout=60)

    assert [element.value for element in app.exception] == []

    return app


def edit_cell(app, row_index, column, value, revision=0):
    """Inject an edit the way the data editor's frontend sends one."""
    app.session_state[f"metadata_editor_{revision}"] = {"edited_rows": {row_index: {column: value}},
                                                        "added_rows": [],
                                                        "deleted_rows": []}

    apply_button = next(button for button in app.button if "Apply metadata" in button.label)
    apply_button.click()
    app.run(timeout=60)

    assert [element.value for element in app.exception] == []

    return app


def test_the_editor_is_offered_as_its_own_section():
    app = run_page()

    subheaders = [element.value for element in app.subheader]

    assert any("Edit Metadata" in subheader for subheader in subheaders)
    # Renumbered around the new section, and still behind the active-job guard
    assert any("5. 📦 Merge HDF5 Files" == subheader for subheader in subheaders)
    assert any("6. 💾 Download Dataset" == subheader for subheader in subheaders)
    assert METADATA_EDITOR_REVISION_KEY in app.session_state


def test_a_dataset_without_declarations_gets_an_explanation_instead():
    app = run_page(declarations='{}')

    messages = [element.value for element in app.info]

    assert any(METADATA_LOADING_KEY in message for message in messages)
    assert METADATA_EDITOR_REVISION_KEY not in app.session_state


def test_editing_a_processing_column_flags_the_experiment():
    app = edit_cell(run_page(), 0, "Irradiance [mW/cm2]", 55.0)

    dataset = app.session_state["experimental_dataset"]

    assert dataset.overview_df["Irradiance [mW/cm2]"].tolist() == [55.0, 80.0]
    assert dataset.overview_df["Processed"].tolist() == ["False", "True"]
    assert dataset.experiments["Exp_001"].metadata["Irradiance [mW/cm2]"] == 55.0

    # Both the editor's own report and the standing warning above section 3
    warnings = [element.value for element in app.warning]
    assert sum("Exp_001" in warning for warning in warnings) == 2

    # A fresh widget key, so the applied delta cannot be replayed
    assert app.session_state[METADATA_EDITOR_REVISION_KEY] == 1


def test_editing_a_free_column_leaves_the_flag_alone():
    app = edit_cell(run_page(), 1, "Comment", "corrected")

    dataset = app.session_state["experimental_dataset"]

    assert dataset.overview_df["Comment"].tolist() == ["first", "corrected"]
    assert dataset.overview_df["Processed"].tolist() == ["True", "True"]
    assert [element.value for element in app.warning] == []


def test_the_reprocessing_shortcut_offers_exactly_the_flagged_experiments():
    app = edit_cell(run_page(), 0, "Irradiance [mW/cm2]", 55.0)

    shortcut = next(checkbox for checkbox in app.checkbox
                    if "Only experiments needing reprocessing" in checkbox.label)

    assert "(1)" in shortcut.label
    assert not shortcut.disabled


def test_the_reprocessing_shortcut_is_disabled_while_nothing_needs_it():
    app = run_page()

    shortcut = next(checkbox for checkbox in app.checkbox
                    if "Only experiments needing reprocessing" in checkbox.label)

    assert "(0)" in shortcut.label
    assert shortcut.disabled
