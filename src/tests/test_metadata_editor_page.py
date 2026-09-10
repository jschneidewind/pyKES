"""
Tests for the metadata editor as it is wired into the Data Upload page.

`AppTest` runs a Streamlit script in-process, so the grid and the save that
follows a committed cell can be driven without a browser. The edits themselves
are injected the way the frontend sends them — as the ``edited_rows`` delta
`st.data_editor` keeps in session state — and applied with the form's own
submit button, which is what a form does: nothing reaches Python until it is
pressed, so a drag-fill or a pasted block is left undisturbed while it is
being made.

Applying ends in an app-scoped rerun, so section 3 — drawn by the page body,
above the editor — shows the reprocessing warning and enables its shortcut on
the same press. `test_the_reprocessing_shortcut_offers_exactly_the_flagged_experiments`
is what pins that: it asserts on the press itself, with no extra run.

`test_an_edit_survives_the_workbook_staying_in_the_uploader` is the regression
that matters most here. The uploader keeps its file for the whole session, and
re-merging the sheet on every rerun silently reverted every edit one rerun
after it was made — while leaving the experiment's own metadata holding it, so
`overview_df` and the stored metadata disagreed. The suite missed it because
no test had ever put a file in the uploader.
"""

import io

import pandas as pd
from streamlit.testing.v1 import AppTest

from pyKES.streamlit_app.components.metadata_editor import (METADATA_EDITOR_REVISION_KEY,
                                                            METADATA_LOADING_KEY,
                                                            METADATA_PROCESSING_KEY)

XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


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


def edit_cells(app, edited_rows):
    """
    Inject edits the way the data editor's frontend sends them, and apply them.

    ``edited_rows`` is the delta itself — ``{row_index: {column: value}}`` —
    so one call can stand in for a drag-fill down several rows, which is what
    the grid sends as a single delta on submit.
    """
    revision = app.session_state[METADATA_EDITOR_REVISION_KEY]

    app.session_state[f"metadata_editor_{revision}"] = {"edited_rows": edited_rows,
                                                        "added_rows": [],
                                                        "deleted_rows": []}

    apply_button = next(button for button in app.button if "Apply metadata" in button.label)
    apply_button.click()
    app.run(timeout=60)

    assert [element.value for element in app.exception] == []

    return app


def edit_cell(app, row_index, column, value):
    """Inject and apply one edited cell."""
    return edit_cells(app, {row_index: {column: value}})


def build_workbook(overview_df):
    """Serialize an overview table as the bytes of an uploaded workbook."""
    buffer = io.BytesIO()
    overview_df.to_excel(buffer, sheet_name='Sheet1', index=False)

    return buffer.getvalue()


SHEET = pd.DataFrame({"Experiment": ["Exp_001", "Exp_002"],
                      "File name O2": ["one.csv", "two.csv"],
                      "Irradiance [mW/cm2]": [40, 80],
                      "Comment": ["first", "second"]})


def upload_workbook(app, overview_df=SHEET):
    """Put a metadata workbook in the uploader, as a user would."""
    app.file_uploader[0].set_value(('metadata.xlsx', build_workbook(overview_df), XLSX_MIME))
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

    # The editor reports it straight away, from inside its own fragment — as a
    # caption, so the fragment keeps the same height and nothing below it moves.
    # Two captions name it: what was stored, and what now needs reprocessing.
    captions = [element.value for element in app.caption]
    assert sum("Exp_001" in caption for caption in captions) == 2
    assert any("Saved 1 change(s)" in caption for caption in captions)
    assert any("need reprocessing" in caption for caption in captions)

    # Section 3 is drawn by the page body, above the editor, so applying ends
    # in an app-scoped rerun to bring it up to date on the same press rather
    # than leaving it stale until the next page interaction.
    assert sum("Exp_001" in element.value for element in app.warning) == 1

    # The widget key is untouched: changing it would reset the grid's scroll
    # position, and there is no stale delta to escape from.
    assert app.session_state[METADATA_EDITOR_REVISION_KEY] == 0


def test_editing_a_free_column_leaves_the_flag_alone():
    app = edit_cell(run_page(), 1, "Comment", "corrected")

    dataset = app.session_state["experimental_dataset"]

    assert dataset.overview_df["Comment"].tolist() == ["first", "corrected"]
    assert dataset.overview_df["Processed"].tolist() == ["True", "True"]
    assert [element.value for element in app.warning] == []


def test_the_reprocessing_shortcut_offers_exactly_the_flagged_experiments():
    app = edit_cell(run_page(), 0, "Irradiance [mW/cm2]", 55.0)

    # Selectable on the same press: the app-scoped rerun redraws section 3.
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


def test_an_edit_survives_the_workbook_staying_in_the_uploader():
    app = upload_workbook(run_page())

    edit_cell(app, 0, "Irradiance [mW/cm2]", 55.0)
    app.run(timeout=60)                      # an unrelated rerun must not revert it

    dataset = app.session_state["experimental_dataset"]
    overview = dataset.overview_df.set_index("Experiment")

    assert overview.loc["Exp_001", "Irradiance [mW/cm2]"] == 55.0
    assert dataset.experiments["Exp_001"].metadata["Irradiance [mW/cm2]"] == 55.0
    assert dataset.metadata_divergences() == {}


def test_the_workbook_is_merged_once_per_upload():
    app = upload_workbook(run_page())

    edit_cell(app, 1, "Comment", "corrected")
    app.run(timeout=60)

    overview = app.session_state["experimental_dataset"].overview_df.set_index("Experiment")

    assert overview.loc["Exp_002", "Comment"] == "corrected"


def test_uploading_a_sheet_overrides_edits_made_in_the_app():
    app = upload_workbook(run_page())
    edit_cell(app, 0, "Irradiance [mW/cm2]", 55.0)

    # Clearing and re-adding the workbook is a new upload, so it merges again.
    app.file_uploader[0].set_value(None)
    app.run(timeout=60)
    upload_workbook(app)

    dataset = app.session_state["experimental_dataset"]
    overview = dataset.overview_df.set_index("Experiment")

    assert overview.loc["Exp_001", "Irradiance [mW/cm2]"] == 40
    assert dataset.experiments["Exp_001"].metadata["Irradiance [mW/cm2]"] == 40
    # A fresh grid per merged workbook, so a client-side delta from before an
    # upload cannot reassert itself over the sheet. Two uploads, two bumps.
    assert app.session_state[METADATA_EDITOR_REVISION_KEY] == 2


def test_a_fraction_survives_a_whole_number_column():
    app = upload_workbook(run_page())

    edit_cell(app, 0, "Irradiance [mW/cm2]", 42.5)

    overview = app.session_state["experimental_dataset"].overview_df.set_index("Experiment")

    # int64 from Excel; the column is widened rather than rounding to 42
    assert overview.loc["Exp_001", "Irradiance [mW/cm2]"] == 42.5


def test_a_drag_fill_down_several_rows_is_applied_to_all_of_them():
    app = upload_workbook(run_page())

    # What the grid sends after dragging one value down: one delta, many rows.
    edit_cells(app, {0: {"Comment": "checked"}, 1: {"Comment": "checked"}})

    overview = app.session_state["experimental_dataset"].overview_df.set_index("Experiment")

    assert overview["Comment"].tolist() == ["checked", "checked"]


def test_pressing_the_button_with_nothing_changed_says_so():
    app = upload_workbook(run_page())

    apply_button = next(button for button in app.button if "Apply metadata" in button.label)
    apply_button.click()
    app.run(timeout=60)

    assert [element.value for element in app.exception] == []
    assert any("nothing was stored" in caption.value for caption in app.caption)
