"""
Spreadsheet-style metadata editor for the Data Upload page.

``overview_df`` is rendered as an `st.data_editor` grid, one row per
experiment, so a wrong irradiance or a missing volume can be corrected without
producing a new Excel sheet. The grid is what makes editing many experiments
at once cheap: Streamlit's editor already supports multi-cell selection,
drag-fill and pasting a block straight out of Excel, so no bulk-edit UI of our
own is needed.

Which columns may be written, and what each write costs, is decided by the
dataset rather than by this page — see `pyKES.database.metadata_editing`.
Datasets that declare nothing get an explanatory note instead of a grid, which
is what keeps files written before the declarations existed working unchanged.

The grid sits **in a form, inside a fragment**, and both halves are
load-bearing. Measured in headless Chromium against the deployed layout:

* **The form is what makes the spreadsheet gestures work.** Inside one, a
  committed cell sends nothing and triggers no rerun, so a drag-fill or a
  pasted block is left alone until the button is pressed: dragging a value
  down four rows arrived as one five-cell delta and all five were saved.
  Saving on every committed cell instead — no form — reruns the grid
  mid-gesture, which is what made drag-fill drop rows and single edits go
  missing.
* **The fragment is what keeps the page still.** A submit inside one reruns
  the fragment alone: the page body does not execute, so neither the uploaded
  workbook is re-read nor the whole HDF5 file rewritten for the download
  button, and the scroll position does not move (537 -> 537 px with the button
  below the grid, 70 -> 70 above it). The earlier version reran at app scope
  and moved the page.

Two more things not to reintroduce. `st.rerun` after a submit is unnecessary
here and is what an app-scoped rerun costs. And changing the grid's widget key
resets its horizontal scroll to the first columns — scroll survives a rerun of
either scope (1200 -> 1200 px) and is lost only to a new key — so the key is
bumped by an uploaded workbook and nothing else.
"""

import streamlit as st

from pyKES.database.data_processing import select_experiments_needing_reprocessing
from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.database.metadata_editing import (METADATA_LOADING_KEY, METADATA_PROCESSING_KEY,
                                             apply_metadata_edits, changed_metadata_cells,
                                             editable_metadata_columns,
                                             locked_metadata_columns, metadata_editing_available,
                                             metadata_editor_view, missing_declared_columns)
from pyKES.streamlit_app.config_interface import DataUploadConfig


# Carried in the grid's widget key, and bumped only by an uploaded workbook.
# `st.data_editor` keeps its `edited_rows` delta in session state and replays
# it on top of whatever data it is handed, so after a sheet upload — which
# takes precedence over anything edited here — a stale delta would reassert
# the values the upload just replaced. A new key re-seeds the grid clean, at
# the cost of resetting its scroll position, which is why nothing else bumps
# it.
METADATA_EDITOR_REVISION_KEY = 'metadata_editor_revision'

# Key of the form the grid and its button share. Fixed rather than following
# the revision: the form's identity does not need to change when the grid is
# re-seeded, and a stable key keeps the button's own state intact.
METADATA_EDITOR_FORM_KEY = 'metadata_editor_form'

# Outcome of an applied edit, rendered on the run *after* it. Applying ends in
# an app-scoped rerun, which discards everything the applying run had drawn.
METADATA_EDIT_SUMMARY_KEY = 'metadata_editor_last_summary'


def metadata_editor_widget_key() -> str:
    """
    Widget key of the metadata grid.

    Returns
    -------
    str
        Key carrying the current editor revision.
    """

    revision = st.session_state.setdefault(METADATA_EDITOR_REVISION_KEY, 0)

    return f"metadata_editor_{revision}"


def discard_metadata_editor_state() -> None:
    """
    Re-seed the metadata grid, dropping the edits it still holds client-side.

    Called when a newly uploaded workbook has replaced the overview sheet, so
    that the grid shows the sheet rather than replaying edits over it.

    Returns
    -------
    None : None
        Bumped on every merged workbook, including the first. Whether the grid
        had rendered yet cannot be told from here — a widget's key is not
        visible in session state until its widget is created in the current
        run, and the uploader runs before the grid — and re-keying a grid that
        holds nothing costs nothing.
    """

    st.session_state[METADATA_EDITOR_REVISION_KEY] = (
        st.session_state.get(METADATA_EDITOR_REVISION_KEY, 0) + 1)


def render_metadata_editor(config: DataUploadConfig, dataset: ExperimentalDataset) -> None:
    """
    Render the metadata editing section.

    Parameters
    ----------
    config : DataUploadConfig
        Supplies the column naming the experiments.
    dataset : ExperimentalDataset
        Dataset whose metadata is edited; mutated in place as cells are
        committed.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """

    if not metadata_editing_available(dataset):
        st.info(
            "Metadata editing is not available for this dataset. It becomes available "
            f"once the dataset declares both `{METADATA_LOADING_KEY}` and "
            f"`{METADATA_PROCESSING_KEY}` in its processing parameters, which files "
            "written before those declarations existed do not. Start a fresh dataset "
            "with an app that declares them, or edit the metadata Excel sheet and "
            "upload it above."
        )
        return

    if dataset.overview_df.empty:
        st.info("No metadata to edit yet — upload a metadata Excel sheet first.")
        return

    experiment_column = config.metadata_excel_experiment_column

    render_editing_policy(dataset, locked_metadata_columns(dataset, experiment_column))

    render_metadata_grid(dataset, experiment_column)


def render_editing_policy(dataset: ExperimentalDataset, locked_columns: list) -> None:
    """
    Explain which columns are locked and which invalidate the processed data.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset whose declarations are described.
    locked_columns : list of str
        Columns rendered read-only in the grid.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """

    st.markdown(
        "Correct metadata directly in the table below, then press **Apply metadata "
        "changes**. Cells can be selected, dragged down and pasted into as in a "
        "spreadsheet, so several experiments can be corrected in one go — nothing is "
        "stored until the button is pressed, which is what leaves those gestures "
        "undisturbed. Editing a column the processing function reads clears the "
        "experiment's `Processed` flag and lists it for reprocessing in section 4 below. "
        "Uploading a metadata sheet above replaces whatever is edited here."
    )

    if locked_columns:
        st.caption(
            "Read-only: " + ", ".join(f"`{column}`" for column in locked_columns)
            + ". These select *which* measurement an experiment is, so correcting one "
            "means uploading the corrected sheet and the raw-data files again."
        )

    missing_columns = missing_declared_columns(dataset)

    if missing_columns:
        st.warning(
            "Declared in the processing parameters but absent from the overview sheet: "
            + ", ".join(f"`{column}`" for column in missing_columns)
            + ". They are neither locked nor invalidating, because they are not there "
            "to be edited — the app's configuration and the uploaded sheet disagree."
        )


@st.fragment
def render_metadata_grid(dataset: ExperimentalDataset, experiment_column: str) -> None:
    """
    Render the grid and its button, and save what the grid changed on submit.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset mutated in place when the button is pressed.
    experiment_column : str
        Column of ``overview_df`` naming the experiments.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.

    Notes
    -----
    A fragment, so the submit reruns this function and nothing else. The page
    body not re-running would leave section 4's standing warning and its
    shortcut count stale, so an applied edit ends in an app-scoped rerun. That
    section sits *below* this one, which is what keeps its warning from pushing
    the grid down when it appears.
    """

    parked_summary = st.session_state.pop(METADATA_EDIT_SUMMARY_KEY, None)

    view = metadata_editor_view(dataset, experiment_column)

    with st.form(key=METADATA_EDITOR_FORM_KEY, clear_on_submit=False):
        edited_view = st.data_editor(
            view,
            key=metadata_editor_widget_key(),
            num_rows="fixed",
            disabled=locked_metadata_columns(dataset, experiment_column),
            width='stretch',
        )
        submitted = st.form_submit_button("💾 Apply metadata changes", width="stretch")

    # The grid sends nothing until the form is submitted, so there is only
    # something to compare against on the run the button was pressed.
    changed_cells = changed_metadata_cells(
        view, edited_view,
        editable_metadata_columns(dataset, experiment_column)) if submitted else {}

    if changed_cells:
        apply_metadata_edits(dataset, changed_cells, experiment_column)
        st.session_state[METADATA_EDIT_SUMMARY_KEY] = describe_saved_edits(changed_cells)

        # Section 4's warning and its "only experiments needing reprocessing"
        # checkbox are drawn by the page body, so a fragment-scoped rerun
        # cannot refresh them — they would stay stale until the next page
        # interaction. An app-scoped rerun refreshes them without moving the
        # page: measured at 686 -> 686 px with the button in view, because
        # what moved the page was the widget key changing and the status box
        # changing height, not the scope of the rerun. The section is below
        # this one, so its warning appearing costs the grid no movement.
        st.rerun(scope="app")

    render_editor_status(dataset, experiment_column, submitted, parked_summary)


def describe_saved_edits(changed_cells: dict) -> str:
    """
    Describe an applied edit, for the run that renders after it.

    Parameters
    ----------
    changed_cells : dict
        ``{experiment_name: {column: new_value}}`` that was stored.

    Returns
    -------
    str
        One line of status text.
    """

    changed_count = sum(len(values) for values in changed_cells.values())

    return (f"✅ Saved {changed_count} change(s) to {len(changed_cells)} experiment(s): "
            + ", ".join(sorted(changed_cells)))


def describe_editor_state(submitted: bool, parked_summary) -> str:
    """
    Say what the last press of the button did, or that nothing is stored yet.

    Parameters
    ----------
    submitted : bool
        Whether the button was pressed on this run.
    parked_summary : str or None
        Outcome of an edit applied by the previous run, if there was one.

    Returns
    -------
    str
        One line of status text.
    """

    if parked_summary:
        return parked_summary

    if submitted:
        return "Nothing had been changed, so nothing was stored."

    return "Edits are stored when you press **Apply metadata changes**."


def render_editor_status(dataset: ExperimentalDataset,
                         experiment_column: str,
                         submitted: bool,
                         parked_summary) -> None:
    """
    Report what the button did and what still needs reprocessing.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset the status is read from.
    experiment_column : str
        Column of ``overview_df`` naming the experiments.
    submitted : bool
        Whether the button was pressed on this run.
    parked_summary : str or None
        Outcome of an edit applied by the previous run, if there was one.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.

    Notes
    -----
    Always exactly two captions, whatever the state. An `st.success` box that
    comes and goes changes the height of the fragment and shifts everything
    below it on the page, which reads as the page moving under the reader —
    the complaint this section is meant to have stopped. Section 4 carries the
    same reprocessing list as a proper warning, where it has room to be loud.
    """

    st.caption(describe_editor_state(submitted, parked_summary))

    stale_experiments = select_experiments_needing_reprocessing(dataset, experiment_column)

    if stale_experiments:
        st.caption(
            f"⚠️ {len(stale_experiments)} experiment(s) need reprocessing before their "
            "results can be used: " + ", ".join(stale_experiments)
            + " — use section 4 below."
        )
        return

    st.caption("All experiments in the dataset are up to date with their metadata.")
