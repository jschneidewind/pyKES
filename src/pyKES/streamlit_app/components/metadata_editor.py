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

Edits save themselves. There is no submit button, and the grid lives in an
`st.fragment`, which is what makes that affordable: a committed cell reruns
the fragment alone, so the page body — which re-reads the uploaded workbook
and rewrites the whole HDF5 file to feed its download button — does not run.
Measured in headless Chromium: typing into a cell 1500 px to the right of the
grid's origin saved the value, left the page body un-executed, and left the
grid scrolled exactly where it was.

Both of those were bugs in the first version of this page, and both are worth
not reintroducing:

* **A submit button that ends in `st.rerun` moves the page.** The app-scoped
  rerun re-focuses the submit button and Streamlit's scroll container jumped
  ~300 px (measured 180 -> 476), pushing the grid off screen.
* **Changing the widget key resets the grid's horizontal scroll.** Scroll
  survives a fragment rerun and an app-scoped rerun alike (1200 -> 1200) and
  is lost only when the key changes (1200 -> 0), which is why the revision
  counter is bumped on an uploaded sheet and nothing else.
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
        "Correct metadata directly in the table below. **Edits save as you make them** — "
        "cells can be selected, dragged and pasted into as in a spreadsheet, so several "
        "experiments can be corrected in one go. Editing a column the processing function "
        "reads clears the experiment's `Processed` flag and lists it for reprocessing in "
        "section 3. Uploading a metadata sheet above replaces whatever is edited here."
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
    Render the grid, save whatever it changed, and report what that invalidated.

    A fragment on purpose: a committed cell reruns this function and nothing
    else, so saving an edit costs neither a re-read of the uploaded workbook
    nor a rewrite of the HDF5 file, and moves nothing on screen. The flip side
    is that the page body does not re-run either, so section 3's standing
    warning and its shortcut count catch up on the next page interaction —
    which is why the same information is repeated here, where it is live.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset mutated in place.
    experiment_column : str
        Column of ``overview_df`` naming the experiments.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """

    view = metadata_editor_view(dataset, experiment_column)

    edited_view = st.data_editor(
        view,
        key=metadata_editor_widget_key(),
        num_rows="fixed",
        disabled=locked_metadata_columns(dataset, experiment_column),
        width='stretch',
    )

    changed_cells = changed_metadata_cells(
        view, edited_view, editable_metadata_columns(dataset, experiment_column))

    if changed_cells:
        apply_metadata_edits(dataset, changed_cells, experiment_column)

    render_editor_status(dataset, experiment_column, changed_cells)


def render_editor_status(dataset: ExperimentalDataset,
                         experiment_column: str,
                         changed_cells: dict) -> None:
    """
    Report the edit just saved and what still needs reprocessing.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset the status is read from.
    experiment_column : str
        Column of ``overview_df`` naming the experiments.
    changed_cells : dict
        ``{experiment_name: {column: new_value}}`` saved by this run, empty
        when the grid was only redrawn.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """

    if changed_cells:
        changed_count = sum(len(values) for values in changed_cells.values())
        st.success(
            f"✅ Saved {changed_count} change(s) to {len(changed_cells)} experiment(s): "
            + ", ".join(sorted(changed_cells))
        )

    stale_experiments = select_experiments_needing_reprocessing(dataset, experiment_column)

    if stale_experiments:
        st.warning(
            f"⚠️ {len(stale_experiments)} experiment(s) need reprocessing before their "
            "results can be used: " + ", ".join(stale_experiments)
            + ". Use section 3 above."
        )
