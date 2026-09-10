"""
Spreadsheet-style metadata editor for the Data Upload page.

``overview_df`` is rendered as an `st.data_editor` grid, one row per
experiment, so a wrong irradiance or a missing volume can be corrected without
producing a new Excel sheet. The grid is what makes editing many experiments
at once cheap: Streamlit's editor already supports multi-cell selection,
drag-fill and pasting a block straight out of Excel, so no bulk-edit UI of our
own is needed.

Which columns may be written, and what each write costs, is decided by the
dataset rather than by this page — see
`pyKES.database.metadata_editing`. Datasets that declare nothing get an
explanatory note instead of a grid, which is what keeps files written before
the declarations existed working unchanged.

The grid and its submit button live in a form on purpose. Outside one, every
committed cell reruns the whole page, and the Data Upload page rewrites the
entire HDF5 file on each run to feed its download button — a cost worth paying
once per edit session, not once per cell.
"""

import streamlit as st

from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.database.metadata_editing import (METADATA_LOADING_KEY, METADATA_PROCESSING_KEY,
                                             apply_metadata_edits, changed_metadata_cells,
                                             editable_metadata_columns,
                                             locked_metadata_columns, metadata_editing_available,
                                             metadata_editor_view, missing_declared_columns)
from pyKES.streamlit_app.config_interface import DataUploadConfig


# Bumped on every applied edit and carried in the widget key. Load-bearing:
# `st.data_editor` keeps its `edited_rows` delta in session state and replays
# it on top of whatever data it is handed, so a stale delta would silently
# reassert an old value over the applied one. A new key re-seeds it clean.
METADATA_EDITOR_REVISION_KEY = 'metadata_editor_revision'

# Outcome of the last applied edit, rendered on the run *after* it. Written
# before `st.rerun`, which discards everything the finished run had drawn.
METADATA_EDIT_SUMMARY_KEY = 'metadata_editor_last_summary'


def render_metadata_editor(config: DataUploadConfig, dataset: ExperimentalDataset) -> None:
    """
    Render the metadata grid and apply its edits on submit.

    Parameters
    ----------
    config : DataUploadConfig
        Supplies the column naming the experiments.
    dataset : ExperimentalDataset
        Dataset whose metadata is edited; mutated in place on submit.

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

    _render_last_edit_summary()

    experiment_column = config.metadata_excel_experiment_column
    locked_columns = locked_metadata_columns(dataset, experiment_column)

    _render_editing_policy(locked_columns, dataset)

    view = metadata_editor_view(dataset, experiment_column)

    with st.form(key="metadata_editor_form", clear_on_submit=False):
        edited_view = st.data_editor(
            view,
            key=f"metadata_editor_{st.session_state.setdefault(METADATA_EDITOR_REVISION_KEY, 0)}",
            num_rows="fixed",
            disabled=locked_columns,
            width='stretch',
        )
        submitted = st.form_submit_button("💾 Apply metadata changes", width="stretch")

    if not submitted:
        return

    _apply_edited_view(dataset, view, edited_view, experiment_column)


def _render_editing_policy(locked_columns: list, dataset: ExperimentalDataset) -> None:
    """
    Explain which columns are locked and which invalidate the processed data.

    Parameters
    ----------
    locked_columns : list of str
        Columns rendered read-only in the grid.
    dataset : ExperimentalDataset
        Dataset whose declarations are described.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """

    st.markdown(
        "Correct metadata directly in the table below, then apply the changes. "
        "Cells can be selected, dragged and pasted into as in a spreadsheet, so "
        "several experiments can be corrected in one go. Editing a column the "
        "processing function reads clears the experiment's `Processed` flag and "
        "lists it for reprocessing in section 3."
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


def _apply_edited_view(dataset: ExperimentalDataset,
                       view,
                       edited_view,
                       experiment_column: str) -> None:
    """
    Write back what the grid changed, then rerun with a clean editor.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset mutated in place.
    view, edited_view : pandas.DataFrame
        The table handed to the grid and the table it returned.
    experiment_column : str
        Column of ``overview_df`` naming the experiments.

    Returns
    -------
    None : None
    """

    changed_cells = changed_metadata_cells(
        view, edited_view, editable_metadata_columns(dataset, experiment_column))

    if not changed_cells:
        st.info("No changes to apply.")
        return

    invalidated_experiments = apply_metadata_edits(dataset, changed_cells, experiment_column)

    st.session_state[METADATA_EDIT_SUMMARY_KEY] = {
        'changed_cells': sum(len(values) for values in changed_cells.values()),
        'changed_experiments': len(changed_cells),
        'invalidated_experiments': invalidated_experiments,
    }
    st.session_state[METADATA_EDITOR_REVISION_KEY] += 1

    st.rerun()


def _render_last_edit_summary() -> None:
    """
    Report the edit applied by the previous run, once.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """

    summary = st.session_state.pop(METADATA_EDIT_SUMMARY_KEY, None)

    if summary is None:
        return

    st.success(
        f"✅ Applied {summary['changed_cells']} change(s) to "
        f"{summary['changed_experiments']} experiment(s)."
    )

    if summary['invalidated_experiments']:
        st.warning(
            "⚠️ Processing-relevant metadata changed — these experiments need "
            "reprocessing before their results can be used: "
            + ", ".join(summary['invalidated_experiments'])
        )
