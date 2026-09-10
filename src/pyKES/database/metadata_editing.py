"""Edit the metadata of a dataset in place, and record what that invalidates.

``overview_df`` is the single source of every experiment's metadata, and until
now the only way to correct a value in it was to upload a new Excel sheet. The
functions here let a caller write single cells instead, and — more importantly
— decide what each edit means for the data already derived from them.

Which columns may be edited is declared by the dataset itself, through two
lists in ``processing_parameters``::

    'metadata_used_for_raw_data_loading': ['Experiment',
                                           'File name O2',
                                           'Well or set-up number'],
    'metadata_used_for_processing': ['Irradiance A [mW/cm2]',
                                     'Liquid phase volume [mL]',
                                     'Offset'],

The policy those two lists express:

* **Raw-data-loading columns are locked.** A filename or a well number selects
  which measurement the experiment *is*; changing it does not correct the
  dataset, it describes a different one. That needs the corrected sheet and
  the raw files uploaded again.
* **Processing columns invalidate the results.** Editing one leaves
  ``processed_data`` describing metadata that is no longer there, so the
  experiment's ``Processed`` flag is cleared and the Data Upload page asks for
  a reprocessing run.
* **Every other column is free.** A comment or a label is read by nobody but
  the reader, so editing one changes nothing about the results.

A dataset that declares neither list gets no editor at all — see
`metadata_editing_available`, which is what keeps files written before the
declarations existed working exactly as before.
"""

from typing import List, Optional

import pandas as pd

from pyKES.database.data_processing import mark_experiments_unprocessed
from pyKES.database.database_experiments import (ExperimentalDataset,
                                                 METADATA_LOADING_KEY,
                                                 METADATA_PROCESSING_KEY,
                                                 NON_METADATA_OVERVIEW_COLUMNS,
                                                 columns_invalidating_processing,
                                                 declared_columns,
                                                 metadata_editing_available)


# The declaration lists themselves live with the dataset that carries them, in
# `pyKES.database.database_experiments`, and are re-exported here: they say
# what a dataset's pipeline depends on, which the loader and the merge need to
# know as much as the editor does. Imported by name above, so that
# `from pyKES.database.metadata_editing import METADATA_LOADING_KEY` and the
# rest keep working.


def locked_metadata_columns(dataset: ExperimentalDataset,
                            experiment_column: str) -> List[str]:
    """
    List the columns shown in the editor but never writable.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset holding the declarations and the overview sheet.
    experiment_column : str
        Column naming the experiments, which the editor uses as its index.

    Returns
    -------
    list of str
        Declared raw-data-loading columns present in ``overview_df``.
    """

    candidates = declared_columns(dataset, METADATA_LOADING_KEY)

    return [column for column in dict.fromkeys(candidates)
            if column in editor_columns(dataset, experiment_column)]


def processing_relevant_metadata_columns(dataset: ExperimentalDataset) -> List[str]:
    """
    List the editable columns whose change invalidates the processed data.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset holding the declarations and the overview sheet.

    Returns
    -------
    list of str
        Declared processing columns present in ``overview_df``.
    """

    return [column for column in declared_columns(dataset, METADATA_PROCESSING_KEY)
            if column in dataset.overview_df.columns
            and column not in NON_METADATA_OVERVIEW_COLUMNS]


def missing_declared_columns(dataset: ExperimentalDataset) -> List[str]:
    """
    List declared columns the overview sheet does not actually have.

    A declaration naming a column the sheet lacks is a mismatch between the
    app's configuration and the uploaded metadata: the column is neither
    locked nor invalidating, because it is not there to be edited. Surfaced so
    the mismatch is visible rather than silent.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset holding the declarations and the overview sheet.

    Returns
    -------
    list of str
        Declared names absent from ``overview_df``.
    """

    declared = (declared_columns(dataset, METADATA_LOADING_KEY)
                + declared_columns(dataset, METADATA_PROCESSING_KEY))

    return [column for column in dict.fromkeys(declared)
            if column not in dataset.overview_df.columns]


def editable_metadata_columns(dataset: ExperimentalDataset,
                              experiment_column: str) -> List[str]:
    """
    List the columns of the overview sheet a user may write to.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset holding the declarations and the overview sheet.
    experiment_column : str
        Column naming the experiments, which the editor uses as its index.

    Returns
    -------
    list of str
        Every overview column that is neither locked nor the experiment name.
    """

    locked = set(locked_metadata_columns(dataset, experiment_column))

    return [column for column in editor_columns(dataset, experiment_column)
            if column not in locked]


def editor_columns(dataset: ExperimentalDataset, experiment_column: str) -> List[str]:
    """
    List the overview columns the editor deals with at all.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset holding the overview sheet.
    experiment_column : str
        Column naming the experiments, which becomes the editor's index.

    Returns
    -------
    list of str
        Every overview column but the experiment name and the derived ones.

    Notes
    -----
    The processed flag is left out rather than shown read-only. It is owned by
    the processing pipeline, and an edit changes it *after* the grid for this
    run has already been drawn — so showing it put a stale ``True`` on screen
    directly above the warning saying the experiment needs reprocessing. The
    Dataset Overview table further down the page shows the flag.
    """

    return [column for column in dataset.overview_df.columns
            if column != experiment_column and column not in NON_METADATA_OVERVIEW_COLUMNS]


def metadata_editor_view(dataset: ExperimentalDataset,
                         experiment_column: str) -> pd.DataFrame:
    """
    Build the table the editor shows: the overview sheet, ordered for editing.

    Parameters
    ----------
    dataset : ExperimentalDataset
        Dataset whose ``overview_df`` is displayed.
    experiment_column : str
        Column naming the experiments; becomes the index.

    Returns
    -------
    view : pandas.DataFrame
        Copy of ``overview_df`` indexed by experiment name, with the locked
        columns first and the declared processing columns next, so the two
        kinds of edit are not scattered through the sheet's own column order.
    """

    ordered_columns = (locked_metadata_columns(dataset, experiment_column)
                       + processing_relevant_metadata_columns(dataset)
                       + editor_columns(dataset, experiment_column))

    # dict.fromkeys keeps the first occurrence of each name, so the two
    # declared groups lead and the remaining columns follow in sheet order.
    unique_columns = list(dict.fromkeys(ordered_columns))

    view = dataset.overview_df.set_index(experiment_column)[unique_columns]

    return widen_integer_columns(view, editable_metadata_columns(dataset, experiment_column))


def widen_integer_columns(view: pd.DataFrame, editable_columns: List[str]) -> pd.DataFrame:
    """
    Offer whole-number editable columns as fractional fields.

    Parameters
    ----------
    view : pandas.DataFrame
        Table about to be handed to the editing widget.
    editable_columns : list of str
        Columns the user may write to.

    Returns
    -------
    pandas.DataFrame
        The view with its editable integer columns held as floats.

    Notes
    -----
    An overview column holding only whole numbers arrives from Excel as
    ``int64``, and `st.data_editor` takes the field type from the dtype: it
    quietly rounds anything typed into such a column, so an offset corrected
    to 62.5 was stored as 62 with nothing to show that it had been changed.
    Neither an explicit fractional ``step`` nor a decimal format changes that
    — only the dtype does.
    """

    integer_columns = [column for column in editable_columns
                       if column in view.columns and pd.api.types.is_integer_dtype(view[column])]

    if not integer_columns:
        return view

    return view.astype({column: 'float64' for column in integer_columns})


def changed_metadata_cells(original_df: pd.DataFrame,
                           edited_df: pd.DataFrame,
                           editable_columns: List[str]) -> dict:
    """
    Find the cells an editing widget changed.

    Parameters
    ----------
    original_df, edited_df : pandas.DataFrame
        The table handed to the widget and the table it returned, both indexed
        by experiment name.
    editable_columns : List[str]
        Columns a change is accepted from. Restricting the comparison here
        means a locked column can never be written, whatever the widget hands
        back.

    Returns
    -------
    changed_cells : dict
        ``{experiment_name: {column: new_value}}``, holding only the cells
        that actually differ.
    """

    changed_cells = {}

    for column in editable_columns:
        original_values = original_df[column]
        edited_values = edited_df[column]

        # `eq` is False for NaN against NaN, so the both-null case has to be
        # excluded explicitly. DataFrame.compare would not do: it cannot tell
        # an unchanged cell from one the user cleared.
        changed = ~(original_values.eq(edited_values)
                    | (original_values.isna() & edited_values.isna()))

        # to_dict boxes NumPy scalars into the native Python types the
        # metadata dict of an ingested experiment already holds.
        for experiment_name, value in edited_values[changed].to_dict().items():
            changed_cells.setdefault(experiment_name, {})[column] = value

    return changed_cells


def assign_overview_value(overview_df: pd.DataFrame,
                          row_mask,
                          column: str,
                          value) -> None:
    """
    Write one edited value into the overview sheet, widening the column if need be.

    Parameters
    ----------
    overview_df : pandas.DataFrame
        Overview sheet, mutated in place.
    row_mask : pandas.Series
        Boolean mask selecting the experiment's row.
    column : str
        Column to write.
    value : Any
        New value.

    Returns
    -------
    None : None

    Notes
    -----
    A sheet column holding only whole numbers arrives as ``int64``, and pandas
    refuses to store a fraction in one — so an offset of 62.5 would either
    raise or land as 62. The column is widened to float first, which is also
    what lets `metadata_editor_view` offer it as a fractional field.
    """

    if isinstance(value, float) and not float(value).is_integer():
        if pd.api.types.is_integer_dtype(overview_df[column]):
            overview_df[column] = overview_df[column].astype('float64')

    overview_df.loc[row_mask, column] = value


def apply_metadata_edits(database: ExperimentalDataset,
                         changed_cells: dict,
                         experiment_column: str,
                         processing_relevant_columns: Optional[List[str]] = None) -> List[str]:
    """
    Write edited cells into a dataset and clear the flags they invalidate.

    Parameters
    ----------
    database : ExperimentalDataset
        Dataset mutated in place: ``overview_df``, the affected experiments'
        ``metadata``, and their processed flags.
    changed_cells : dict
        ``{experiment_name: {column: new_value}}``, as returned by
        `changed_metadata_cells`.
    experiment_column : str
        Column of ``overview_df`` naming the experiments.
    processing_relevant_columns : list of str, optional
        Columns whose change invalidates the processed data. Defaults to the
        dataset's own declaration.

    Returns
    -------
    invalidated_experiments : list of str
        Experiments now flagged as needing reprocessing, sorted by name.

    Notes
    -----
    Only ``overview_df`` is written here. The stored metadata of the affected
    experiments then follows from it through
    `ExperimentalDataset.synchronize_experiment_metadata`, so an edit cannot
    reach one of the two and miss the other.
    """

    if processing_relevant_columns is None:
        processing_relevant_columns = processing_relevant_metadata_columns(database)

    for experiment_name, edited_values in changed_cells.items():
        row_mask = database.overview_df[experiment_column].eq(experiment_name)

        for column, value in edited_values.items():
            assign_overview_value(database.overview_df, row_mask, column, value)

    database.synchronize_experiment_metadata(
        [name for name in changed_cells if name in database.experiments])

    invalidated_experiments = sorted(
        experiment_name for experiment_name, edited_values in changed_cells.items()
        if not set(edited_values).isdisjoint(processing_relevant_columns))

    mark_experiments_unprocessed(database, invalidated_experiments, experiment_column)

    return invalidated_experiments
