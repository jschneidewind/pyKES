"""
Tests for editing a dataset's metadata in place.

Two things are pinned here. The first is the **policy**: which columns may be
written, and which of those invalidate the results derived from them. The
second is the ``Processed`` flag, whose value has to be right at every
transition — the flag table of ``docs/metadata_editing.md``. The transitions
belonging to ingestion live in `test_data_processing.py` and those belonging
to reprocessing in `test_reprocessing.py`; the edit and sheet-upload halves
are here.
"""

import numpy as np
import pandas as pd
import pytest

from pyKES.database.data_processing import select_experiments_needing_reprocessing
from pyKES.database.database_experiments import Experiment, ExperimentalDataset
from pyKES.database.metadata_editing import (METADATA_LOADING_KEY, METADATA_PROCESSING_KEY,
                                             apply_metadata_edits, changed_metadata_cells,
                                             columns_invalidating_processing,
                                             editable_metadata_columns,
                                             locked_metadata_columns, metadata_editing_available,
                                             metadata_editor_view, missing_declared_columns,
                                             processing_relevant_metadata_columns)


EXPERIMENT_COLUMN = 'Experiment'
LOCKED_COLUMN = 'File name O2'
PROCESSING_COLUMN = 'Irradiance [mW/cm2]'
FREE_COLUMN = 'Comment'

# Declared but absent from the overview sheet, so the mismatch is visible
UNDECLARED_COLUMN = 'Fraction of photons reaching inside [-]'

# Held by overview_df but never ingested, so it needs processing rather than
# reprocessing
NOT_INGESTED = 'Exp_003'


@pytest.fixture
def dataset():
    experimental_dataset = ExperimentalDataset(
        overview_df=pd.DataFrame({
            EXPERIMENT_COLUMN: ['Exp_001', 'Exp_002', NOT_INGESTED],
            LOCKED_COLUMN: ['one.csv', 'two.csv', 'three.csv'],
            PROCESSING_COLUMN: [40.0, 80.0, 120.0],
            'Offset': [60, 60, 60],
            FREE_COLUMN: ['first', None, 'third'],
            'color': ['red', 'green', 'blue'],
            'group': ['Intensity', 'Intensity', 'Intensity'],
            'Processed': ['True', 'True', 'False'],
        }),
        processing_parameters={
            METADATA_LOADING_KEY: [EXPERIMENT_COLUMN, LOCKED_COLUMN],
            METADATA_PROCESSING_KEY: [PROCESSING_COLUMN, 'Offset', UNDECLARED_COLUMN],
        },
    )

    for experiment_name in ('Exp_001', 'Exp_002'):
        experimental_dataset.add_experiment(Experiment(
            experiment_name=experiment_name,
            raw_data_file=f'{experiment_name}.csv',
            color='red',
            group='Intensity',
            metadata={'experiment_name': experiment_name, PROCESSING_COLUMN: 40.0},
            raw_data={'signal': np.array([1.0, 2.0])},
            processed_data={'maximum': 2.0},
        ))

    return experimental_dataset


def edit_view(dataset, experiment_name, column, value):
    """Return the editor's view and a copy of it with one cell changed."""
    view = metadata_editor_view(dataset, EXPERIMENT_COLUMN)
    edited_view = view.copy()
    edited_view.loc[experiment_name, column] = value

    return view, edited_view


def apply_single_edit(dataset, experiment_name, column, value):
    """Drive one cell edit all the way through the editor's pipeline."""
    view, edited_view = edit_view(dataset, experiment_name, column, value)
    changed_cells = changed_metadata_cells(
        view, edited_view, editable_metadata_columns(dataset, EXPERIMENT_COLUMN))

    return apply_metadata_edits(dataset, changed_cells, EXPERIMENT_COLUMN)


def processed_flags(dataset):
    return dataset.overview_df['Processed'].tolist()


# ---------------------------------------------------------------------------
# Availability: the backward-compatibility gate
# ---------------------------------------------------------------------------

def test_editing_is_available_when_both_lists_are_declared(dataset):
    assert metadata_editing_available(dataset)


def test_editing_is_unavailable_for_a_dataset_declaring_nothing(dataset):
    # What a file written before the declarations existed loads as
    dataset.processing_parameters = {}

    assert not metadata_editing_available(dataset)


def test_editing_is_unavailable_when_only_one_list_is_declared(dataset):
    # Without the locked list, a filename column would be editable by omission
    dataset.processing_parameters = {METADATA_PROCESSING_KEY: [PROCESSING_COLUMN]}

    assert not metadata_editing_available(dataset)


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def test_columns_are_classified_by_the_declarations(dataset):
    # The experiment name is the view's index, so it is not a locked column
    assert locked_metadata_columns(dataset, EXPERIMENT_COLUMN) == [LOCKED_COLUMN]
    assert processing_relevant_metadata_columns(dataset) == [PROCESSING_COLUMN, 'Offset']
    assert missing_declared_columns(dataset) == [UNDECLARED_COLUMN]

    editable = editable_metadata_columns(dataset, EXPERIMENT_COLUMN)
    assert LOCKED_COLUMN not in editable
    assert 'Processed' not in editable
    assert set(editable) == {PROCESSING_COLUMN, 'Offset', FREE_COLUMN, 'color', 'group'}


def test_the_view_leads_with_the_declared_columns(dataset):
    view = metadata_editor_view(dataset, EXPERIMENT_COLUMN)

    # The derived processed flag is not a column of the editor at all
    assert 'Processed' not in view.columns
    assert list(view.columns)[:3] == [LOCKED_COLUMN, PROCESSING_COLUMN, 'Offset']
    assert list(view.index) == ['Exp_001', 'Exp_002', NOT_INGESTED]
    assert EXPERIMENT_COLUMN not in view.columns


def test_sheet_upload_columns_cover_both_declarations(dataset):
    assert columns_invalidating_processing(dataset) == [EXPERIMENT_COLUMN, LOCKED_COLUMN,
                                                        PROCESSING_COLUMN, 'Offset',
                                                        UNDECLARED_COLUMN]


def test_a_dataset_declaring_nothing_invalidates_on_any_difference(dataset):
    dataset.processing_parameters = {}

    # None is `update_overview_df`'s "every difference counts" sentinel
    assert columns_invalidating_processing(dataset) is None


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def test_a_changed_value_is_detected(dataset):
    view, edited_view = edit_view(dataset, 'Exp_001', PROCESSING_COLUMN, 55.0)

    changed = changed_metadata_cells(view, edited_view,
                                     editable_metadata_columns(dataset, EXPERIMENT_COLUMN))

    assert changed == {'Exp_001': {PROCESSING_COLUMN: 55.0}}


def test_a_cleared_value_is_detected(dataset):
    view, edited_view = edit_view(dataset, 'Exp_001', FREE_COLUMN, None)

    changed = changed_metadata_cells(view, edited_view,
                                     editable_metadata_columns(dataset, EXPERIMENT_COLUMN))

    assert list(changed) == ['Exp_001']
    assert pd.isna(changed['Exp_001'][FREE_COLUMN])


def test_a_filled_value_is_detected(dataset):
    # Exp_002 starts with an empty comment; NaN against NaN must not read as
    # a change, but NaN against a value must
    view, edited_view = edit_view(dataset, 'Exp_002', FREE_COLUMN, 'second')

    changed = changed_metadata_cells(view, edited_view,
                                     editable_metadata_columns(dataset, EXPERIMENT_COLUMN))

    assert changed == {'Exp_002': {FREE_COLUMN: 'second'}}


def test_an_untouched_table_reports_no_changes(dataset):
    view = metadata_editor_view(dataset, EXPERIMENT_COLUMN)

    assert changed_metadata_cells(view, view.copy(),
                                  editable_metadata_columns(dataset, EXPERIMENT_COLUMN)) == {}


def test_locked_columns_are_never_written(dataset):
    # The grid disables them; this is the second line of defence, so a widget
    # handing one back cannot reach overview_df
    view, edited_view = edit_view(dataset, 'Exp_001', LOCKED_COLUMN, 'somewhere_else.csv')

    changed = changed_metadata_cells(view, edited_view,
                                     editable_metadata_columns(dataset, EXPERIMENT_COLUMN))

    assert changed == {}


# ---------------------------------------------------------------------------
# Applying edits, and what they do to the flag
# ---------------------------------------------------------------------------

def test_a_processing_edit_clears_the_flag(dataset):
    invalidated = apply_single_edit(dataset, 'Exp_001', PROCESSING_COLUMN, 55.0)

    assert invalidated == ['Exp_001']
    assert processed_flags(dataset) == ['False', 'True', 'False']
    assert dataset.overview_df.loc[0, PROCESSING_COLUMN] == 55.0
    assert dataset.experiments['Exp_001'].metadata[PROCESSING_COLUMN] == 55.0


def test_a_free_edit_keeps_the_flag(dataset):
    invalidated = apply_single_edit(dataset, 'Exp_001', FREE_COLUMN, 'corrected')

    assert invalidated == []
    assert processed_flags(dataset) == ['True', 'True', 'False']
    assert dataset.experiments['Exp_001'].metadata[FREE_COLUMN] == 'corrected'


def test_several_experiments_can_be_edited_at_once(dataset):
    view = metadata_editor_view(dataset, EXPERIMENT_COLUMN)
    edited_view = view.copy()
    edited_view.loc['Exp_001', PROCESSING_COLUMN] = 55.0
    edited_view.loc['Exp_002', PROCESSING_COLUMN] = 95.0
    edited_view.loc['Exp_002', FREE_COLUMN] = 'second'

    changed_cells = changed_metadata_cells(
        view, edited_view, editable_metadata_columns(dataset, EXPERIMENT_COLUMN))
    invalidated = apply_metadata_edits(dataset, changed_cells, EXPERIMENT_COLUMN)

    assert invalidated == ['Exp_001', 'Exp_002']
    assert processed_flags(dataset) == ['False', 'False', 'False']
    assert dataset.overview_df[PROCESSING_COLUMN].tolist() == [55.0, 95.0, 120.0]


def test_color_and_group_follow_the_edit(dataset):
    apply_single_edit(dataset, 'Exp_001', 'color', 'darkgreen')
    apply_single_edit(dataset, 'Exp_001', 'group', 'Loading')

    assert dataset.experiments['Exp_001'].color == 'darkgreen'
    assert dataset.experiments['Exp_001'].group == 'Loading'
    # Neither is declared as processing-relevant, so nothing was invalidated
    assert processed_flags(dataset) == ['True', 'True', 'False']


def test_editing_a_row_without_an_experiment_only_touches_the_overview(dataset):
    invalidated = apply_single_edit(dataset, NOT_INGESTED, PROCESSING_COLUMN, 200.0)

    assert invalidated == [NOT_INGESTED]
    assert dataset.overview_df.loc[2, PROCESSING_COLUMN] == 200.0
    assert NOT_INGESTED not in dataset.experiments


def test_an_experiment_never_ingested_is_not_offered_for_reprocessing(dataset):
    apply_single_edit(dataset, 'Exp_001', PROCESSING_COLUMN, 55.0)

    # NOT_INGESTED is flagged 'False' too, but has no raw data to reprocess
    assert select_experiments_needing_reprocessing(dataset, EXPERIMENT_COLUMN) == ['Exp_001']


# ---------------------------------------------------------------------------
# The same policy applied to a re-uploaded sheet
# ---------------------------------------------------------------------------

def upload_sheet(dataset, **changed_columns):
    """Merge a sheet holding the dataset's rows with some columns replaced."""
    incoming_df = dataset.overview_df.drop(columns=['Processed']).copy()

    for column, values in changed_columns.items():
        incoming_df[column] = values

    dataset.update_overview_df(incoming_df, EXPERIMENT_COLUMN,
                               columns_invalidating_processing(dataset))


def test_a_sheet_changing_a_declared_column_clears_the_flag(dataset):
    upload_sheet(dataset, **{PROCESSING_COLUMN: [45.0, 80.0, 120.0]})

    assert processed_flags(dataset) == ['False', 'True', 'False']


def test_a_sheet_changing_only_a_comment_keeps_the_flag(dataset):
    upload_sheet(dataset, **{FREE_COLUMN: ['corrected', 'second', 'third']})

    assert processed_flags(dataset) == ['True', 'True', 'False']
    assert dataset.overview_df[FREE_COLUMN].tolist() == ['corrected', 'second', 'third']


def test_a_sheet_changing_nothing_keeps_the_flag(dataset):
    upload_sheet(dataset)

    assert processed_flags(dataset) == ['True', 'True', 'False']


def test_a_sheet_merged_without_declarations_clears_the_flag(dataset):
    # A dataset that declares nothing keeps the old behaviour: any difference
    # invalidates. Left to `combine_first`, the flag came back as NaN.
    dataset.processing_parameters = {}

    upload_sheet(dataset, **{FREE_COLUMN: ['corrected', 'second', 'third']})

    assert processed_flags(dataset) == ['False', 'False', 'False']
