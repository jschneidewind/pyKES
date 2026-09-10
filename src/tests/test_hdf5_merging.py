"""
Tests for merging several HDF5 files into one dataset.

Merging used to concatenate the overview sheets and drop rows that were
identical in every column, which is a weaker condition than describing the
same experiment: two files listing the same experiment with different
processed flags left two rows behind, and a column one of the sheets did not
carry emptied that column for the other file's experiments. The duplicate
experiments a merge skipped were only ever printed, so a deployed app dropped
them without saying anything.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from pyKES.database.database_experiments import Experiment, ExperimentalDataset
from pyKES.streamlit_app.components.metadata_editor import METADATA_EDITOR_REVISION_KEY
from src.tests.test_metadata_consistency import corrupt_stored_metadata
from src.tests.test_metadata_editor_page import run_page

HDF5_MIME = 'application/x-hdf'


def build_experiment(name, metadata, maximum=2.0):
    """An experiment carrying the given metadata and a recognisable result."""
    return Experiment(experiment_name=name,
                      raw_data_file=f'{name}.csv',
                      color='blue',
                      group='Intensity',
                      metadata=dict(metadata),
                      raw_data={'signal': np.array([1.0, 2.0])},
                      processed_data={'maximum': maximum})


def write_dataset(path, overview_df, experiments, experiment_column='Experiment'):
    """
    Save a dataset holding the given sheet and experiments, and return its path.

    Parameters
    ----------
    path : pathlib.Path
        File written to.
    overview_df : pandas.DataFrame
        Overview sheet.
    experiments : list of Experiment
    experiment_column : str, optional
        Column naming the experiments.

    Returns
    -------
    str
        The path written, ready to hand to `merge_hdf5_files`.
    """

    dataset = ExperimentalDataset(overview_df=overview_df, experiment_column=experiment_column)

    for experiment in experiments:
        dataset.add_experiment(experiment)

    dataset.save_to_hdf5(str(path), verbose=False)

    return str(path)


@pytest.fixture
def two_rows():
    """A sheet listing two experiments, only the first of them processed."""
    return pd.DataFrame({'Experiment': ['Exp_001', 'Exp_002'],
                         'Irradiance [mW/cm2]': [40.0, 80.0],
                         'Notes': ['first', 'second'],
                         'Processed': ['True', 'False']})


@pytest.fixture
def first_file(tmp_path, two_rows):
    """A file whose sheet lists both experiments but which only holds Exp_001."""
    return write_dataset(
        tmp_path / 'first.h5', two_rows,
        [build_experiment('Exp_001', {'Experiment': 'Exp_001',
                                      'Irradiance [mW/cm2]': 40.0, 'Notes': 'first'})])


@pytest.fixture
def second_file(tmp_path, two_rows):
    """The same sheet, but the file holds Exp_002 and has it flagged processed."""
    overview_df = two_rows.copy()
    overview_df.loc[overview_df['Experiment'] == 'Exp_002', 'Processed'] = 'True'

    return write_dataset(
        tmp_path / 'second.h5', overview_df,
        [build_experiment('Exp_002', {'Experiment': 'Exp_002',
                                      'Irradiance [mW/cm2]': 80.0, 'Notes': 'second'},
                          maximum=9.0)])


# =============================================================================
# Overview rows
# =============================================================================

def test_two_sheets_listing_the_same_experiment_produce_one_row(first_file, second_file):
    # The reported bug: both sheets list both experiments, and the two files
    # hold different ones, so the rows differ in their processed flag and
    # dropping identical rows left the sheet with a duplicate.
    merged = ExperimentalDataset.merge_hdf5_files([first_file, second_file])

    assert len(merged.overview_df) == 2
    assert sorted(merged.overview_df['Experiment']) == ['Exp_001', 'Exp_002']


def test_the_merged_sheet_is_indexed_contiguously(first_file, second_file):
    # Dropping rows after concatenating left gaps in the index, which every
    # positional row lookup — the editor grid included — then reads wrongly.
    merged = ExperimentalDataset.merge_hdf5_files([first_file, second_file])

    assert list(merged.overview_df.index) == list(range(len(merged.overview_df)))


def test_the_row_of_the_file_that_brought_the_experiment_wins(first_file, second_file):
    # Both sheets describe Exp_002, but only the second file holds it. Keeping
    # the first file's row would describe the second file's measurements with a
    # sheet that says they were never processed.
    merged = ExperimentalDataset.merge_hdf5_files([first_file, second_file])

    processed_flags = merged.overview_df.set_index('Experiment')['Processed'].to_dict()

    assert processed_flags == {'Exp_001': 'True', 'Exp_002': 'True'}


def test_a_row_naming_no_experiment_is_kept(tmp_path, two_rows):
    # A sheet can carry a row that names no experiment at all; it cannot be a
    # duplicate of one, so matching rows by name must not swallow it.
    with_blank_row = pd.concat(
        [two_rows, pd.DataFrame([{'Experiment': None, 'Notes': 'a note to self'}])],
        ignore_index=True)
    path = write_dataset(tmp_path / 'blank.h5', with_blank_row, [])

    merged = ExperimentalDataset.merge_hdf5_files([path])

    assert len(merged.overview_df) == 3
    assert 'a note to self' in list(merged.overview_df['Notes'])


def test_a_column_one_sheet_lacks_keeps_the_other_files_values(tmp_path, two_rows):
    # The merged sheet gains a column the second file never had, so its rows
    # have nothing in it. Since the sheet is what the stored metadata is
    # re-derived from, that emptied a measured value.
    first = write_dataset(
        tmp_path / 'with_column.h5', two_rows,
        [build_experiment('Exp_001', {'Experiment': 'Exp_001',
                                      'Irradiance [mW/cm2]': 40.0, 'Notes': 'first'})])
    second = write_dataset(
        tmp_path / 'without_column.h5',
        pd.DataFrame({'Experiment': ['Exp_003'], 'Notes': ['third'], 'Processed': ['True']}),
        [build_experiment('Exp_003', {'Experiment': 'Exp_003', 'Notes': 'third',
                                      'Irradiance [mW/cm2]': 120.0})])

    merged = ExperimentalDataset.merge_hdf5_files([first, second])

    assert merged.experiments['Exp_003'].metadata['Irradiance [mW/cm2]'] == 120.0
    assert merged.overview_df.set_index('Experiment').loc[
        'Exp_003', 'Irradiance [mW/cm2]'] == 120.0


# =============================================================================
# Duplicate experiments
# =============================================================================

def test_a_duplicate_experiment_is_reported_and_the_first_one_kept(tmp_path, two_rows,
                                                                   first_file):
    later = write_dataset(
        tmp_path / 'later.h5', two_rows,
        [build_experiment('Exp_001', {'Experiment': 'Exp_001',
                                      'Irradiance [mW/cm2]': 40.0, 'Notes': 'first'},
                          maximum=99.0)])

    merged = ExperimentalDataset.merge_hdf5_files([first_file, later],
                                                  source_labels=['loaded', 'uploaded.h5'])

    assert merged.merge_report['skipped_experiments'] == {'Exp_001': 'uploaded.h5'}
    assert merged.experiments['Exp_001'].processed_data['maximum'] == 2.0


def test_nothing_is_reported_as_skipped_for_a_clean_merge(first_file, second_file):
    merged = ExperimentalDataset.merge_hdf5_files([first_file, second_file])

    assert merged.merge_report['skipped_experiments'] == {}
    assert merged.merge_report['metadata_corrected'] == {}


def test_the_report_names_the_sources_it_was_given(first_file, second_file):
    merged = ExperimentalDataset.merge_hdf5_files([first_file, second_file],
                                                  source_labels=['loaded', 'uploaded.h5'])

    assert merged.merge_report['sources'] == ['loaded', 'uploaded.h5']
    assert merged.version['merged_from'] == ['loaded', 'uploaded.h5']


def test_the_sources_default_to_the_filenames(first_file):
    merged = ExperimentalDataset.merge_hdf5_files([first_file])

    assert merged.merge_report['sources'] == [first_file]


# =============================================================================
# Consistency of the merged dataset
# =============================================================================

def test_the_merged_dataset_satisfies_the_metadata_invariant(first_file, second_file,
                                                             tmp_path):
    merged = ExperimentalDataset.merge_hdf5_files([first_file, second_file])

    assert merged.metadata_divergences() == {}

    # The invariant is what a save enforces, so this is the end-to-end check.
    merged.save_to_hdf5(str(tmp_path / 'merged.h5'), verbose=False)


def test_a_source_file_whose_metadata_diverged_is_reported(tmp_path, two_rows, first_file):
    diverged = write_dataset(
        tmp_path / 'diverged.h5',
        pd.DataFrame({'Experiment': ['Exp_004'], 'Irradiance [mW/cm2]': [160.0],
                      'Notes': ['fourth'], 'Processed': ['True']}),
        [build_experiment('Exp_004', {'Experiment': 'Exp_004',
                                      'Irradiance [mW/cm2]': 160.0, 'Notes': 'fourth'})])
    corrupt_stored_metadata(diverged, 'Exp_004', 'Irradiance [mW/cm2]', 999.0)

    merged = ExperimentalDataset.merge_hdf5_files([first_file, diverged])

    assert merged.merge_report['metadata_corrected'] == {
        'Exp_004': {'Irradiance [mW/cm2]': (999.0, 160.0)}}
    assert merged.experiments['Exp_004'].metadata['Irradiance [mW/cm2]'] == 160.0


def test_dataset_level_configuration_is_carried_over(tmp_path, two_rows):
    first = ExperimentalDataset(overview_df=two_rows,
                                plotting_instruction={'xlabel': 'Time'},
                                processing_parameters={'offset': 60})
    first.save_to_hdf5(str(tmp_path / 'configured.h5'), verbose=False)
    second = write_dataset(tmp_path / 'plain.h5', two_rows, [])

    merged = ExperimentalDataset.merge_hdf5_files([str(tmp_path / 'configured.h5'), second])

    assert merged.plotting_instruction == {'xlabel': 'Time'}
    assert merged.processing_parameters == {'offset': 60}


# =============================================================================
# Sources that cannot be merged
# =============================================================================

def test_files_naming_their_experiments_differently_are_refused(tmp_path, two_rows,
                                                                first_file):
    renamed = write_dataset(
        tmp_path / 'renamed.h5',
        pd.DataFrame({'Sample': ['Exp_005'], 'Notes': ['fifth'], 'Processed': ['True']}),
        [build_experiment('Exp_005', {'Sample': 'Exp_005', 'Notes': 'fifth'})],
        experiment_column='Sample')

    with pytest.raises(ValueError, match="different overview columns"):
        ExperimentalDataset.merge_hdf5_files([first_file, renamed],
                                             source_labels=['loaded', 'renamed.h5'])


def test_a_file_without_a_sheet_adopts_the_other_ones_column(tmp_path, first_file):
    # A dataset started fresh carries the default column name because
    # something has to be the default, not because its sheet says so.
    empty = ExperimentalDataset(overview_df=pd.DataFrame())
    empty.save_to_hdf5(str(tmp_path / 'empty.h5'), verbose=False)

    renamed = write_dataset(
        tmp_path / 'renamed.h5',
        pd.DataFrame({'Sample': ['Exp_005'], 'Notes': ['fifth'], 'Processed': ['True']}),
        [build_experiment('Exp_005', {'Sample': 'Exp_005', 'Notes': 'fifth'})],
        experiment_column='Sample')

    merged = ExperimentalDataset.merge_hdf5_files([str(tmp_path / 'empty.h5'), renamed])

    assert merged.experiment_column == 'Sample'
    assert merged.metadata_divergences() == {}


# =============================================================================
# The page's merge section
# =============================================================================

def build_file_to_merge(tmp_path, two_rows):
    """
    An HDF5 file holding one experiment the page's dataset has and one it lacks.

    Parameters
    ----------
    tmp_path : pathlib.Path
    two_rows : pandas.DataFrame
        Unused shape reference; the sheet is written to match the page's own.

    Returns
    -------
    bytes
        The file's contents, ready to put in the merge uploader.
    """

    overview_df = pd.DataFrame({'Experiment': ['Exp_002', 'Exp_003'],
                                'File name O2': ['two.csv', 'three.csv'],
                                'Irradiance [mW/cm2]': [80.0, 120.0],
                                'Comment': ['second', 'third'],
                                'Processed': ['True', 'True']})
    path = write_dataset(
        tmp_path / 'incoming.h5', overview_df,
        [build_experiment('Exp_002', {'experiment_name': 'Exp_002'}, maximum=7.0),
         build_experiment('Exp_003', {'experiment_name': 'Exp_003'}, maximum=7.0)])

    return pathlib.Path(path).read_bytes()


def merge_on_page(tmp_path, two_rows):
    """Run the Data Upload page and merge one file in through its uploader."""
    app = run_page()

    # The third uploader on the page: metadata workbook, raw data, merge.
    app.file_uploader[2].set_value([('incoming.h5', build_file_to_merge(tmp_path, two_rows),
                                     HDF5_MIME)])
    next(button for button in app.button if 'Merge HDF5' in button.label).click()
    app.run(timeout=90)

    assert [element.value for element in app.exception] == []

    return app


def test_the_page_describes_the_merged_dataset_on_the_same_press(tmp_path, two_rows):
    # The reported bug: merging built a new dataset object, but every section
    # below the merge kept rendering the one the run had started with — so the
    # download button offered the unmerged file.
    app = merge_on_page(tmp_path, two_rows)

    statistics = {element.label: element.value for element in app.metric}

    assert statistics['Experiments Loaded'] == '3'
    assert statistics['Overview Records'] == '3'


def test_the_page_warns_about_the_duplicates_it_skipped(tmp_path, two_rows):
    # Skipped duplicates were only printed, so a deployed app dropped
    # experiments without telling anybody.
    app = merge_on_page(tmp_path, two_rows)

    warnings = [element.value for element in app.warning]

    assert any('skipped' in warning and 'Exp_002' in warning for warning in warnings)


def test_the_page_reseeds_the_editor_grid_after_a_merge(tmp_path, two_rows):
    # The grid addresses its rows by position, and the merged sheet is a
    # different table: a delta still held client-side would land on another
    # experiment's row.
    app = run_page()
    revision_before = app.session_state[METADATA_EDITOR_REVISION_KEY]

    app.file_uploader[2].set_value([('incoming.h5', build_file_to_merge(tmp_path, two_rows),
                                     HDF5_MIME)])
    next(button for button in app.button if 'Merge HDF5' in button.label).click()
    app.run(timeout=90)

    assert app.session_state[METADATA_EDITOR_REVISION_KEY] > revision_before


def test_a_sheet_the_merged_column_cannot_address_is_refused(tmp_path, first_file):
    # A sheet naming its experiments in a column the merge does not resolve to
    # holds rows nothing can match to an experiment. Concatenating carried them
    # along unreachable; dropping them silently would be worse.
    mismatched = ExperimentalDataset(
        overview_df=pd.DataFrame({'Sample': ['Exp_005'], 'Notes': ['fifth']}))
    mismatched.save_to_hdf5(str(tmp_path / 'mismatched.h5'), verbose=False)

    with pytest.raises(ValueError, match="rows name no experiment"):
        ExperimentalDataset.merge_hdf5_files(
            [first_file, str(tmp_path / 'mismatched.h5')],
            source_labels=['loaded', 'mismatched.h5'])


def test_two_uploads_sharing_a_filename_are_both_merged(tmp_path):
    # Both were staged under the uploaded name, so one overwrote the other on
    # disk and was then merged twice — the difference showing up only as a
    # skipped duplicate.
    payloads = []

    for name in ('Exp_003', 'Exp_004'):
        path = write_dataset(
            tmp_path / f'{name}.h5',
            pd.DataFrame({'Experiment': [name], 'Comment': [name], 'Processed': ['True']}),
            [build_experiment(name, {'experiment_name': name})])
        payloads.append(pathlib.Path(path).read_bytes())

    app = run_page()
    app.file_uploader[2].set_value([('data.h5', payloads[0], HDF5_MIME),
                                    ('data.h5', payloads[1], HDF5_MIME)])
    next(button for button in app.button if 'Merge HDF5' in button.label).click()
    app.run(timeout=90)

    dataset = app.session_state['experimental_dataset']

    assert sorted(dataset.experiments) == ['Exp_001', 'Exp_002', 'Exp_003', 'Exp_004']
    assert dataset.merge_report['skipped_experiments'] == {}


def test_a_column_a_source_sheet_lacks_is_not_reported_as_a_divergence(tmp_path, first_file):
    # The merged sheet gains columns this file's sheet never had, so its rows
    # are blank in them and its experiments never carried the keys. Adopting a
    # blank cell replaces no value with no value, and reporting it made every
    # merge of files with different column sets warn about metadata.
    narrow = write_dataset(
        tmp_path / 'narrow.h5',
        pd.DataFrame({'Experiment': ['Exp_009'], 'Processed': ['True']}),
        [build_experiment('Exp_009', {'experiment_name': 'Exp_009'})])

    merged = ExperimentalDataset.merge_hdf5_files([first_file, narrow])

    assert merged.merge_report['metadata_corrected'] == {}
