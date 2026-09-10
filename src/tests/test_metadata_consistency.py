"""
Tests for the invariant that ``overview_df`` owns every experiment's metadata.

The two could diverge before, and did in real files: an edit reached the
overview table while a re-merged sheet reverted it, or reached an experiment's
stored metadata while the table kept the old value. Whichever way round, the
grid and the plots then disagreed about the same number.

The dataset now closes that off from three sides — every experiment that
enters is aligned with its row, every overview change is pushed into the
experiments, and a save refuses to write a dataset where the two disagree —
and repairs files written before the guarantee existed as they load.
"""

import numpy as np
import pandas as pd
import pytest

from pyKES.database.database_experiments import (METADATA_KEY_ABSENT, Experiment,
                                                 ExperimentalDataset,
                                                 describe_metadata_divergences,
                                                 overview_metadata_for_experiment, values_agree)


def build_experiment(name, metadata):
    """An experiment carrying the given metadata and nothing else of interest."""
    return Experiment(experiment_name=name,
                      raw_data_file=f'{name}.csv',
                      color='blue',
                      group='Intensity',
                      metadata=dict(metadata),
                      raw_data={'signal': np.array([1.0, 2.0])},
                      processed_data={'maximum': 2.0})


@pytest.fixture
def dataset():
    return ExperimentalDataset(
        overview_df=pd.DataFrame({'Experiment': ['Exp_001', 'Exp_002'],
                                  'Irradiance [mW/cm2]': [40.0, 80.0],
                                  'Notes': ['first', 'second'],
                                  'Processed': ['True', 'True']}))


# =============================================================================
# Value comparison
# =============================================================================

def test_the_same_number_agrees_across_types():
    # The same cell arrives as a NumPy scalar from a DataFrame and as a plain
    # float from JSON; that is a round trip, not a divergence.
    assert values_agree(np.float64(40.0), 40.0)
    assert values_agree(40, 40.0)
    assert values_agree(np.bool_(True), True)


def test_two_missing_values_agree():
    assert values_agree(np.nan, np.nan)
    assert values_agree(None, float('nan'))


def test_a_different_value_does_not_agree():
    assert not values_agree(40.0, 55.0)
    assert not values_agree(np.nan, 40.0)
    assert not values_agree('blue', 'red')


# =============================================================================
# What the overview sheet owns
# =============================================================================

def test_the_processed_flag_is_not_metadata(dataset):
    # It changes when an experiment is processed, without its metadata
    # changing, so policing it would report a divergence after every run.
    owned = overview_metadata_for_experiment(dataset.overview_df, 'Experiment', 'Exp_001')

    assert 'Processed' not in owned
    assert owned['Irradiance [mW/cm2]'] == 40.0


def test_an_experiment_without_a_row_is_left_alone(dataset):
    dataset.add_experiment(build_experiment('Exp_099', {'Irradiance [mW/cm2]': 12.0}))

    # Nothing in the sheet claims anything about it, so nothing is imposed.
    assert dataset.experiments['Exp_099'].metadata['Irradiance [mW/cm2]'] == 12.0
    assert dataset.metadata_divergences() == {}


# =============================================================================
# Every experiment is aligned as it enters
# =============================================================================

def test_adding_an_experiment_imposes_the_sheet(dataset):
    repaired = dataset.add_experiment(
        build_experiment('Exp_001', {'Irradiance [mW/cm2]': 999.0}))

    assert dataset.experiments['Exp_001'].metadata['Irradiance [mW/cm2]'] == 40.0
    assert repaired['Irradiance [mW/cm2]'] == (999.0, 40.0)
    # The row's other columns were absent from the metadata and are added too
    assert repaired['Notes'] == (METADATA_KEY_ABSENT, 'first')


def test_missing_columns_are_filled_in(dataset):
    repaired = dataset.add_experiment(build_experiment('Exp_001', {'experiment_name': 'Exp_001'}))

    assert dataset.experiments['Exp_001'].metadata['Notes'] == 'first'
    assert repaired['Notes'] == (METADATA_KEY_ABSENT, 'first')


def test_keys_the_sheet_does_not_have_survive(dataset):
    dataset.add_experiment(build_experiment('Exp_001', {'experiment_name': 'Exp_001',
                                                        'boost': 2.0}))

    # A metadata_retrival_function is free to add keys of its own.
    assert dataset.experiments['Exp_001'].metadata['boost'] == 2.0


def test_colour_and_group_follow_the_sheet(dataset):
    dataset.overview_df['color'] = ['red', 'green']
    dataset.overview_df['group'] = ['Reference', 'Intensity']

    dataset.add_experiment(build_experiment('Exp_001', {}))

    assert dataset.experiments['Exp_001'].color == 'red'
    assert dataset.experiments['Exp_001'].group == 'Reference'


def test_a_blank_colour_cell_does_not_replace_a_usable_one(dataset):
    dataset.overview_df['color'] = [np.nan, 'green']

    dataset.add_experiment(build_experiment('Exp_001', {}))

    # Assigning the NaN would leave a colour no plotting library accepts.
    assert dataset.experiments['Exp_001'].color == 'blue'


# =============================================================================
# Overview changes reach the experiments
# =============================================================================

def test_merging_a_sheet_pushes_it_into_the_experiments(dataset):
    dataset.add_experiment(build_experiment('Exp_001', {}))

    dataset.update_overview_df(
        pd.DataFrame({'Experiment': ['Exp_001'], 'Irradiance [mW/cm2]': [55.0],
                      'Notes': ['corrected']}),
        'Experiment')

    assert dataset.experiments['Exp_001'].metadata['Irradiance [mW/cm2]'] == 55.0
    assert dataset.experiments['Exp_001'].metadata['Notes'] == 'corrected'
    assert dataset.metadata_divergences() == {}


def test_merging_a_sheet_adopts_its_key_column():
    dataset = ExperimentalDataset()

    dataset.update_overview_df(pd.DataFrame({'Run': ['A'], 'value': [1.0]}), 'Run')

    assert dataset.experiment_column == 'Run'


def test_synchronizing_reports_what_it_corrected(dataset):
    dataset.add_experiment(build_experiment('Exp_001', {}))

    # Reaching around the dataset is exactly what used to go unnoticed.
    dataset.experiments['Exp_001'].metadata['Irradiance [mW/cm2]'] = 999.0

    assert dataset.synchronize_experiment_metadata() == {
        'Exp_001': {'Irradiance [mW/cm2]': (999.0, 40.0)}}
    assert dataset.experiments['Exp_001'].metadata['Irradiance [mW/cm2]'] == 40.0
    assert dataset.metadata_divergences() == {}


# =============================================================================
# The file on disk
# =============================================================================

def test_a_diverged_dataset_refuses_to_save(dataset, tmp_path):
    dataset.add_experiment(build_experiment('Exp_001', {}))
    dataset.experiments['Exp_001'].metadata['Irradiance [mW/cm2]'] = 999.0

    with pytest.raises(ValueError, match='Irradiance'):
        dataset.save_to_hdf5(str(tmp_path / 'diverged.h5'))

    assert not (tmp_path / 'diverged.h5').exists()


def test_a_consistent_dataset_round_trips(dataset, tmp_path):
    dataset.add_experiment(build_experiment('Exp_001', {'boost': 2.0}))
    path = str(tmp_path / 'consistent.h5')

    dataset.save_to_hdf5(path, verbose=False)
    loaded = ExperimentalDataset.load_from_hdf5(path)

    assert loaded.experiment_column == 'Experiment'
    assert loaded.metadata_repair_report == {}
    assert loaded.metadata_divergences() == {}
    assert loaded.experiments['Exp_001'].metadata['boost'] == 2.0


def test_a_file_written_before_the_guarantee_is_repaired_on_load(dataset, tmp_path):
    dataset.add_experiment(build_experiment('Exp_001', {}))
    path = str(tmp_path / 'legacy.h5')
    dataset.save_to_hdf5(path, verbose=False)

    # Rewrite the stored metadata behind the dataset's back, the way a file
    # written before the invariant existed can already hold it.
    corrupt_stored_metadata(path, 'Exp_001', 'Irradiance [mW/cm2]', 999.0)

    loaded = ExperimentalDataset.load_from_hdf5(path)

    assert loaded.experiments['Exp_001'].metadata['Irradiance [mW/cm2]'] == 40.0
    assert loaded.metadata_repair_report == {
        'Exp_001': {'Irradiance [mW/cm2]': (999.0, 40.0)}}
    assert 'Irradiance' in describe_metadata_divergences(loaded.metadata_repair_report)


def corrupt_stored_metadata(path, experiment_name, column, value):
    """Overwrite one stored metadata value inside an HDF5 file."""
    import h5py

    from pyKES.database.database_experiments import sanitize_key

    with h5py.File(path, 'a') as h5_file:
        metadata_group = h5_file[experiment_name]['metadata']
        key = sanitize_key(column)
        del metadata_group[key]
        metadata_group.create_dataset(key, data=value)


def test_a_dataset_naming_its_experiments_differently_round_trips(tmp_path):
    dataset = ExperimentalDataset(
        overview_df=pd.DataFrame({'Run': ['A'], 'value': [1.0]}),
        experiment_column='Run')
    dataset.add_experiment(build_experiment('A', {}))
    path = str(tmp_path / 'renamed.h5')

    dataset.save_to_hdf5(path, verbose=False)
    loaded = ExperimentalDataset.load_from_hdf5(path)

    assert loaded.experiment_column == 'Run'
    assert loaded.experiments['A'].metadata['value'] == 1.0
    assert loaded.metadata_divergences() == {}
