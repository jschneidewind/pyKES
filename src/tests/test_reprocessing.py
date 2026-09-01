"""
Tests for reprocessing experiments that are already held by a dataset.

The processing callables are synthetic: `process_raw_data` multiplies the raw
signal by a factor taken from the metadata, so a changed overview sheet and a
changed processing function are both visible in the results.
"""

import numpy as np
import pandas as pd
import pytest

from pyKES.database.data_processing import reprocess_experiments, reprocess_single_experiment
from pyKES.database.database_experiments import Experiment, ExperimentalDataset
from pyKES.utilities.version_information import LAST_PROCESSED_KEY, PYKES_VERSION_KEY


FAILING_EXPERIMENT = 'Exp_002'


def retrieve_metadata(experiment_name, overview_df):
    row = overview_df.loc[overview_df['Experiment'] == experiment_name].iloc[0].to_dict()
    row['experiment_name'] = experiment_name

    return row


def process_raw_data(raw_data_dict, metadata_dict):
    return {'maximum': float(np.max(raw_data_dict['signal'])) * metadata_dict['scale']}


def failing_processing(raw_data_dict, metadata_dict):
    if metadata_dict['experiment_name'] == FAILING_EXPERIMENT:
        raise ValueError("processing unavailable")

    return process_raw_data(raw_data_dict, metadata_dict)


class ProgressRecorder:
    """Records every `(completed, total, experiment_name)` triple it receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, completed, total, experiment_name):
        self.calls.append((completed, total, experiment_name))


@pytest.fixture
def dataset():
    experimental_dataset = ExperimentalDataset(
        overview_df=pd.DataFrame({'Experiment': ['Exp_001', 'Exp_002'],
                                  'scale': [10.0, 100.0]})
    )

    for name in ('Exp_001', 'Exp_002'):
        experimental_dataset.add_experiment(Experiment(
            experiment_name=name,
            raw_data_file=f'{name}.csv',
            color='blue',
            group='Intensity',
            metadata={'experiment_name': name, 'scale': 1.0},
            raw_data={'signal': np.array([1.0, 2.0, 3.0])},
            processed_data={'maximum': -1.0},
        ))

    return experimental_dataset


def test_stored_metadata_is_reused_when_not_refreshed(dataset):
    results = reprocess_experiments(dataset, process_raw_data)

    assert all(result['success'] for result in results)
    # scale = 1.0 from the stored metadata, not the 10.0 of the overview sheet
    assert dataset.experiments['Exp_001'].processed_data == {'maximum': 3.0}


def test_refreshed_metadata_takes_effect(dataset):
    reprocess_experiments(dataset, process_raw_data,
                          metadata_retrival_function=retrieve_metadata)

    assert dataset.experiments['Exp_001'].processed_data == {'maximum': 30.0}
    assert dataset.experiments['Exp_002'].processed_data == {'maximum': 300.0}
    assert dataset.experiments['Exp_001'].metadata['scale'] == 10.0


def test_a_subset_can_be_reprocessed(dataset):
    reprocess_experiments(dataset, process_raw_data,
                          metadata_retrival_function=retrieve_metadata,
                          experiment_names=['Exp_001'])

    assert dataset.experiments['Exp_001'].processed_data == {'maximum': 30.0}
    assert dataset.experiments['Exp_002'].processed_data == {'maximum': -1.0}


def test_unknown_experiment_names_are_rejected(dataset):
    with pytest.raises(ValueError, match='Deleted_experiment'):
        reprocess_experiments(dataset, process_raw_data,
                              experiment_names=['Deleted_experiment'])


def test_a_failure_keeps_the_previous_results(dataset):
    results = reprocess_experiments(dataset, failing_processing)

    failures = [result for result in results if not result['success']]

    assert [failure['file'] for failure in failures] == [FAILING_EXPERIMENT]
    assert dataset.experiments[FAILING_EXPERIMENT].processed_data == {'maximum': -1.0}
    assert dataset.experiments['Exp_001'].processed_data == {'maximum': 3.0}


def test_progress_is_reported_for_every_experiment(dataset):
    recorder = ProgressRecorder()

    reprocess_experiments(dataset, process_raw_data, progress_callback=recorder)

    assert recorder.calls == [(0, 2, None), (1, 2, 'Exp_001'), (2, 2, 'Exp_002')]


def test_reprocessing_stamps_the_version_information(dataset):
    dataset.set_external_version({'commit': 'abc123'})

    reprocess_experiments(dataset, process_raw_data)

    assert dataset.version[LAST_PROCESSED_KEY] is not None
    assert dataset.version['external_version'] == {'commit': 'abc123'}

    experiment_version = dataset.experiments['Exp_001'].version
    assert experiment_version[LAST_PROCESSED_KEY] is not None
    assert experiment_version[PYKES_VERSION_KEY]
    # The dataset's external version is inherited without repeating it
    assert experiment_version['external_version'] == {'commit': 'abc123'}


def test_color_and_group_follow_the_refreshed_metadata(dataset):
    dataset.overview_df['color'] = ['red', 'green']
    dataset.overview_df['group'] = ['Loading', 'Loading']

    reprocess_single_experiment(dataset.experiments['Exp_001'],
                                process_raw_data,
                                metadata_retrival_function=retrieve_metadata,
                                overview_df=dataset.overview_df)

    assert dataset.experiments['Exp_001'].color == 'red'
    assert dataset.experiments['Exp_001'].group == 'Loading'
