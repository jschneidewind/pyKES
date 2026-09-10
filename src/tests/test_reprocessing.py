"""
Tests for reprocessing experiments that are already held by a dataset.

The processing callables are synthetic: `process_raw_data` multiplies the raw
signal by a factor taken from the metadata, so a changed overview sheet and a
changed processing function are both visible in the results.

``scale`` is an overview column, so the dataset owns it and the stored
metadata always mirrors the sheet — which is why reprocessing without a
metadata refresh still sees the sheet's value. ``boost`` is a key the
retrieval function invents, and the invariant leaves those alone; the two
together pin both halves of the contract.
"""

import numpy as np
import pandas as pd
import pytest

from pyKES.database.data_processing import (mark_experiments_unprocessed, reprocess_experiments,
                                            reprocess_single_experiment,
                                            select_experiments_needing_reprocessing)
from pyKES.database.database_experiments import Experiment, ExperimentalDataset
from pyKES.utilities.version_information import LAST_PROCESSED_KEY, PYKES_VERSION_KEY


FAILING_EXPERIMENT = 'Exp_002'


def retrieve_metadata(experiment_name, overview_df):
    row = overview_df.loc[overview_df['Experiment'] == experiment_name].iloc[0].to_dict()
    row['experiment_name'] = experiment_name
    row['boost'] = 2.0

    return row


def process_raw_data(raw_data_dict, metadata_dict):
    scaled = float(np.max(raw_data_dict['signal'])) * metadata_dict['scale']

    return {'maximum': scaled * metadata_dict.get('boost', 1.0)}


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


def test_stored_metadata_already_matches_the_overview_sheet(dataset):
    # The fixture asks for scale = 1.0, which contradicts the sheet's 10.0.
    # Adding the experiment resolves that in favour of the sheet, so even a
    # run without a metadata refresh works from the sheet's value.
    results = reprocess_experiments(dataset, process_raw_data)

    assert all(result['success'] for result in results)
    assert dataset.experiments['Exp_001'].metadata['scale'] == 10.0
    assert dataset.experiments['Exp_001'].processed_data == {'maximum': 30.0}
    assert dataset.metadata_divergences() == {}


def test_refreshed_metadata_keeps_the_retrieval_functions_own_keys(dataset):
    reprocess_experiments(dataset, process_raw_data,
                          metadata_retrival_function=retrieve_metadata)

    # scale comes from the sheet; boost is the retrieval function's own key
    # and is not an overview column, so the invariant does not touch it.
    assert dataset.experiments['Exp_001'].processed_data == {'maximum': 60.0}
    assert dataset.experiments['Exp_002'].processed_data == {'maximum': 600.0}
    assert dataset.experiments['Exp_001'].metadata['scale'] == 10.0
    assert dataset.experiments['Exp_001'].metadata['boost'] == 2.0
    assert dataset.metadata_divergences() == {}


def test_a_subset_can_be_reprocessed(dataset):
    reprocess_experiments(dataset, process_raw_data,
                          metadata_retrival_function=retrieve_metadata,
                          experiment_names=['Exp_001'])

    assert dataset.experiments['Exp_001'].processed_data == {'maximum': 60.0}
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
    assert dataset.experiments['Exp_001'].processed_data == {'maximum': 30.0}


def test_progress_is_reported_for_every_experiment(dataset):
    recorder = ProgressRecorder()

    reprocess_experiments(dataset, process_raw_data, progress_callback=recorder)

    assert recorder.calls == [(0, 2, None), (1, 2, 'Exp_001'), (2, 2, 'Exp_002')]


def test_reprocessing_stamps_the_version_information(dataset):
    dataset.set_external_version({'version': '0.3.0'})

    reprocess_experiments(dataset, process_raw_data)

    assert dataset.version[LAST_PROCESSED_KEY] is not None
    assert dataset.version['external_version'] == {'version': '0.3.0'}

    experiment_version = dataset.experiments['Exp_001'].version
    assert experiment_version[LAST_PROCESSED_KEY] is not None
    assert experiment_version[PYKES_VERSION_KEY]
    # The dataset's external version is inherited without repeating it
    assert experiment_version['external_version'] == {'version': '0.3.0'}


def test_successful_reprocessing_flags_the_experiments_as_processed(dataset):
    reprocess_experiments(dataset, process_raw_data)

    assert dataset.overview_df['Processed'].tolist() == ['True', 'True']


def test_a_failing_reprocessing_leaves_the_flag_down(dataset):
    # The state a metadata edit leaves behind, and the reason to reprocess
    dataset.overview_df['Processed'] = ['False', 'False']

    reprocess_experiments(dataset, failing_processing)

    # The failing experiment keeps its previous results, so its flag has to
    # keep saying that those results do not follow from the current metadata.
    assert dataset.overview_df['Processed'].tolist() == ['True', 'False']
    assert dataset.experiments[FAILING_EXPERIMENT].processed_data == {'maximum': -1.0}


def test_a_failing_reprocessing_does_not_invalidate_an_up_to_date_experiment(dataset):
    # Reprocessing everything after an algorithm change: the failure keeps the
    # previous results, which still follow from the unchanged metadata, so the
    # flag is left alone rather than cleared.
    dataset.overview_df['Processed'] = ['True', 'True']

    reprocess_experiments(dataset, failing_processing)

    assert dataset.overview_df['Processed'].tolist() == ['True', 'True']


def test_reprocessing_clears_the_flag_a_metadata_edit_set(dataset):
    """The round trip the Data Upload page drives: edit, then reprocess."""
    dataset.overview_df['Processed'] = ['True', 'True']

    mark_experiments_unprocessed(dataset, ['Exp_001'])

    assert select_experiments_needing_reprocessing(dataset) == ['Exp_001']

    reprocess_experiments(dataset, process_raw_data, experiment_names=['Exp_001'])

    assert dataset.overview_df['Processed'].tolist() == ['True', 'True']
    assert select_experiments_needing_reprocessing(dataset) == []


def test_color_and_group_follow_the_refreshed_metadata(dataset):
    dataset.overview_df['color'] = ['red', 'green']
    dataset.overview_df['group'] = ['Loading', 'Loading']

    reprocess_single_experiment(dataset.experiments['Exp_001'],
                                process_raw_data,
                                metadata_retrival_function=retrieve_metadata,
                                overview_df=dataset.overview_df)

    assert dataset.experiments['Exp_001'].color == 'red'
    assert dataset.experiments['Exp_001'].group == 'Loading'
