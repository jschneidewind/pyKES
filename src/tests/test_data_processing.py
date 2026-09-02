"""
Tests for the single-threaded ingestion pipeline: its progress reporting, the
per-experiment steps the Streamlit page drives one rerun at a time, and the
equivalence of the two.

The processing callables are synthetic: `retrieve_metadata` fails deliberately
for one experiment so the progress sequence can be checked across both the
success and the failure branch of the loop.
"""

import pandas as pd

from pyKES.database.data_processing import (ingest_experiment,
                                            read_in_experiments_single_threaded,
                                            select_unprocessed_experiments)
from pyKES.database.database_experiments import ExperimentalDataset


FAILING_EXPERIMENT = 'Exp_002'


def retrieve_metadata(file_name, overview_df):
    if file_name == FAILING_EXPERIMENT:
        raise ValueError("metadata unavailable")

    return {'experiment_name': file_name}


def read_raw_data(directory, metadata_dict):
    return {'signal': [1.0, 2.0, 3.0]}


def process_raw_data(raw_data_dict, metadata_dict):
    return {'maximum': max(raw_data_dict['signal'])}


class ProgressRecorder:
    """Records every `(completed, total, experiment_name)` triple it receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, completed, total, experiment_name):
        self.calls.append((completed, total, experiment_name))


def make_dataset(experiment_names):
    return ExperimentalDataset(
        overview_df = pd.DataFrame({'Experiment': experiment_names})
    )


def test_progress_callback_reports_every_experiment():
    experiment_names = ['Exp_001', FAILING_EXPERIMENT, 'Exp_003']
    dataset = make_dataset(experiment_names)
    recorder = ProgressRecorder()

    results = read_in_experiments_single_threaded(
        database = dataset,
        metadata_retrival_function = retrieve_metadata,
        raw_data_reading_function = read_raw_data,
        processing_function = process_raw_data,
        progress_callback = recorder,
    )

    assert recorder.calls == [
        (0, 3, None),
        (1, 3, 'Exp_001'),
        (2, 3, FAILING_EXPERIMENT),
        (3, 3, 'Exp_003'),
    ]

    # The failing experiment must still advance the bar without being ingested.
    assert [result['success'] for result in results] == [True, False, True]
    assert set(dataset.experiments) == {'Exp_001', 'Exp_003'}


def test_progress_callback_reports_zero_total_when_nothing_to_process():
    dataset = make_dataset(['Exp_001'])
    read_in_experiments_single_threaded(
        database = dataset,
        metadata_retrival_function = retrieve_metadata,
        raw_data_reading_function = read_raw_data,
        processing_function = process_raw_data,
    )

    recorder = ProgressRecorder()
    results = read_in_experiments_single_threaded(
        database = dataset,
        metadata_retrival_function = retrieve_metadata,
        raw_data_reading_function = read_raw_data,
        processing_function = process_raw_data,
        progress_callback = recorder,
    )

    assert results == []
    assert recorder.calls == [(0, 0, None)]


def test_select_unprocessed_experiments_seeds_the_flag_as_text():
    dataset = make_dataset(['Exp_001', 'Exp_003'])

    experiment_names = select_unprocessed_experiments(dataset)

    assert experiment_names == ['Exp_001', 'Exp_003']

    # A bool column would make pandas reject the 'True' written on success.
    assert dataset.overview_df['Processed'].tolist() == ['False', 'False']


def test_select_unprocessed_experiments_skips_processed_experiments():
    dataset = make_dataset(['Exp_001', FAILING_EXPERIMENT, 'Exp_003'])

    read_in_experiments_single_threaded(
        database = dataset,
        metadata_retrival_function = retrieve_metadata,
        raw_data_reading_function = read_raw_data,
        processing_function = process_raw_data,
    )

    # The experiment that failed was never flagged, so it stays on the list.
    assert select_unprocessed_experiments(dataset) == [FAILING_EXPERIMENT]


def test_stepping_experiment_by_experiment_matches_the_loop():
    """
    The Streamlit page drives `ingest_experiment` one rerun at a time instead
    of calling the loop function; both must leave the same dataset behind.
    """
    experiment_names = ['Exp_001', FAILING_EXPERIMENT, 'Exp_003']

    looped_dataset = make_dataset(experiment_names)
    looped_results = read_in_experiments_single_threaded(
        database = looped_dataset,
        metadata_retrival_function = retrieve_metadata,
        raw_data_reading_function = read_raw_data,
        processing_function = process_raw_data,
    )

    stepped_dataset = make_dataset(experiment_names)
    stepped_results = [
        ingest_experiment(
            experiment_name = experiment_name,
            database = stepped_dataset,
            metadata_retrival_function = retrieve_metadata,
            raw_data_reading_function = read_raw_data,
            processing_function = process_raw_data,
        )
        for experiment_name in select_unprocessed_experiments(stepped_dataset)
    ]

    assert ([result['success'] for result in stepped_results]
            == [result['success'] for result in looped_results])
    assert set(stepped_dataset.experiments) == set(looped_dataset.experiments)
    assert (stepped_dataset.overview_df['Processed'].tolist()
            == looped_dataset.overview_df['Processed'].tolist())


def test_ingest_experiment_seeds_the_flag_on_its_own():
    """
    The flag must be text whichever entry point creates the column.

    Seeding it with a bool gives the column bool dtype, which pandas then
    refuses to write 'True' into once the first experiment succeeds — the
    `TypeError: Invalid value 'True' for dtype 'bool'` fixed in 0.2.0. Stepping
    through experiments reaches the flag without going through the loop
    function, so the guarantee is checked here too.
    """
    dataset = make_dataset(['Exp_001', 'Exp_003'])

    ingest_experiment(
        experiment_name = 'Exp_001',
        database = dataset,
        metadata_retrival_function = retrieve_metadata,
        raw_data_reading_function = read_raw_data,
        processing_function = process_raw_data,
    )

    assert dataset.overview_df['Processed'].tolist() == ['True', 'False']


def test_ingest_experiment_leaves_the_dataset_alone_on_failure():
    dataset = make_dataset([FAILING_EXPERIMENT])

    result = ingest_experiment(
        experiment_name = FAILING_EXPERIMENT,
        database = dataset,
        metadata_retrival_function = retrieve_metadata,
        raw_data_reading_function = read_raw_data,
        processing_function = process_raw_data,
    )

    assert result['success'] is False
    assert dataset.experiments == {}
    assert dataset.overview_df['Processed'].tolist() == ['False']
