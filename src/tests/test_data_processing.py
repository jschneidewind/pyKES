"""
Tests for the progress reporting of the single-threaded ingestion pipeline.

The processing callables are synthetic: `retrieve_metadata` fails deliberately
for one experiment so the progress sequence can be checked across both the
success and the failure branch of the loop.
"""

import pandas as pd

from pyKES.database.data_processing import read_in_experiments_single_threaded
from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.streamlit_app.components.data_upload_component import _update_ingestion_progress


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


class PlaceholderSpy:
    """Stand-in for an `st.empty()` placeholder."""

    def __init__(self):
        self.progress_calls = []
        self.info_calls = []

    def progress(self, value, text):
        self.progress_calls.append((value, text))

    def info(self, body):
        self.info_calls.append(body)


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


def test_ingestion_progress_renders_fraction_and_name():
    placeholder = PlaceholderSpy()

    _update_ingestion_progress(placeholder, 0, 4, None)
    _update_ingestion_progress(placeholder, 1, 4, 'Exp_001')

    fractions = [value for value, _ in placeholder.progress_calls]
    assert fractions == [0.0, 0.25]
    assert '4' in placeholder.progress_calls[0][1]
    assert 'Exp_001' in placeholder.progress_calls[1][1]


def test_ingestion_progress_handles_empty_workload():
    placeholder = PlaceholderSpy()

    _update_ingestion_progress(placeholder, 0, 0, None)

    assert placeholder.progress_calls == []
    assert placeholder.info_calls == ["No new experiments to process"]
