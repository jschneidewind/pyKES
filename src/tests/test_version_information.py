"""
Tests for the dataset version information and its HDF5 round trip.

The timestamps are only checked for their ordering semantics (``created`` is
preserved, ``last_processed`` is set on processing), never against wall-clock
values, so the tests do not depend on how fast they run.
"""

import numpy as np
import pandas as pd
import pytest

from pyKES.database.database_experiments import (
    SCHEMA_VERSION,
    Experiment,
    ExperimentalDataset,
)
from pyKES.utilities.version_information import (
    CREATED_KEY,
    EXTERNAL_VERSION_KEY,
    LAST_MODIFIED_KEY,
    LAST_PROCESSED_KEY,
    PYKES_VERSION_KEY,
    SCHEMA_VERSION_KEY,
    build_version_information,
    describe_version_information,
    get_project_version,
    get_pykes_version,
    stamp_version_information,
)


def make_dataset():
    dataset = ExperimentalDataset(
        overview_df=pd.DataFrame({'Experiment': ['Exp_001'], 'scale': [2.0]})
    )
    dataset.add_experiment(Experiment(
        experiment_name='Exp_001',
        raw_data_file='exp_001.csv',
        color='blue',
        group='Intensity',
        metadata={'experiment_name': 'Exp_001', 'scale': 1.0},
        raw_data={'signal': np.array([1.0, 2.0, 3.0])},
        processed_data={'maximum': 3.0},
    ))

    return dataset


def test_fresh_version_information_is_complete():
    version = build_version_information('1.1', external_version={'version': '0.3.0'})

    assert version[PYKES_VERSION_KEY] == get_pykes_version()
    assert version[SCHEMA_VERSION_KEY] == '1.1'
    assert version[CREATED_KEY] == version[LAST_MODIFIED_KEY]
    assert version[LAST_PROCESSED_KEY] is None
    assert version[EXTERNAL_VERSION_KEY] == {'version': '0.3.0'}


def test_stamping_preserves_creation_and_records_processing():
    version = build_version_information('1.1')
    created = version[CREATED_KEY]

    stamp_version_information(version, '1.1', processed=True,
                              external_version={'version': '0.4.0'})

    assert version[CREATED_KEY] == created
    assert version[LAST_PROCESSED_KEY] == version[LAST_MODIFIED_KEY]
    assert version[EXTERNAL_VERSION_KEY] == {'version': '0.4.0'}


def test_stamping_fills_in_keys_of_a_legacy_version_dict():
    # A dict written before `last_processed` existed must not raise
    version = stamp_version_information({PYKES_VERSION_KEY: '0.1.0'}, '1.1')

    assert set(version) == {PYKES_VERSION_KEY, SCHEMA_VERSION_KEY, CREATED_KEY,
                            LAST_MODIFIED_KEY, LAST_PROCESSED_KEY, EXTERNAL_VERSION_KEY}


def test_version_survives_the_hdf5_round_trip(tmp_path):
    dataset = make_dataset()
    dataset.set_external_version({'app': 'demo', 'version': '0.3.0'})

    filename = str(tmp_path / 'dataset.h5')
    dataset.save_to_hdf5(filename)
    loaded = ExperimentalDataset.load_from_hdf5(filename)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.version[SCHEMA_VERSION_KEY] == SCHEMA_VERSION
    assert loaded.version[PYKES_VERSION_KEY] == get_pykes_version()
    assert loaded.version[EXTERNAL_VERSION_KEY] == {'app': 'demo', 'version': '0.3.0'}
    assert loaded.version[CREATED_KEY] == dataset.version[CREATED_KEY]


def test_saving_preserves_the_creation_timestamp(tmp_path):
    dataset = make_dataset()

    first_file = str(tmp_path / 'first.h5')
    dataset.save_to_hdf5(first_file)
    created = dataset.version[CREATED_KEY]

    reloaded = ExperimentalDataset.load_from_hdf5(first_file)
    reloaded.save_to_hdf5(str(tmp_path / 'second.h5'))

    assert reloaded.version[CREATED_KEY] == created


def test_experiment_version_round_trips(tmp_path):
    dataset = make_dataset()
    dataset.experiments['Exp_001'].version = {PYKES_VERSION_KEY: '9.9.9',
                                              LAST_PROCESSED_KEY: '2026-08-31T12:00:00+02:00'}

    filename = str(tmp_path / 'dataset.h5')
    dataset.save_to_hdf5(filename)
    loaded = ExperimentalDataset.load_from_hdf5(filename)

    assert loaded.experiments['Exp_001'].version[PYKES_VERSION_KEY] == '9.9.9'


def test_integer_keys_in_nested_dicts_are_saved(tmp_path):
    dataset = make_dataset()
    dataset.experiments['Exp_001'].metadata['nested'] = {0: {'value': 7}, 1: 'one'}

    filename = str(tmp_path / 'dataset_with_integer_keys.h5')
    dataset.save_to_hdf5(filename)
    loaded = ExperimentalDataset.load_from_hdf5(filename)

    assert loaded.experiments['Exp_001'].metadata['nested']['0']['value'] == 7
    assert loaded.experiments['Exp_001'].metadata['nested']['1'] == 'one'


def test_dataset_without_version_information_is_described():
    assert 'no version information' in describe_version_information({})


def test_project_version_is_read_from_the_nearest_pyproject(tmp_path):
    project_directory = tmp_path / 'external_app'
    package_directory = project_directory / 'app' / 'processing'
    package_directory.mkdir(parents=True)
    (project_directory / 'pyproject.toml').write_text(
        '[project]\nname = "external_app"\nversion = "2.5.1"\n', encoding='utf-8')
    processing_module = package_directory / 'functions.py'
    processing_module.write_text('', encoding='utf-8')

    assert get_project_version(str(processing_module)) == '2.5.1'


def test_project_version_of_pykes_itself_matches_the_package_version():
    # This test file lives inside the pyKES repository, so the upward search
    # must land on its pyproject.toml rather than on some parent project.
    assert get_project_version(__file__) == get_pykes_version()


def test_project_version_is_none_outside_a_python_project(tmp_path):
    # tmp_path is outside any checkout, so the search reaches the root
    assert get_project_version(str(tmp_path)) is None
