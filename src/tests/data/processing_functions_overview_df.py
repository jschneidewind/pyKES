from tests.data.raw_file_reading_functions import reading_H2_file, reading_O2_file

from pathlib import Path

import numpy as np


def metadata_retrival_function(experiment_name, overview_df):
    '''
    Given an experiment name and an overview DataFrame, retrieves the metadata for the specified experiment.
    Returns a dictionary containing the metadata.


    '''

    experiment_row = overview_df[overview_df['Experiment'] == experiment_name]
    
    if experiment_row.empty:
        raise ValueError(f"No experiment found with name: {experiment_name}")
    
    if len(experiment_row) > 1:
        raise ValueError(f"Multiple experiments found with name: {experiment_name}")
    
    metadata_dict = experiment_row.iloc[0].to_dict()
    metadata_dict['experiment_name'] = metadata_dict['Experiment']

    return metadata_dict

def raw_data_reading_function(directory: Path, metadata_dict):
    '''
    '''

    file_H2 = directory / metadata_dict['File name H2']
    file_O2 = directory / metadata_dict['File name O2']

    if 'Gas phase' in metadata_dict['group']:
        raw_data_H2 = reading_H2_file(file_H2, mode = 'gas')
        raw_data_O2 = reading_O2_file(file_O2, channel = 4)

    else:
        raw_data_H2 = reading_H2_file(file_H2, mode = 'liquid')
        raw_data_O2 = reading_O2_file(file_O2, channel = 2)

    return raw_data_H2 | raw_data_O2

def produced_during_irradiation(time_s, signal, irradiation_start_s, irradiation_end_s):
    '''
    Amount a sensor signal rose while the lamp was on.

    Parameters
    ----------
    time_s : numpy.ndarray
        Sample times of the trace.
    signal : numpy.ndarray
        Sensor readings, in whatever unit the sensor reports.
    irradiation_start_s, irradiation_end_s : float
        Bounds of the irradiation window.

    Returns
    -------
    float
        Difference between the largest and the smallest reading inside the
        window.
    '''

    irradiated = (time_s >= irradiation_start_s) & (time_s <= irradiation_end_s)

    return float(np.max(signal[irradiated]) - np.min(signal[irradiated]))


def baseline_noise(time_s, signal, irradiation_start_s):
    '''
    Scatter of a sensor trace before the lamp was switched on.

    Parameters
    ----------
    time_s : numpy.ndarray
        Sample times of the trace.
    signal : numpy.ndarray
        Sensor readings.
    irradiation_start_s : float
        Start of the irradiation window; everything before it is baseline.

    Returns
    -------
    float
        Standard deviation of the baseline, used here as the uncertainty of
        a difference read off the same trace.
    '''

    return float(np.std(signal[time_s < irradiation_start_s]))


def processing_function(raw_data_dict, metadata_dict):
    '''
    Derive two illustrative results per experiment.

    Deliberately crude: a peak-to-trough difference over the irradiation
    window, with the pre-irradiation scatter as its uncertainty. The point is
    that the example app produces numbers the Analysis Results and Results
    Table pages can show, and that both depend on metadata the dataset
    declares as processing-relevant — so editing a cell in the metadata editor
    and reprocessing is visible on those pages. The real machinery for rates
    is `pyKES.utilities.max_rate`.

    Parameters
    ----------
    raw_data_dict : dict
        Raw traces as returned by `raw_data_reading_function`.
    metadata_dict : dict
        Overview row of the experiment.

    Returns
    -------
    dict
        Processed data of the experiment.
    '''

    # Liquid-phase runs report dissolved H2 in µmol/L, gas-phase runs the
    # partial pressure in Pa; which one is present follows from 'group'.
    h2_signal_key = 'H2_umol_L' if 'H2_umol_L' in raw_data_dict else 'H2_Pa'

    return {
        'H2_produced': produced_during_irradiation(
            raw_data_dict['H2_time_s'], raw_data_dict[h2_signal_key],
            metadata_dict['Unisense Irradiation start [s]'],
            metadata_dict['Unisense Irradiation end [s]']),
        'H2_baseline_noise': baseline_noise(
            raw_data_dict['H2_time_s'], raw_data_dict[h2_signal_key],
            metadata_dict['Unisense Irradiation start [s]']),
        'O2_produced': produced_during_irradiation(
            raw_data_dict['O2_time_s'], raw_data_dict['O2_data'],
            metadata_dict['Pyroscience Irradiation start [s]'],
            metadata_dict['Pyroscience Irradiation end [s]']),
    }


