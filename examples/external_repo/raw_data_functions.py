"""
Readers for the two sensor file formats of the example data.

An external repository writes one of these per instrument it owns. They are
handed to pyKES through ``FileUploadHandler.raw_data_reading_function``, which
is called as ``(directory, metadata_dict) -> raw_data_dict``: the directory is
where the upload page staged the files the user just submitted, and the
metadata row says which files in it belong to this experiment.

Everything stays at module level, with no closures: the non-Streamlit
ingestion path runs the callables in a `ProcessPoolExecutor`, which needs them
importable by name.
"""

from pathlib import Path

import pandas as pd


# Column of the UniAmp export holding the sample times
H2_TIME_COLUMN = 'Time since start (s)'

# Sensor channels of the FireStingO2 export. The instrument reports dissolved
# oxygen on one channel and the gas phase on another, under column names that
# encode both the channel and the quantity.
O2_LIQUID_COLUMNS = {'data': 'Oxygen (µmol/L) [A Ch.2 Main]',
                     'time': ' dt (s) [A Ch.2 Main]'}
O2_GAS_COLUMNS = {'data': 'Oxygen (%O2) [A Ch.2 Main]',
                  'time': ' dt (s) [A Ch.2 Main]'}


def read_hydrogen_file(path: Path, dissolved: bool = True) -> dict:
    """
    Read a UniAmp hydrogen sensor export.

    Parameters
    ----------
    path : Path
        File to read.
    dissolved : bool, optional
        True for a liquid-phase run, where the sensor reports a dissolved
        concentration; False for a gas-phase run, where it reports a partial
        pressure.

    Returns
    -------
    dict
        ``H2_time_s`` and either ``H2_umol_L`` or ``H2_Pa``.
    """

    raw_data = pd.read_csv(path, sep=';')

    signal_column = 'Sensor 1 - H2 (μmol/L)' if dissolved else 'Sensor 1 - H2 (Pa)'
    signal_key = 'H2_umol_L' if dissolved else 'H2_Pa'

    return {'H2_time_s': raw_data[H2_TIME_COLUMN].to_numpy(),
            signal_key: raw_data[signal_column].to_numpy()}


def read_oxygen_file(path: Path, dissolved: bool = True) -> dict:
    """
    Read a FireStingO2 oxygen sensor export.

    Parameters
    ----------
    path : Path
        File to read.
    dissolved : bool, optional
        True for a liquid-phase run (µmol/L on channel 2), False for a
        gas-phase run (%O2 on channel 2).

    Returns
    -------
    dict
        ``O2_time_s`` and ``O2_data``.
    """

    # Tab-separated, Latin-1, with '#' comment banners above the table and a
    # first column of day-first timestamps.
    raw_data = pd.read_csv(path,
                           encoding='ISO8859',
                           sep='\t',
                           skip_blank_lines=True,
                           comment='#',
                           parse_dates=[0],
                           dayfirst=True)

    columns = O2_LIQUID_COLUMNS if dissolved else O2_GAS_COLUMNS

    return {'O2_time_s': raw_data[columns['time']].to_numpy(),
            'O2_data': raw_data[columns['data']].to_numpy()}


def read_liquid_phase_raw_data(directory: Path, metadata_dict: dict) -> dict:
    """
    Read both sensor files of a liquid-phase experiment.

    Parameters
    ----------
    directory : Path
        Directory the upload page staged the submitted files in.
    metadata_dict : dict
        Overview row of the experiment, naming its two files.

    Returns
    -------
    dict
        Raw traces of both sensors.
    """

    return (read_hydrogen_file(Path(directory) / metadata_dict['File name H2'], dissolved=True)
            | read_oxygen_file(Path(directory) / metadata_dict['File name O2'], dissolved=True))


def read_gas_phase_raw_data(directory: Path, metadata_dict: dict) -> dict:
    """
    Read both sensor files of a gas-phase experiment.

    Parameters
    ----------
    directory : Path
        Directory the upload page staged the submitted files in.
    metadata_dict : dict
        Overview row of the experiment, naming its two files.

    Returns
    -------
    dict
        Raw traces of both sensors.
    """

    return (read_hydrogen_file(Path(directory) / metadata_dict['File name H2'], dissolved=False)
            | read_oxygen_file(Path(directory) / metadata_dict['File name O2'], dissolved=False))
