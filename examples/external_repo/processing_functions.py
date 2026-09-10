"""
Processing pipeline of the example app.

Handed to pyKES through ``FileUploadHandler.processing_function``, which is
called as ``(raw_data_dict, metadata_dict) -> processed_data_dict``. What it
returns is the experiment's ``processed_data``, and the paths in
`parameters.PLOTTING_INSTRUCTIONS` are what the pyKES pages resolve against
it.

The pipeline is the one a real photocatalysis run gets:

1. cut the irradiation window out of the recorded trace and zero its origin
   (`offset_correction`),
2. average it into fixed bins (`resample_time_series`) — the sensors log far
   finer than the chemistry, so this suppresses noise rather than thinning it,
3. turn the concentration into an amount of substance using the reactor volume,
4. extract the largest sustained rate with its uncertainty
   (`extract_max_rate`, a Gaussian-process fit that separates the kinetics
   from correlated sensor drift), and
5. turn that rate into an apparent quantum yield
   (`calculate_apparent_quantum_yield`).

Every number this produces depends on a column the dataset declares as
processing-relevant, so correcting one in the metadata editor and reprocessing
is visible on the Analysis Results and Results Table pages — which is what
makes this app a full end-to-end test of pyKES rather than a stub.
"""

import numpy as np

from pyKES.utilities.calculate_efficiency import calculate_apparent_quantum_yield
from pyKES.utilities.max_rate import extract_max_rate
from pyKES.utilities.offset_correction import offset_correction
from pyKES.utilities.time_series_resampling import resample_time_series
from pyKES.utilities.unit_handler import Quantity

from parameters import (ELECTRONS_PER_H2, ELECTRONS_PER_O2, MAX_RATE_WINDOW_S,
                        RESAMPLING_INTERVAL_S)


# Percent, for the apparent quantum yield, which the pages label as a percentage
PERCENT = 100.0

# Millilitres per litre, for turning a dissolved concentration into an amount
MILLILITRES_PER_LITRE = 1000.0


def reaction_window_amount(time_s, concentration_umol_L, volume_mL,
                           irradiation_start_s, irradiation_end_s):
    """
    Reduce one sensor trace to the amount of substance evolved under irradiation.

    Parameters
    ----------
    time_s : numpy.ndarray
        Sample times of the recorded trace.
    concentration_umol_L : numpy.ndarray
        Dissolved concentration reported by the sensor.
    volume_mL : float
        Volume of the phase the sensor sits in.
    irradiation_start_s, irradiation_end_s : float
        Bounds of the irradiation window, in the trace's own time base.

    Returns
    -------
    time_reaction_s : numpy.ndarray
        Binned times within the window, starting at zero.
    amount_umol : numpy.ndarray
        Amount of substance evolved since the start of the window.
    """

    time_reaction_s, concentration_reaction = offset_correction(
        time_s, concentration_umol_L, 0, irradiation_start_s, irradiation_end_s)

    binned_time_s, binned_concentration = resample_time_series(
        time_reaction_s, concentration_reaction, interval=RESAMPLING_INTERVAL_S)

    return binned_time_s, binned_concentration * volume_mL / MILLILITRES_PER_LITRE


def maximum_rate_results(time_reaction_s, amount_umol, prefix: str) -> dict:
    """
    Extract the maximum rate of one reduced trace.

    Parameters
    ----------
    time_reaction_s : numpy.ndarray
        Binned times of the reaction window.
    amount_umol : numpy.ndarray
        Amount of substance evolved.
    prefix : str
        Key prefix identifying the sensor, e.g. ``'H2'``.

    Returns
    -------
    dict
        The curves and scalars the pages plot and tabulate, in explicit units.

    Notes
    -----
    Stored as plain floats and arrays in named units rather than as
    `Quantity` objects: the results table understands a `Quantity` and would
    convert it, but a float survives the HDF5 round trip with no question
    about how it was serialized. The unit lives in the key and in the
    plotting instruction.
    """

    result = extract_max_rate(Quantity(time_reaction_s, 's'),
                              Quantity(amount_umol, 'umol'),
                              window=Quantity(MAX_RATE_WINDOW_S, 's'))

    return {
        f'{prefix}_time_reaction_s': time_reaction_s,
        f'{prefix}_amount_umol': amount_umol,
        f'{prefix}_amount_smoothed_umol': result.smooth.unit['umol'],
        f'{prefix}_rate_umol_h': result.rate.unit['umol / h'],
        f'{prefix}_max_rate_umol_h': result.max_rate.unit['umol / h'],
        f'{prefix}_max_rate_std_umol_h': result.max_rate_std.unit['umol / h'],
        f'{prefix}_amount_evolved_umol': float(amount_umol[-1]),
        # Quality flags the fit raises for human review; shown as a column of
        # the results table, and empty for a clean series.
        f'{prefix}_max_rate_flags': ', '.join(result.flags) if result.flags else '',
    }


def apparent_quantum_yield_percent(metadata_dict: dict,
                                   max_rate_umol_h: float,
                                   electrons_per_molecule: int) -> float:
    """
    Turn a maximum rate into an apparent quantum yield.

    Parameters
    ----------
    metadata_dict : dict
        Overview row, supplying the irradiation conditions.
    max_rate_umol_h : float
        Largest sustained rate of the product.
    electrons_per_molecule : int
        Electrons transferred per product molecule.

    Returns
    -------
    float
        Apparent quantum yield, in percent.
    """

    yield_fraction = calculate_apparent_quantum_yield(
        irradiation_wavelength=Quantity(metadata_dict['Irradiation wavelength [nm]'], 'nm'),
        irradiation_area=Quantity(metadata_dict['Irradiated area [cm2]'], 'cm2'),
        irradiance_power=Quantity(metadata_dict['Irradiance [mW/cm2]'], 'mW / cm2'),
        reaction_rate=Quantity(max_rate_umol_h, 'umol / h'),
        fraction_of_photons_reaching_inside=Quantity(
            metadata_dict['Fraction of photons reaching inside [-]'], '-'),
        electron_transfer_per_reaction=electrons_per_molecule)

    return yield_fraction.unit['-'] * PERCENT


def stoichiometric_ratio(hydrogen_umol: float, oxygen_umol: float) -> float:
    """
    Ratio of the two products, which water splitting expects to be two.

    Parameters
    ----------
    hydrogen_umol, oxygen_umol : float
        Amounts evolved over the irradiation window.

    Returns
    -------
    float
        H2 / O2, or NaN when no oxygen was evolved — a missing cell in the
        table says more than an infinity would.
    """

    if oxygen_umol == 0:
        return float('nan')

    return hydrogen_umol / oxygen_umol


def process_experiment(raw_data_dict: dict, metadata_dict: dict, dissolved: bool) -> dict:
    """
    Reduce both sensor traces of one experiment to its results.

    Parameters
    ----------
    raw_data_dict : dict
        Raw traces, as the matching reader returned them.
    metadata_dict : dict
        Overview row of the experiment.
    dissolved : bool
        True for a liquid-phase run, False for a gas-phase one. Decides which
        sensor signal is present and which reactor volume applies.

    Returns
    -------
    dict
        ``processed_data`` of the experiment.
    """

    hydrogen_signal = raw_data_dict['H2_umol_L'] if dissolved else raw_data_dict['H2_Pa']
    volume_mL = (metadata_dict['Liquid phase volume [mL]'] if dissolved
                 else metadata_dict['Gas phase volume [mL]'])

    hydrogen = maximum_rate_results(
        *reaction_window_amount(raw_data_dict['H2_time_s'], hydrogen_signal, volume_mL,
                                metadata_dict['Unisense Irradiation start [s]'],
                                metadata_dict['Unisense Irradiation end [s]']),
        prefix='H2')

    oxygen = maximum_rate_results(
        *reaction_window_amount(raw_data_dict['O2_time_s'], raw_data_dict['O2_data'], volume_mL,
                                metadata_dict['Pyroscience Irradiation start [s]'],
                                metadata_dict['Pyroscience Irradiation end [s]']),
        prefix='O2')

    processed_data = hydrogen | oxygen

    processed_data['H2_apparent_quantum_yield_percent'] = apparent_quantum_yield_percent(
        metadata_dict, processed_data['H2_max_rate_umol_h'], ELECTRONS_PER_H2)
    processed_data['O2_apparent_quantum_yield_percent'] = apparent_quantum_yield_percent(
        metadata_dict, processed_data['O2_max_rate_umol_h'], ELECTRONS_PER_O2)

    processed_data['H2_to_O2_ratio'] = stoichiometric_ratio(
        processed_data['H2_amount_evolved_umol'], processed_data['O2_amount_evolved_umol'])

    return processed_data


def process_liquid_phase(raw_data_dict: dict, metadata_dict: dict) -> dict:
    """
    Processing function of the liquid-phase pipeline.

    Parameters
    ----------
    raw_data_dict : dict
        Raw traces from `raw_data_functions.read_liquid_phase_raw_data`.
    metadata_dict : dict
        Overview row of the experiment.

    Returns
    -------
    dict
        ``processed_data`` of the experiment.
    """

    return process_experiment(raw_data_dict, metadata_dict, dissolved=True)


def process_gas_phase(raw_data_dict: dict, metadata_dict: dict) -> dict:
    """
    Processing function of the gas-phase pipeline.

    Parameters
    ----------
    raw_data_dict : dict
        Raw traces from `raw_data_functions.read_gas_phase_raw_data`.
    metadata_dict : dict
        Overview row of the experiment.

    Returns
    -------
    dict
        ``processed_data`` of the experiment.
    """

    return process_experiment(raw_data_dict, metadata_dict, dissolved=False)
