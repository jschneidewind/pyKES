"""Maximum theoretical solar-to-hydrogen efficiency of multi-photon water splitting.

A molecular water-splitting cycle driven by sunlight rarely runs on a single
photon. Its intermediates absorb one after the other, each up to its own
absorption edge, and every reaction event needs a fixed number of photons
from each of them. How efficient such a cycle can be at best -- and at which
absorption edges -- follows from three ingredients:

1. Photon budget: the solar spectrum, converted into a photon flux, is cut
   into consecutive bands, one per absorber. Absorber *i* harvests the photons
   between the edges of absorbers *i - 1* and *i*; the first one everything
   below its own edge. The cycle turns over only as fast as the absorber with
   the smallest flux per photon it needs allows.
2. Energy balance: every absorbed photon delivers the energy of its absorber's
   edge minus an excess energy lost to the driving forces of the steps it
   powers. Where the photons together fall short of the Gibbs energy of the
   reaction, only the fraction of events thermally activated across the
   deficit -- a two-state Boltzmann population -- goes on to react.
3. Efficiency: the solar-to-hydrogen (STH) efficiency is the Gibbs energy
   stored per second and square metre of illuminated area, divided by the
   incident solar irradiance.

`maximize_solar_to_hydrogen_efficiency` finds the absorption edges that
maximize the efficiency by differential evolution, the global search the
original analysis used (Schneidewind et al., Energy Environ. Sci. 2021,
doi:10.1039/D1EE01053K). `evaluate_solar_to_hydrogen_efficiency` evaluates
given edges, `calculate_thermal_contribution` adds the share of a reaction's
Gibbs energy supplied as heat, and the plotting functions draw the efficiency
landscape and the partition of the spectrum.

The model is a port of ``Solar_to_Hydrogen_Efficiency.py`` from
https://github.com/jschneidewind/Water-Splitting and reproduces its results:
see docs/solar_to_hydrogen_efficiency.md for a detailed explanation of the
model and of that validation.

Inputs and outputs are `Quantity` objects. Energies may be given per event
(``Quantity(4 * 1.229, 'eV')``) or per mole (``Quantity(17.5, 'kcal / mol')``).
All internal arithmetic runs on plain floats in the calculation units below.

Typical use::

    from pyKES.utilities.unit_handler import Quantity
    from pyKES.utilities.solar_to_hydrogen_efficiency import (
        SolarToHydrogenModel, maximize_solar_to_hydrogen_efficiency)

    model = SolarToHydrogenModel(gibbs_energy=Quantity(4 * 1.229, 'eV'),
                                 excess_energy=Quantity(1.0, 'eV'))
    optimum = maximize_solar_to_hydrogen_efficiency(model, number_photons=[2, 2], seed=0)
    print(optimum.efficiency.unit['%'], optimum.wavelengths.unit['nm'])
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid
from scipy.optimize import OptimizeResult, differential_evolution

from pyKES.utilities.find_nearest import find_nearest
from pyKES.utilities.unit_handler import Quantity

# The namespace also holds SciPy's `differential_evolution` and `trapezoid`,
# which are kept out of a star import and out of the generated API reference.
__all__ = [
    'read_magnitude',
    'read_energy_per_event',
    'read_number_photons',
    'read_absorbers',
    'calculate_photon_energies',
    'select_band',
    'integrate_spectrum',
    'convert_irradiance_to_photon_flux',
    'SolarSpectrum',
    'load_am15g_spectrum',
    'SolarToHydrogenModel',
    'calculate_energy_deficit',
    'calculate_thermal_population',
    'calculate_band_photon_fluxes',
    'calculate_reaction_rate',
    'convert_rate_to_efficiency',
    'calculate_efficiency',
    'negative_efficiency',
    'SolarToHydrogenResult',
    'evaluate_solar_to_hydrogen_efficiency',
    'maximize_solar_to_hydrogen_efficiency',
    'calculate_thermal_contribution',
    'calculate_efficiency_map',
    'format_scientific_latex',
    'format_band_label',
    'plot_efficiency_curve',
    'plot_efficiency_map',
    'plot_spectrum_partition',
    'test_function',
]

# --- Physical constants -------------------------------------------------------
PLANCK_CONSTANT = constants.h               # J s
SPEED_OF_LIGHT = constants.c                # m / s
AVOGADRO_CONSTANT = constants.Avogadro      # 1 / mol
BOLTZMANN_CONSTANT = constants.k            # J / K
NANOMETRES_PER_METRE = 1e9
# The electronvolt of the pyKES unit handler, 1.602176e-19 J -- truncated
# against CODATA's 1.602176634e-19 J, and the value the original model used.
# Taking it from the unit handler keeps a Quantity given in eV and the energy
# balance below on the same definition.
ELECTRON_VOLT = Quantity(1.0, 'eV').unit['J']

# --- Calculation units --------------------------------------------------------
# Quantity inputs are reduced to plain floats in these units on entry and
# results are re-wrapped on exit. The energy balance itself runs in eV.
WAVELENGTH_UNIT = 'nm'
SPECTRAL_IRRADIANCE_UNIT = 'W / m2 / nm'
IRRADIANCE_UNIT = 'W / m2'
ENERGY_UNIT = 'J'                     # per event: one photon, or one reaction
MOLAR_ENERGY_UNIT = 'J / mol'
TEMPERATURE_UNIT = 'K'
AREAL_RATE_UNIT = 'mol / (s * m2)'    # reaction events per second and square metre
EFFICIENCY_UNIT = '-'
ENERGY_PER_EVENT_DIMENSION = 'energy'
MOLAR_ENERGY_DIMENSION = 'energy / substance'

# --- Reference spectrum -------------------------------------------------------
# ASTM G173-03 reference spectra as distributed by NREL, shipped with the
# package. Columns: wavelength (nm), then extraterrestrial, global tilt and
# direct + circumsolar spectral irradiance (W m-2 nm-1); AM 1.5 G is the
# global tilt column.
AM15G_PATH = Path(__file__).resolve().parent / 'data' / 'ASTMG173.csv'
AM15G_HEADER_ROWS = 2
AM15G_WAVELENGTH_COLUMN = 0
AM15G_GLOBAL_TILT_COLUMN = 2
AM15G_NAME = 'AM 1.5 G'

# --- Energy balance -----------------------------------------------------------
STANDARD_TEMPERATURE = Quantity(298.15, 'K')  # temperature of the Boltzmann population across an energy deficit

# --- Global optimization ------------------------------------------------------
# Search range of every absorption edge, in nm. It holds every edge worth
# having: the solar spectrum carries almost no photons below 300 nm, and a
# photon at 3000 nm brings only 0.41 eV.
ABSORPTION_EDGE_BOUNDS = (300.0, 3000.0)

# --- Plotting -----------------------------------------------------------------
COLORMAP = 'inferno'
EFFICIENCY_MAP_LEVELS = 200               # colour levels of the efficiency map
# Band colours are sampled from the colormap at this many evenly spaced points
# (more if there are more bands). Band i is filled with colour i + 1 and
# labelled with colour i, one shade darker, so the first label is black.
MINIMUM_BAND_COLORS = 5
BAND_LABEL_HEIGHT = 5e18                  # photons m-2 s-1 nm-1, height of the band labels
PHOTON_FLUX_AXIS_MAXIMUM = 6.2e18         # photons m-2 s-1 nm-1, upper limit of the flux axis
# Axes fraction. The original placed this label at (0.81, 0.73) of the figure,
# which is the same point in matplotlib's default single-axes layout but lands
# in the wrong place once the axes is one panel of a larger figure.
EFFICIENCY_LABEL_POSITION = (0.884, 0.805)
FLUX_LABEL_DECIMALS = 2
PARTITION_PLOT_START = Quantity(200.0, 'nm')
PARTITION_PLOT_END = Quantity(1000.0, 'nm')


def read_magnitude(quantity, unit, name):
    """
    Read the magnitude of a `Quantity` in one of the calculation units.

    Parameters
    ----------
    quantity : Quantity
        Value to reduce to a magnitude.
    unit : str
        Target unit; a dimension mismatch raises inside the unit handler.
    name : str
        Argument name, used only in the error message.

    Returns
    -------
    float or numpy.ndarray
        The magnitude expressed in `unit`.
    """
    if not isinstance(quantity, Quantity):
        raise TypeError(f"{name} must be a Quantity compatible with '{unit}', "
                        f'got {type(quantity).__name__}')

    return quantity.unit[unit]


def read_energy_per_event(energy, name):
    """
    Read an energy per reaction event or per photon, in J.

    Reaction and loss energies are quoted both ways -- 4 x 1.229 eV per O2,
    17.5 kcal/mol per photon -- so both are accepted: an energy per event (e.g.
    eV) is read directly, a molar energy (e.g. kcal / mol) is divided by the
    Avogadro constant.

    Parameters
    ----------
    energy : Quantity
        Energy of dimension energy, or energy / substance.
    name : str
        Argument name, used only in the error message.

    Returns
    -------
    float
        Energy per event in J.
    """
    if not isinstance(energy, Quantity):
        raise TypeError(f'{name} must be a Quantity, got {type(energy).__name__}')

    if energy.dimension == ENERGY_PER_EVENT_DIMENSION:
        return energy.unit[ENERGY_UNIT]

    if energy.dimension == MOLAR_ENERGY_DIMENSION:
        return energy.unit[MOLAR_ENERGY_UNIT] / AVOGADRO_CONSTANT

    raise ValueError(f"{name} must be an energy per event (e.g. 'eV') or a molar energy "
                     f"(e.g. 'kJ / mol'), got '{energy.supplied_unit}'")


def read_number_photons(number_photons, number_absorbers):
    """
    Read the photons every absorber takes up per reaction event.

    Parameters
    ----------
    number_photons : array_like
        One positive number per absorber.
    number_absorbers : int
        Number of absorbers the photon numbers must match.

    Returns
    -------
    numpy.ndarray
        Photon numbers as floats.
    """
    number_photons = np.atleast_1d(np.asarray(number_photons, dtype=float))

    if number_photons.shape != (number_absorbers,) or np.any(number_photons <= 0):
        raise ValueError(f'number_photons must hold one positive number for each of the '
                         f'{number_absorbers} absorber(s), got {number_photons}')

    return number_photons


def read_absorbers(wavelengths, number_photons):
    """
    Read the absorption edges and photon numbers of a set of absorbers.

    Parameters
    ----------
    wavelengths : Quantity
        Absorption edge of every absorber, in order of increasing wavelength
        (dimension length).
    number_photons : array_like
        Photons every absorber takes up per reaction event.

    Returns
    -------
    wavelengths : numpy.ndarray
        Absorption edges in nm.
    number_photons : numpy.ndarray
        Photon numbers as floats, one per absorber.
    """
    wavelengths = np.atleast_1d(np.asarray(read_magnitude(wavelengths, WAVELENGTH_UNIT, 'wavelengths'),
                                           dtype=float))

    if wavelengths.ndim != 1:
        raise ValueError(f'wavelengths must be one-dimensional, got shape {wavelengths.shape}')

    return wavelengths, read_number_photons(number_photons, wavelengths.size)


def calculate_photon_energies(wavelengths):
    """
    Energy of a photon at each wavelength.

    Parameters
    ----------
    wavelengths : float or numpy.ndarray
        Wavelengths in nm.

    Returns
    -------
    float or numpy.ndarray
        Photon energies in J.
    """
    return PLANCK_CONSTANT * SPEED_OF_LIGHT / (wavelengths / NANOMETRES_PER_METRE)


def select_band(spectrum, start_wavelength, end_wavelength):
    """
    Rows of a tabulated spectrum between two wavelengths.

    Both limits snap to the nearest tabulated wavelength, and the selection is
    half-open: it runs from the row nearest `start_wavelength` up to, but not
    including, the row nearest `end_wavelength`. Limits beyond the tabulated
    range snap to its ends, and an end at or before the start selects nothing.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of shape ``(n, 2)``: wavelength in nm, ascending, and value.
    start_wavelength : float
        Start of the band, in nm.
    end_wavelength : float
        End of the band, in nm.

    Returns
    -------
    numpy.ndarray
        The selected rows, a view into `spectrum`.
    """
    start_index, end_index = find_nearest(spectrum, (start_wavelength, end_wavelength))

    return spectrum[start_index:end_index]


def integrate_spectrum(spectrum, start_wavelength, end_wavelength):
    """
    Integrate a tabulated spectrum between two wavelengths.

    The trapezoidal rule over the rows `select_band` returns. Because that
    selection is half-open, the grid interval just below `end_wavelength` is
    left out: integrating consecutive bands [a, b) and [b, c) misses one grid
    interval at b. The original implementation integrated this way, and it is
    kept so that results reproduce it; the loss is one 0.5-5 nm step of the
    ASTM G173 grid per band.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of shape ``(n, 2)``: wavelength in nm, ascending, and spectral
        density per nm.
    start_wavelength : float
        Start of the band, in nm.
    end_wavelength : float
        End of the band, in nm.

    Returns
    -------
    float
        The integral, in the unit of the spectral density times nm; zero for an
        empty band.
    """
    band = select_band(spectrum, start_wavelength, end_wavelength)

    return trapezoid(band[:, 1], band[:, 0])


def convert_irradiance_to_photon_flux(irradiance):
    """
    Convert a spectral irradiance into a spectral photon flux.

    Parameters
    ----------
    irradiance : numpy.ndarray
        Array of shape ``(n, 2)``: wavelength in nm and spectral irradiance in
        W m-2 nm-1.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2)``: wavelength in nm and spectral photon flux in
        photons m-2 s-1 nm-1.
    """
    wavelengths = irradiance[:, 0]

    return np.column_stack((wavelengths, irradiance[:, 1] / calculate_photon_energies(wavelengths)))


@dataclass
class SolarSpectrum:
    """
    Tabulated solar spectral irradiance and the photon flux it carries.

    Constructed from `Quantity` objects; the derived tables are plain arrays in
    the calculation units, which is what the model works on.

    Parameters
    ----------
    wavelengths : Quantity
        Tabulated wavelengths, strictly increasing (dimension length).
    spectral_irradiance : Quantity
        Spectral irradiance at each wavelength (dimension power / area /
        length, e.g. ``'W / m2 / nm'``).
    name : str, optional
        Label of the spectrum in figures.
    """

    wavelengths: Quantity
    spectral_irradiance: Quantity
    name: str = 'spectrum'
    #: Shape ``(n, 2)``: wavelength in nm and spectral irradiance in W m-2 nm-1.
    irradiance_table: np.ndarray = field(init=False, repr=False)
    #: Shape ``(n, 2)``: wavelength in nm and spectral photon flux in photons m-2 s-1 nm-1.
    photon_flux_table: np.ndarray = field(init=False, repr=False)
    #: Irradiance integrated over the whole table (dimension power / area), the
    #: denominator of the efficiency. Integrated like every band, so the last
    #: grid interval is left out (see `integrate_spectrum`).
    total_irradiance: Quantity = field(init=False, repr=False)

    def __post_init__(self):
        """
        Validate the table and derive the photon flux and total irradiance.

        Returns
        -------
        None
        """
        wavelengths = np.asarray(read_magnitude(self.wavelengths, WAVELENGTH_UNIT, 'wavelengths'),
                                 dtype=float)
        spectral_irradiance = np.asarray(read_magnitude(self.spectral_irradiance,
                                                        SPECTRAL_IRRADIANCE_UNIT,
                                                        'spectral_irradiance'),
                                         dtype=float)

        if wavelengths.ndim != 1 or wavelengths.shape != spectral_irradiance.shape:
            raise ValueError('wavelengths and spectral_irradiance must be one-dimensional and of '
                             f'equal length, got shapes {wavelengths.shape} and '
                             f'{spectral_irradiance.shape}')

        # Band limits are located by binary search, which needs a sorted table
        if np.any(np.diff(wavelengths) <= 0):
            raise ValueError('wavelengths must be strictly increasing')

        self.irradiance_table = np.column_stack((wavelengths, spectral_irradiance))
        self.photon_flux_table = convert_irradiance_to_photon_flux(self.irradiance_table)
        self.total_irradiance = Quantity(integrate_spectrum(self.irradiance_table,
                                                            wavelengths[0], wavelengths[-1]),
                                         IRRADIANCE_UNIT)


def load_am15g_spectrum():
    """
    Load the AM 1.5 G reference solar spectrum (ASTM G173-03, global tilt).

    Returns
    -------
    SolarSpectrum
        The spectrum, tabulated from 280 to 4000 nm.
    """
    table = np.loadtxt(AM15G_PATH, skiprows=AM15G_HEADER_ROWS, delimiter=',')

    return SolarSpectrum(Quantity(table[:, AM15G_WAVELENGTH_COLUMN], WAVELENGTH_UNIT),
                         Quantity(table[:, AM15G_GLOBAL_TILT_COLUMN], SPECTRAL_IRRADIANCE_UNIT),
                         name=AM15G_NAME)


@dataclass
class SolarToHydrogenModel:
    """
    Energetics and illumination of a multi-photon water-splitting cycle.

    The `Quantity` inputs are validated and reduced to the calculation units
    once, here, so the efficiency can be evaluated many thousands of times
    during an optimization without touching the unit handler.

    Parameters
    ----------
    gibbs_energy : Quantity
        Gibbs energy stored per reaction event, per event (e.g.
        ``Quantity(4 * 1.229, 'eV')`` for 2 H2O -> 2 H2 + O2) or molar (e.g.
        ``Quantity(474.4, 'kJ / mol')``). The reaction rate counts these
        events.
    excess_energy : Quantity
        Energy every absorbed photon loses to the driving forces of the steps
        it powers, per photon or molar.
    temperature : Quantity, optional
        Temperature of the Boltzmann population across an energy deficit;
        298.15 K by default.
    spectrum : SolarSpectrum, optional
        Illumination; AM 1.5 G by default.
    """

    gibbs_energy: Quantity
    excess_energy: Quantity
    temperature: Quantity = STANDARD_TEMPERATURE
    spectrum: SolarSpectrum = field(default_factory=load_am15g_spectrum)
    #: Gibbs energy per reaction event, in eV.
    gibbs_energy_per_event: float = field(init=False, repr=False)
    #: Gibbs energy per mole of reaction events, in J/mol.
    gibbs_energy_per_mole: float = field(init=False, repr=False)
    #: Excess energy per photon, in eV.
    excess_energy_per_photon: float = field(init=False, repr=False)
    #: Temperature in K.
    absolute_temperature: float = field(init=False, repr=False)
    #: Total irradiance of the spectrum, in W m-2.
    solar_irradiance: float = field(init=False, repr=False)

    def __post_init__(self):
        """
        Reduce the energies, temperature and irradiance to plain floats.

        Returns
        -------
        None
        """
        gibbs_energy = read_energy_per_event(self.gibbs_energy, 'gibbs_energy')

        self.gibbs_energy_per_event = gibbs_energy / ELECTRON_VOLT
        self.gibbs_energy_per_mole = gibbs_energy * AVOGADRO_CONSTANT
        self.excess_energy_per_photon = (read_energy_per_event(self.excess_energy, 'excess_energy')
                                         / ELECTRON_VOLT)
        self.absolute_temperature = read_magnitude(self.temperature, TEMPERATURE_UNIT, 'temperature')
        self.solar_irradiance = self.spectrum.total_irradiance.unit[IRRADIANCE_UNIT]


def calculate_energy_deficit(wavelengths, number_photons, gibbs_energy_per_event,
                             excess_energy_per_photon):
    """
    Energy by which the absorbed photons fall short of the reaction's Gibbs energy.

    Every photon is counted at the energy of its absorber's edge, the lowest
    energy that absorber takes up -- anything above it relaxes before it can
    do work -- minus the excess energy.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Absorption edge of every absorber, in nm.
    number_photons : numpy.ndarray
        Photons every absorber takes up per reaction event.
    gibbs_energy_per_event : float
        Gibbs energy stored per reaction event, in eV.
    excess_energy_per_photon : float
        Energy lost per absorbed photon, in eV.

    Returns
    -------
    float
        The deficit in eV; negative when the photons bring more than enough.
    """
    available_energies = calculate_photon_energies(wavelengths) / ELECTRON_VOLT - excess_energy_per_photon

    return gibbs_energy_per_event - np.sum(number_photons * available_energies)


def calculate_thermal_population(energy_deficit, temperature):
    """
    Fraction of reaction events thermally activated across an energy deficit.

    A two-state Boltzmann population, ``exp(-deficit / kT)``. A surplus needs
    no activation, so every event proceeds and the population is 1.

    Parameters
    ----------
    energy_deficit : float
        Deficit in eV, as returned by `calculate_energy_deficit`.
    temperature : float
        Temperature in K.

    Returns
    -------
    float
        Population between 0 and 1.
    """
    # Written as 1 / exp(x) rather than exp(-x) because that is how the
    # original computed it, and the two can differ in the last bit
    return 1.0 / np.exp(np.maximum(energy_deficit, 0.0) * ELECTRON_VOLT
                        / (BOLTZMANN_CONSTANT * temperature))


def calculate_band_photon_fluxes(wavelengths, photon_flux):
    """
    Photon flux every absorber harvests from its band of the spectrum.

    Absorber *i* takes up the photons between the edges of absorbers *i - 1*
    and *i*; the first absorber's band opens at the short-wavelength end of the
    spectrum. Edges out of increasing order leave an empty band, so the
    absorber behind it harvests nothing.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Absorption edge of every absorber, in nm.
    photon_flux : numpy.ndarray
        Spectral photon flux table, as `SolarSpectrum.photon_flux_table`.

    Returns
    -------
    numpy.ndarray
        Photon flux of every band, in photons m-2 s-1.
    """
    band_starts = np.concatenate(([photon_flux[0, 0]], wavelengths[:-1]))

    return np.array([integrate_spectrum(photon_flux, band_start, band_end)
                     for band_start, band_end in zip(band_starts, wavelengths)])


def calculate_reaction_rate(wavelengths, number_photons, model):
    """
    Reaction events per second and square metre of illuminated area.

    The absorber with the smallest photon flux per photon it needs is the
    bottleneck; its turnover, scaled by the thermally activated fraction of
    events, is the rate of the whole cycle.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Absorption edge of every absorber, in nm.
    number_photons : numpy.ndarray
        Photons every absorber takes up per reaction event.
    model : SolarToHydrogenModel
        Energetics and illumination.

    Returns
    -------
    float
        Reaction rate in mol m-2 s-1.
    """
    energy_deficit = calculate_energy_deficit(wavelengths, number_photons,
                                              model.gibbs_energy_per_event,
                                              model.excess_energy_per_photon)
    population = calculate_thermal_population(energy_deficit, model.absolute_temperature)
    band_photon_fluxes = calculate_band_photon_fluxes(wavelengths, model.spectrum.photon_flux_table)

    return population * np.min(band_photon_fluxes / AVOGADRO_CONSTANT / number_photons)


def convert_rate_to_efficiency(reaction_rate, molar_energy, solar_irradiance):
    """
    Fraction of the incident solar power stored by a reaction.

    Parameters
    ----------
    reaction_rate : float
        Reaction rate in mol m-2 s-1.
    molar_energy : float
        Energy stored per mole of reaction events, in J/mol.
    solar_irradiance : float
        Incident irradiance in W m-2.

    Returns
    -------
    float
        Efficiency as a fraction.
    """
    return reaction_rate * molar_energy / solar_irradiance


def calculate_efficiency(wavelengths, number_photons, model):
    """
    Solar-to-hydrogen efficiency of a set of absorbers.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Absorption edge of every absorber, in nm.
    number_photons : numpy.ndarray
        Photons every absorber takes up per reaction event.
    model : SolarToHydrogenModel
        Energetics and illumination.

    Returns
    -------
    float
        Efficiency as a fraction.
    """
    reaction_rate = calculate_reaction_rate(wavelengths, number_photons, model)

    return convert_rate_to_efficiency(reaction_rate, model.gibbs_energy_per_mole,
                                      model.solar_irradiance)


def negative_efficiency(wavelengths, number_photons, model):
    """
    Objective of the global optimization: the efficiency with its sign flipped.

    `scipy.optimize.differential_evolution` minimizes, so maximizing the
    efficiency means minimizing its negative.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Absorption edge of every absorber, in nm.
    number_photons : numpy.ndarray
        Photons every absorber takes up per reaction event.
    model : SolarToHydrogenModel
        Energetics and illumination.

    Returns
    -------
    float
        Negative efficiency, as a fraction.
    """
    return -calculate_efficiency(wavelengths, number_photons, model)


@dataclass
class SolarToHydrogenResult:
    """
    Efficiency and reaction rate of a set of absorbers.

    Returned by `evaluate_solar_to_hydrogen_efficiency` for given absorption
    edges and by `maximize_solar_to_hydrogen_efficiency` for the optimal ones.

    Parameters
    ----------
    wavelengths : Quantity
        Absorption edge of every absorber (dimension length).
    number_photons : numpy.ndarray
        Photons every absorber takes up per reaction event.
    efficiency : Quantity
        Solar-to-hydrogen efficiency (dimensionless); read it as a percentage
        with ``efficiency.unit['%']``.
    reaction_rate : Quantity
        Reaction events per second and square metre of illuminated area
        (dimension substance / (time * area)).
    optimizer_result : scipy.optimize.OptimizeResult or None
        The raw result of `scipy.optimize.differential_evolution`, kept for
        convergence diagnostics; None for a plain evaluation.
    """

    wavelengths: Quantity
    number_photons: np.ndarray
    efficiency: Quantity
    reaction_rate: Quantity
    optimizer_result: Optional[OptimizeResult] = None


def evaluate_solar_to_hydrogen_efficiency(model, wavelengths, number_photons):
    """
    Efficiency and reaction rate of a given set of absorbers.

    Parameters
    ----------
    model : SolarToHydrogenModel
        Energetics and illumination.
    wavelengths : Quantity
        Absorption edge of every absorber, in order of increasing wavelength
        (dimension length).
    number_photons : array_like
        Photons every absorber takes up per reaction event, e.g. ``[1, 1, 2]``.

    Returns
    -------
    SolarToHydrogenResult
        Efficiency and reaction rate; ``optimizer_result`` is None.
    """
    wavelengths, number_photons = read_absorbers(wavelengths, number_photons)
    reaction_rate = calculate_reaction_rate(wavelengths, number_photons, model)
    efficiency = convert_rate_to_efficiency(reaction_rate, model.gibbs_energy_per_mole,
                                            model.solar_irradiance)

    return SolarToHydrogenResult(wavelengths=Quantity(wavelengths, WAVELENGTH_UNIT),
                                 number_photons=number_photons,
                                 efficiency=Quantity(efficiency, EFFICIENCY_UNIT),
                                 reaction_rate=Quantity(reaction_rate, AREAL_RATE_UNIT))


def maximize_solar_to_hydrogen_efficiency(model, number_photons, seed=None):
    """
    Find the absorption edges that maximize the efficiency.

    Differential evolution searches every edge within `ABSORPTION_EDGE_BOUNDS`,
    with SciPy's default settings. The efficiency is piecewise constant in the
    photon fluxes -- band limits snap to the spectrum's grid -- so it has flat
    plateaus and many near-equal optima, which is why a global search is used
    rather than a local one. It also means the result depends on the random
    stream: unseeded runs scatter by a few nm in the edges (more for the last
    of four absorbers) and by up to a few tenths of a percentage point in
    efficiency, while a seeded run is reproducible.

    Parameters
    ----------
    model : SolarToHydrogenModel
        Energetics and illumination.
    number_photons : array_like
        Photons every absorber takes up per reaction event, e.g. ``[1, 1, 2]``.
        Its length sets the number of absorbers.
    seed : int, numpy.random.Generator or None, optional
        Seed of `scipy.optimize.differential_evolution`. None draws from
        NumPy's global random state, as the original analysis did.

    Returns
    -------
    SolarToHydrogenResult
        Edges, efficiency and reaction rate at the optimum, with the raw
        optimizer result in ``optimizer_result``.
    """
    number_photons = read_number_photons(number_photons, np.size(number_photons))
    bounds = np.tile(ABSORPTION_EDGE_BOUNDS, (len(number_photons), 1))

    optimizer_result = differential_evolution(negative_efficiency, bounds=bounds,
                                              args=(number_photons, model), seed=seed)

    result = evaluate_solar_to_hydrogen_efficiency(model,
                                                   Quantity(optimizer_result.x, WAVELENGTH_UNIT),
                                                   number_photons)
    result.optimizer_result = optimizer_result

    return result


def calculate_thermal_contribution(model, result, complete_gibbs_energy):
    """
    Efficiency share of the part of a reaction's Gibbs energy supplied as heat.

    When light drives only part of a reaction -- ``model.gibbs_energy`` -- and
    the rest of its Gibbs energy is supplied thermally, the complete reaction
    still runs at the light-limited rate. The thermal part then adds
    ``rate x (complete - light-driven Gibbs energy) / irradiance`` to the
    efficiency.

    Parameters
    ----------
    model : SolarToHydrogenModel
        Energetics and illumination `result` was computed with.
    result : SolarToHydrogenResult
        Absorbers and their reaction rate, typically the optimum.
    complete_gibbs_energy : Quantity
        Gibbs energy of the complete reaction per event, per event or molar.

    Returns
    -------
    Quantity
        Thermal contribution to the efficiency (dimensionless); the total
        efficiency is ``result.efficiency`` plus this.
    """
    thermal_energy = (read_energy_per_event(complete_gibbs_energy, 'complete_gibbs_energy')
                      * AVOGADRO_CONSTANT - model.gibbs_energy_per_mole)
    efficiency = convert_rate_to_efficiency(result.reaction_rate.unit[AREAL_RATE_UNIT],
                                            thermal_energy, model.solar_irradiance)

    return Quantity(efficiency, EFFICIENCY_UNIT)


def calculate_efficiency_map(model, first_wavelengths, second_wavelengths, number_photons):
    """
    Efficiency of two absorbers on a grid of absorption edges.

    Parameters
    ----------
    model : SolarToHydrogenModel
        Energetics and illumination.
    first_wavelengths : Quantity
        Edges of the first absorber (dimension length).
    second_wavelengths : Quantity
        Edges of the second absorber (dimension length).
    number_photons : array_like
        Photons each of the two absorbers takes up per reaction event.

    Returns
    -------
    Quantity
        Efficiency (dimensionless), of shape
        ``(len(second_wavelengths), len(first_wavelengths))`` as
        `numpy.meshgrid` lays it out: rows follow the second absorber.
    """
    first_grid, second_grid = np.meshgrid(read_magnitude(first_wavelengths, WAVELENGTH_UNIT,
                                                         'first_wavelengths'),
                                          read_magnitude(second_wavelengths, WAVELENGTH_UNIT,
                                                         'second_wavelengths'))
    number_photons = read_number_photons(number_photons, 2)

    efficiencies = [calculate_efficiency(np.array(edges), number_photons, model)
                    for edges in zip(first_grid.ravel(), second_grid.ravel())]

    return Quantity(np.reshape(efficiencies, first_grid.shape), EFFICIENCY_UNIT)


def format_scientific_latex(value, decimals):
    """
    Format a number in scientific notation for matplotlib's mathtext.

    Parameters
    ----------
    value : float
        Number to format.
    decimals : int
        Decimals of the mantissa.

    Returns
    -------
    str
        For example ``'3.60\\times 10^{20}'``, to be placed between ``$`` signs.
    """
    mantissa, exponent = f'{value:.{decimals}e}'.split('e')

    return rf'{mantissa}\times 10^{{{int(exponent)}}}'


def format_band_label(band_number, wavelength, band_photon_flux):
    """
    Label of one band in `plot_spectrum_partition`.

    Parameters
    ----------
    band_number : int
        Number of the band, counted from 1.
    wavelength : float
        Absorption edge closing the band, in nm.
    band_photon_flux : float
        Photon flux of the band, in photons m-2 s-1.

    Returns
    -------
    str
        Four-line label: edge and photon flux.
    """
    return (rf'$\lambda_{{{band_number}}}$ =' + '\n' + f' {wavelength:.0f} nm'
            + '\nFlux $(m^{-2} s^{-1})$ =\n'
            + f'${format_scientific_latex(band_photon_flux, FLUX_LABEL_DECIMALS)}$')


def plot_efficiency_curve(model, wavelengths, number_photons, ax=None):
    """
    Plot the efficiency of a single absorber against its absorption edge.

    Parameters
    ----------
    model : SolarToHydrogenModel
        Energetics and illumination.
    wavelengths : Quantity
        Absorption edges to evaluate (dimension length).
    number_photons : array_like
        Photons the absorber takes up per reaction event, e.g. ``[2]``.
    ax : matplotlib.axes.Axes, optional
        Axes drawn on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    wavelengths = np.asarray(read_magnitude(wavelengths, WAVELENGTH_UNIT, 'wavelengths'), dtype=float)
    number_photons = read_number_photons(number_photons, 1)

    efficiencies = [calculate_efficiency(np.array([wavelength]), number_photons, model)
                    for wavelength in wavelengths]

    ax.plot(wavelengths, efficiencies)
    ax.set_xlabel('Longest absorption wavelength / nm')
    ax.set_ylabel('STH')

    return ax


def plot_efficiency_map(model, first_wavelengths, second_wavelengths, number_photons, ax=None):
    """
    Plot the efficiency of two absorbers as a colour map of their absorption edges.

    Parameters
    ----------
    model : SolarToHydrogenModel
        Energetics and illumination.
    first_wavelengths : Quantity
        Edges of the first absorber, on the x axis (dimension length).
    second_wavelengths : Quantity
        Edges of the second absorber, on the y axis (dimension length).
    number_photons : array_like
        Photons each of the two absorbers takes up per reaction event.
    ax : matplotlib.axes.Axes, optional
        Axes drawn on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator

    if ax is None:
        _, ax = plt.subplots()

    efficiency_map = calculate_efficiency_map(model, first_wavelengths, second_wavelengths,
                                              number_photons).unit[EFFICIENCY_UNIT]
    first_grid, second_grid = np.meshgrid(first_wavelengths.unit[WAVELENGTH_UNIT],
                                          second_wavelengths.unit[WAVELENGTH_UNIT])

    levels = MaxNLocator(nbins=EFFICIENCY_MAP_LEVELS).tick_values(np.amin(efficiency_map),
                                                                 np.amax(efficiency_map))
    colormap = plt.get_cmap(COLORMAP)
    mesh = ax.pcolormesh(first_grid, second_grid, efficiency_map, cmap=colormap,
                         norm=BoundaryNorm(levels, ncolors=colormap.N, clip=True))
    mesh.set_edgecolor('face')

    ax.figure.colorbar(mesh, ax=ax).set_label('STH')
    ax.set_xlabel(r'Longest absorption wavelength [$\bf{A}$] / nm')
    ax.set_ylabel(r'Longest absorption wavelength [$\bf{B}$] / nm')

    return ax


def plot_spectrum_partition(model, result, start_wavelength=PARTITION_PLOT_START,
                            end_wavelength=PARTITION_PLOT_END, label_offsets=None, ax=None):
    """
    Plot the solar photon flux, shaded by the band every absorber harvests.

    Each band is labelled with the edge that closes it and its photon flux --
    the flux the efficiency was computed from -- and the efficiency of `result`
    is printed beside the plot.

    Parameters
    ----------
    model : SolarToHydrogenModel
        Energetics and illumination `result` was computed with.
    result : SolarToHydrogenResult
        Absorbers to draw, typically the optimum.
    start_wavelength : Quantity, optional
        Start of the plotted spectrum (dimension length).
    end_wavelength : Quantity, optional
        End of the plotted spectrum; extended to the last edge if that lies
        beyond it (dimension length).
    label_offsets : Quantity, optional
        Horizontal shift of every band label from the centre of its band
        (dimension length), for labels that would otherwise overlap.
    ax : matplotlib.axes.Axes, optional
        Axes drawn on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    photon_flux = model.spectrum.photon_flux_table
    wavelengths = np.asarray(result.wavelengths.unit[WAVELENGTH_UNIT], dtype=float)
    start = read_magnitude(start_wavelength, WAVELENGTH_UNIT, 'start_wavelength')
    end = max(read_magnitude(end_wavelength, WAVELENGTH_UNIT, 'end_wavelength'), wavelengths[-1])
    offsets = (np.zeros(len(wavelengths)) if label_offsets is None
               else read_magnitude(label_offsets, WAVELENGTH_UNIT, 'label_offsets'))

    plotted = select_band(photon_flux, start, end)
    ax.plot(plotted[:, 0], plotted[:, 1], color='black', linewidth=1.0, label=model.spectrum.name)

    edge_indices = find_nearest(photon_flux, wavelengths)
    bands = np.split(photon_flux, edge_indices)[:-1]
    band_photon_fluxes = calculate_band_photon_fluxes(wavelengths, photon_flux)
    band_starts = np.concatenate(([max(start, photon_flux[0, 0])], wavelengths[:-1]))
    label_positions = (band_starts + wavelengths) / 2 + offsets
    colors = plt.get_cmap(COLORMAP)(np.linspace(0, 1, max(MINIMUM_BAND_COLORS, len(wavelengths) + 1)))

    for band_index, (band, edge_index, wavelength, band_photon_flux, label_position) in enumerate(
            zip(bands, edge_indices, wavelengths, band_photon_fluxes, label_positions)):
        # Close every band with the first point of the next, so the fills meet without a gap
        filled = np.vstack((band, photon_flux[edge_index]))
        ax.fill_between(filled[:, 0], filled[:, 1], color=colors[band_index + 1])
        ax.annotate(format_band_label(band_index + 1, wavelength, band_photon_flux),
                    (label_position, BAND_LABEL_HEIGHT), ha='center', color=colors[band_index])

    ax.annotate('Maximum STH:\n' + r'$\bf{%.2f}$' % result.efficiency.unit['%'] + r'$\bf{\%}$',
                EFFICIENCY_LABEL_POSITION, color='black', ha='center', xycoords='axes fraction')
    ax.set_xlabel('Wavelength / nm')
    ax.set_ylabel(r'Spectral photon flux / $m^{-2} nm^{-1} s^{-1}$')
    ax.legend()
    ax.set_ylim(0, PHOTON_FLUX_AXIS_MAXIMUM)

    return ax


def test_function():
    """
    Run the analyses of the original study on AM 1.5 G and draw both figures.

    Prints the optimum for ten photon sequences of overall water splitting
    (4 x 1.229 eV per O2, 1 eV lost per photon) and for the two-photon
    H2O2 pathway (2 x 1.78 eV, 17.5 kcal/mol lost per photon), then draws the
    efficiency map of that pathway and the spectrum partition of a
    three-absorber cycle. Every optimization is seeded, so the printout is the
    same on every run.

    Note that pyKES defines the kilocalorie as 4184 J, the original analysis
    as 4186.798188 J; efficiencies computed with a loss given in kcal/mol
    therefore differ slightly from the published ones.

    Returns
    -------
    None
        Prints the optima and shows the two figures.
    """
    import matplotlib.pyplot as plt

    water_splitting = SolarToHydrogenModel(gibbs_energy=Quantity(4 * 1.229, 'eV'),
                                           excess_energy=Quantity(1.0, 'eV'))
    photon_sequences = [[2], [4], [1, 1], [2, 2], [4, 4], [1, 1, 1], [2, 2, 2], [1, 1, 2],
                        [1, 1, 1, 1], [2, 2, 2, 2]]

    for number_photons in photon_sequences:
        optimum = maximize_solar_to_hydrogen_efficiency(water_splitting, number_photons, seed=0)
        print(f'photon sequence {number_photons}: '
              f"edges {np.round(optimum.wavelengths.unit['nm'])} nm, "
              f"STH {optimum.efficiency.unit['%']:.2f} %")

    hydrogen_peroxide = SolarToHydrogenModel(gibbs_energy=Quantity(2 * 1.78, 'eV'),
                                             excess_energy=Quantity(17.5, 'kcal / mol'))
    optimum = maximize_solar_to_hydrogen_efficiency(hydrogen_peroxide, [1, 1], seed=0)
    print(f"H2O2 pathway: edges {np.round(optimum.wavelengths.unit['nm'])} nm, "
          f"STH {optimum.efficiency.unit['%']:.2f} %")

    ax = plot_efficiency_map(hydrogen_peroxide, Quantity(np.linspace(300.0, 500.0, 100), 'nm'),
                             Quantity(np.linspace(300.0, 1000.0, 100), 'nm'), [1, 1])
    ax.annotate(f"Maximum STH: {optimum.efficiency.unit['%']:.2f}%\n"
                + ', '.join(f'{edge:.0f} nm' for edge in optimum.wavelengths.unit['nm']),
                (0.6, 0.8), color='white', ha='center', xycoords='axes fraction')

    three_absorbers = SolarToHydrogenModel(gibbs_energy=Quantity(4 * 1.229, 'eV'),
                                           excess_energy=Quantity(17.5, 'kcal / mol'))
    plot_spectrum_partition(three_absorbers,
                            maximize_solar_to_hydrogen_efficiency(three_absorbers, [1, 1, 2], seed=0))
    plt.show()


if __name__ == '__main__':
    test_function()
