"""Tests for pyKES.utilities.solar_to_hydrogen_efficiency.

Two kinds of check.

Ground truth: synthetic spectra with the same photon flux at every
wavelength, on which every band integral is the flux times the band width, so
photon budgets, bottlenecks and the optimal absorption edges all have closed
forms.

Reproduction: values computed by the original implementation,
``Data_Analysis/Solar_to_Hydrogen_Efficiency.py`` of
https://github.com/jschneidewind/Water-Splitting, pinned here so that the port
keeps reproducing it on AM 1.5 G, and optima from the tables the study
published with it. The original defined the kilocalorie as 4186.798188 J where
pyKES uses 4184 J, so its kcal/mol energies are written below in J/mol with
the original's factor; with that, the port reproduces the deterministic values
to within 1e-13 and seeded optimizations bit for bit.
"""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pyKES.utilities.solar_to_hydrogen_efficiency import (  # noqa: E402
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELECTRON_VOLT, PLANCK_CONSTANT, SPEED_OF_LIGHT,
    SolarSpectrum, SolarToHydrogenModel, calculate_band_photon_fluxes, calculate_efficiency,
    calculate_efficiency_map, calculate_energy_deficit, calculate_photon_energies,
    calculate_thermal_contribution, calculate_thermal_population, evaluate_solar_to_hydrogen_efficiency,
    format_scientific_latex, integrate_spectrum, load_am15g_spectrum,
    maximize_solar_to_hydrogen_efficiency, plot_efficiency_curve, plot_efficiency_map,
    plot_spectrum_partition, read_energy_per_event, select_band)
from pyKES.utilities.unit_handler import Quantity  # noqa: E402

ORIGINAL_KILOCALORIE = 4186.798188  # J, the kilocalorie of the original implementation
FLAT_PHOTON_FLUX = 1e18             # photons m-2 s-1 nm-1


def original_kilocalories_per_mole(value):
    """An energy the original gave in kcal/mol, as the same number of joules per mole."""
    return Quantity(value * ORIGINAL_KILOCALORIE, 'J / mol')


# Energetics of the original analyses: Gibbs energy per reaction event and
# energy lost per photon
WATER_SPLITTING_1_EV = (Quantity(4 * 1.229, 'eV'), Quantity(1.0, 'eV'))
WATER_SPLITTING_17_5_KCAL = (Quantity(4 * 1.229, 'eV'), original_kilocalories_per_mole(17.5))
PHOTOCHEMICAL_82_7_KCAL = (original_kilocalories_per_mole(82.7), original_kilocalories_per_mole(17.5))
HYDROGEN_PEROXIDE = (Quantity(2 * 1.78, 'eV'), original_kilocalories_per_mole(17.5))

# (energetics, photons per absorber, absorption edges in nm, efficiency, reaction
# rate in mol m-2 s-1), computed by the original implementation. The edges
# straddle the energy threshold: some deficits are bridged thermally, some are
# surpluses, and the last row has its edges out of order.
ORIGINAL_EVALUATIONS = [
    (WATER_SPLITTING_1_EV, [2.0], [358.4], 0.012150738368922779, 2.562566114054036e-05),
    (WATER_SPLITTING_1_EV, [4.0], [556.3], 0.11821961633864339, 0.0002493227766476335),
    (WATER_SPLITTING_1_EV, [4.0], [560.0], 0.012265490621595083, 2.5867671317437524e-05),
    (WATER_SPLITTING_1_EV, [2.0, 2.0], [509.4, 613.2], 0.1445662324067222, 0.0003048872563575491),
    (WATER_SPLITTING_1_EV, [2.0, 2.0, 2.0, 2.0], [566.1, 716.4, 880.6, 1099.2],
     0.2381405798048353, 0.0005022336599311977),
    (WATER_SPLITTING_17_5_KCAL, [1.0, 1.0, 2.0], [491.6, 583.2, 748.6],
     0.2895904576069531, 0.000610740410241234),
    (WATER_SPLITTING_17_5_KCAL, [4.0, 4.0], [731.9, 1171.5], 0.2772216946628978, 0.0005846549396872223),
    (WATER_SPLITTING_17_5_KCAL, [1.0, 1.0, 1.0, 1.0], [500.2, 596.3, 686.1, 781.4],
     0.29032823863373514, 0.000612296375485008),
    (PHOTOCHEMICAL_82_7_KCAL, [1.0, 1.0], [455.30270576, 516.52728081],
     0.13270077322632273, 0.0003833816517181029),
    (HYDROGEN_PEROXIDE, [1.0, 1.0], [455.30270576, 516.52728081], 0.131642778041886, 0.0003833816517181029),
    (HYDROGEN_PEROXIDE, [1.0, 1.0], [470.0, 540.0], 0.0005495774989912702, 1.6005278257143546e-06),
    (HYDROGEN_PEROXIDE, [1.0, 1.0], [540.0, 470.0], 0.0, 0.0),
]

# (energetics, photons per absorber, optimal edges in nm, maximum efficiency)
# found by the original's differential evolution with NumPy's global seed 0.
# Only optima that nearly every seed agrees on are pinned, so that a SciPy
# release which changes the random path cannot fail them spuriously.
ORIGINAL_OPTIMA = [
    (HYDROGEN_PEROXIDE, [1.0, 1.0], [457.5888441062216, 522.943323445101], 0.13772817991171696),
    (WATER_SPLITTING_1_EV, [4.0], [555.7028977266343], 0.12329578782249785),
    (WATER_SPLITTING_1_EV, [2.0, 2.0], [508.53709125883825, 613.6384688450285], 0.17091764096632556),
    (WATER_SPLITTING_17_5_KCAL, [1.0, 1.0, 2.0], [491.6231459107139, 583.1694710493587, 748.7790098190311],
     0.2895904576069531),
]

# (photons per absorber, optimal edges in nm, maximum efficiency) for overall
# water splitting with a 1 eV loss, as published in ESI Tables 11.5-1 to 11.5-4
# of Schneidewind et al., Energy Environ. Sci. 2021, 14, 4427 -- the entries on
# which 20 seeded runs agree to within a percent.
PUBLISHED_OPTIMA = [
    ([4.0], [556.0], 0.1233),
    ([1.0, 1.0], [348.0, 369.0], 0.0166),
    ([2.0, 2.0], [509.0, 613.0], 0.1709),
    ([1.0, 1.0, 1.0], [428.0, 475.0, 515.0], 0.1179),
    ([1.0, 1.0, 2.0], [460.0, 526.0, 642.0], 0.1958),
    ([1.0, 1.0, 1.0, 1.0], [466.0, 536.0, 600.0, 661.0], 0.2108),
]


@pytest.fixture(scope='module')
def am15g():
    return load_am15g_spectrum()


def flat_spectrum(step):
    """Spectrum from 280 to 4000 nm whose photon flux is FLAT_PHOTON_FLUX at every wavelength."""
    wavelengths = np.arange(280.0, 4000.0 + step / 2, step)
    irradiance = FLAT_PHOTON_FLUX * calculate_photon_energies(wavelengths)
    return SolarSpectrum(Quantity(wavelengths, 'nm'), Quantity(irradiance, 'W / m2 / nm'), name='flat')


def threshold_wavelength(energy):
    """Wavelength in nm of a photon of the given energy in eV."""
    return PLANCK_CONSTANT * SPEED_OF_LIGHT / (energy * ELECTRON_VOLT) * 1e9


def efficiency_of_band_flux(band_flux, model):
    """Efficiency of a cycle limited by a flux of band_flux photons m-2 s-1 per reaction event."""
    return band_flux / AVOGADRO_CONSTANT * model.gibbs_energy_per_mole / model.solar_irradiance


# --- Spectrum handling ----------------------------------------------------------

def test_photon_energy_of_green_light():
    assert calculate_photon_energies(500.0) == pytest.approx(3.972891714e-19, rel=1e-9)


def test_select_band_is_half_open_and_snaps_to_the_grid():
    table = np.column_stack((np.arange(0.0, 10.0), np.zeros(10)))
    assert select_band(table, 2.0, 5.0)[:, 0].tolist() == [2.0, 3.0, 4.0]
    assert select_band(table, 2.4, 4.6)[:, 0].tolist() == [2.0, 3.0, 4.0]
    assert select_band(table, -5.0, 100.0)[:, 0].tolist() == list(np.arange(0.0, 9.0))
    assert len(select_band(table, 6.0, 3.0)) == 0


def test_integral_of_a_linear_spectrum_is_exact():
    wavelengths = np.arange(0.0, 10.5, 0.5)
    table = np.column_stack((wavelengths, 2.0 * wavelengths + 1.0))
    # The band ends one grid step before 6.0, and the trapezoidal rule is exact
    # for a straight line: (5.5^2 + 5.5) - (2^2 + 2)
    assert integrate_spectrum(table, 2.0, 6.0) == pytest.approx(29.75, rel=1e-12)
    assert integrate_spectrum(table, 6.0, 2.0) == 0.0


def test_photon_flux_of_a_flat_spectrum_is_flat():
    spectrum = flat_spectrum(1.0)
    np.testing.assert_allclose(spectrum.photon_flux_table[:, 1], FLAT_PHOTON_FLUX, rtol=1e-12)


def test_am15g_total_irradiance(am15g):
    # ASTM G173 global tilt integrates to about 1000.4 W m-2; the original
    # implementation left out the last grid interval and got this value
    assert am15g.total_irradiance.unit['W / m2'] == pytest.approx(1000.3348698234422, rel=1e-13)
    assert am15g.name == 'AM 1.5 G'


def test_spectrum_rejects_unsorted_wavelengths():
    with pytest.raises(ValueError, match='strictly increasing'):
        SolarSpectrum(Quantity(np.array([300.0, 500.0, 400.0]), 'nm'),
                      Quantity(np.ones(3), 'W / m2 / nm'))


def test_spectrum_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match='equal length'):
        SolarSpectrum(Quantity(np.array([300.0, 400.0]), 'nm'), Quantity(np.ones(3), 'W / m2 / nm'))


# --- Energy balance --------------------------------------------------------------

def test_energy_deficit_counts_every_photon_at_its_edge():
    edges = np.array([threshold_wavelength(2.0), threshold_wavelength(1.5)])
    deficit = calculate_energy_deficit(edges, np.array([1.0, 2.0]), gibbs_energy_per_event=4.0,
                                       excess_energy_per_photon=0.25)
    # 4.0 - (1 * (2.0 - 0.25) + 2 * (1.5 - 0.25))
    assert deficit == pytest.approx(-0.25, abs=1e-12)


def test_thermal_population_is_boltzmann_for_a_deficit_and_one_for_a_surplus():
    thermal_energy = BOLTZMANN_CONSTANT * 298.15 / ELECTRON_VOLT
    assert calculate_thermal_population(thermal_energy, 298.15) == pytest.approx(np.exp(-1.0), rel=1e-12)
    assert calculate_thermal_population(3.0 * thermal_energy, 298.15) == pytest.approx(np.exp(-3.0), rel=1e-12)
    assert calculate_thermal_population(-0.3, 298.15) == 1.0
    assert calculate_thermal_population(0.0, 298.15) == 1.0


# --- Photon budget ---------------------------------------------------------------

def test_band_fluxes_are_flux_times_width():
    spectrum = flat_spectrum(1.0)
    fluxes = calculate_band_photon_fluxes(np.array([500.0, 700.0]), spectrum.photon_flux_table)
    # Each band stops one grid step short of its closing edge
    np.testing.assert_allclose(fluxes, FLAT_PHOTON_FLUX * np.array([499.0 - 280.0, 699.0 - 500.0]), rtol=1e-12)


def test_bottleneck_absorber_sets_the_rate():
    spectrum = flat_spectrum(1.0)
    model = SolarToHydrogenModel(Quantity(0.1, 'eV'), Quantity(0.0, 'eV'), spectrum=spectrum)
    result = evaluate_solar_to_hydrogen_efficiency(model, Quantity(np.array([500.0, 600.0]), 'nm'), [1, 2])
    # Band 1 holds 219 nm of flux for one photon, band 2 99 nm for two: band 2 limits
    bottleneck = FLAT_PHOTON_FLUX * (599.0 - 500.0) / 2.0
    assert result.reaction_rate.unit['mol / (s * m2)'] == pytest.approx(bottleneck / AVOGADRO_CONSTANT, rel=1e-12)
    assert result.efficiency.unit['-'] == pytest.approx(efficiency_of_band_flux(bottleneck, model), rel=1e-12)


def test_edges_out_of_order_give_zero_efficiency():
    spectrum = flat_spectrum(1.0)
    model = SolarToHydrogenModel(Quantity(0.1, 'eV'), Quantity(0.0, 'eV'), spectrum=spectrum)
    result = evaluate_solar_to_hydrogen_efficiency(model, Quantity(np.array([600.0, 500.0]), 'nm'), [1, 1])
    assert result.efficiency.unit['-'] == 0.0


# --- Optimization against closed-form optima ---------------------------------------

@pytest.mark.parametrize('step', [0.5, 1.0, 5.0])
@pytest.mark.parametrize('seed', [0, 1])
def test_single_absorber_optimum_is_its_energy_threshold(step, seed):
    # 2 eV per event and 0.5 eV lost per photon: one photon needs 2.5 eV. Below
    # that energy the population collapses within a few kT, while the flux only
    # grows by one grid step, so the optimum is the grid cell holding the threshold
    spectrum = flat_spectrum(step)
    model = SolarToHydrogenModel(Quantity(2.0, 'eV'), Quantity(0.5, 'eV'), spectrum=spectrum)
    threshold = threshold_wavelength(2.5)
    grid = spectrum.photon_flux_table[:, 0]
    nearest_grid_point = grid[np.argmin(np.abs(grid - threshold))]
    expected = efficiency_of_band_flux(FLAT_PHOTON_FLUX * (nearest_grid_point - step - 280.0), model)

    optimum = maximize_solar_to_hydrogen_efficiency(model, [1], seed=seed)

    assert optimum.efficiency.unit['-'] == pytest.approx(expected, rel=1e-12)
    assert threshold - step <= optimum.wavelengths.unit['nm'][0] <= threshold


def test_two_absorbers_balance_their_bands():
    # A surplus at every edge, so only the photon budget matters: the second
    # edge goes to the end of the search range and the bands split the flux in
    # the ratio of the photons they need, 1 : 2. On the 1 nm grid the best split
    # gives the first band 906 nm of flux.
    spectrum = flat_spectrum(1.0)
    model = SolarToHydrogenModel(Quantity(0.1, 'eV'), Quantity(0.0, 'eV'), spectrum=spectrum)
    best_possible = efficiency_of_band_flux(FLAT_PHOTON_FLUX * 906.0, model)

    optimum = maximize_solar_to_hydrogen_efficiency(model, [1, 2], seed=0)

    efficiency = optimum.efficiency.unit['-']
    assert 0.995 * best_possible <= efficiency <= best_possible * (1 + 1e-12)
    assert optimum.wavelengths.unit['nm'] == pytest.approx([1187.0, 3000.0], abs=5.0)


def test_seeded_optimization_is_reproducible():
    model = SolarToHydrogenModel(*WATER_SPLITTING_1_EV)
    first = maximize_solar_to_hydrogen_efficiency(model, [1, 1], seed=3)
    second = maximize_solar_to_hydrogen_efficiency(model, [1, 1], seed=3)
    np.testing.assert_array_equal(first.wavelengths.unit['nm'], second.wavelengths.unit['nm'])
    assert first.efficiency.unit['-'] == second.efficiency.unit['-']


def test_optimum_is_the_efficiency_of_its_edges():
    model = SolarToHydrogenModel(*HYDROGEN_PEROXIDE)
    optimum = maximize_solar_to_hydrogen_efficiency(model, [1, 1], seed=0)
    evaluated = evaluate_solar_to_hydrogen_efficiency(model, optimum.wavelengths, [1, 1])
    assert optimum.efficiency.unit['-'] == evaluated.efficiency.unit['-'] == -optimum.optimizer_result.fun
    assert optimum.reaction_rate.unit['mol / (s * m2)'] == evaluated.reaction_rate.unit['mol / (s * m2)']
    assert evaluated.optimizer_result is None


# --- Reproduction of the original implementation -----------------------------------

@pytest.mark.parametrize('energetics, number_photons, wavelengths, efficiency, reaction_rate',
                         ORIGINAL_EVALUATIONS)
def test_reproduces_original_evaluations(am15g, energetics, number_photons, wavelengths, efficiency,
                                         reaction_rate):
    model = SolarToHydrogenModel(*energetics, spectrum=am15g)
    result = evaluate_solar_to_hydrogen_efficiency(model, Quantity(np.array(wavelengths), 'nm'), number_photons)
    assert result.efficiency.unit['-'] == pytest.approx(efficiency, rel=1e-12, abs=0.0)
    assert result.reaction_rate.unit['mol / (s * m2)'] == pytest.approx(reaction_rate, rel=1e-12, abs=0.0)


@pytest.mark.parametrize('energetics, number_photons, wavelengths, efficiency', ORIGINAL_OPTIMA)
def test_reproduces_original_optima(am15g, energetics, number_photons, wavelengths, efficiency):
    # With the same seed the optimizer retraces the original's path exactly. The
    # tolerances only leave room for a SciPy release that changes that path:
    # unseeded, the original scatters by a few nm and tenths of a percent.
    model = SolarToHydrogenModel(*energetics, spectrum=am15g)
    optimum = maximize_solar_to_hydrogen_efficiency(model, number_photons, seed=0)
    assert optimum.efficiency.unit['-'] == pytest.approx(efficiency, rel=1e-2)
    assert optimum.wavelengths.unit['nm'] == pytest.approx(wavelengths, abs=3.0)


@pytest.mark.parametrize('number_photons, wavelengths, efficiency', PUBLISHED_OPTIMA)
def test_reproduces_published_optima(am15g, number_photons, wavelengths, efficiency):
    # The published values are rounded to 0.01 % and 1 nm and come from
    # unseeded runs, so they are matched to within the optimizer's scatter
    model = SolarToHydrogenModel(*WATER_SPLITTING_1_EV, spectrum=am15g)
    optimum = maximize_solar_to_hydrogen_efficiency(model, number_photons, seed=0)
    assert optimum.efficiency.unit['-'] == pytest.approx(efficiency, rel=1e-2)
    assert optimum.wavelengths.unit['nm'] == pytest.approx(wavelengths, abs=5.0)


def test_reproduces_original_thermal_contribution(am15g):
    model = SolarToHydrogenModel(*PHOTOCHEMICAL_82_7_KCAL, spectrum=am15g)
    result = evaluate_solar_to_hydrogen_efficiency(model, Quantity(np.array([455.30270576, 516.52728081]), 'nm'),
                                                   [1, 1])
    thermal = calculate_thermal_contribution(model, result, original_kilocalories_per_mole(117.11))
    assert thermal.unit['-'] == pytest.approx(0.05521443297119427, rel=1e-12)


def test_reproduces_original_efficiency_map_maximum(am15g):
    model = SolarToHydrogenModel(*PHOTOCHEMICAL_82_7_KCAL, spectrum=am15g)
    efficiency_map = calculate_efficiency_map(model, Quantity(np.linspace(300.0, 500.0, 100), 'nm'),
                                              Quantity(np.linspace(300.0, 1000.0, 100), 'nm'), [1, 1]).unit['-']
    assert efficiency_map.shape == (100, 100)
    assert np.unravel_index(np.argmax(efficiency_map), efficiency_map.shape) == (31, 77)
    assert np.max(efficiency_map) == pytest.approx(0.1347101241437586, rel=1e-12)
    assert np.count_nonzero(efficiency_map) == 8521


# --- Thermal contribution ----------------------------------------------------------

def test_thermal_contribution_ground_truth():
    spectrum = flat_spectrum(1.0)
    model = SolarToHydrogenModel(Quantity(1.0, 'eV'), Quantity(0.0, 'eV'), spectrum=spectrum)
    result = evaluate_solar_to_hydrogen_efficiency(model, Quantity(np.array([600.0]), 'nm'), [1])
    # Doubling the Gibbs energy with heat doubles the stored energy at the same rate
    assert calculate_thermal_contribution(model, result, Quantity(2.0, 'eV')).unit['-'] \
        == pytest.approx(result.efficiency.unit['-'], rel=1e-12)
    assert calculate_thermal_contribution(model, result, Quantity(1.0, 'eV')).unit['-'] == 0.0


# --- Quantities --------------------------------------------------------------------

def test_energies_per_event_and_per_mole_are_interchangeable(am15g):
    per_event = SolarToHydrogenModel(Quantity(4 * 1.229, 'eV'), Quantity(0.75, 'eV'), spectrum=am15g)
    molar = SolarToHydrogenModel(Quantity(4 * 1.229 * ELECTRON_VOLT * AVOGADRO_CONSTANT / 1e3, 'kJ / mol'),
                                 Quantity(0.75 * ELECTRON_VOLT * AVOGADRO_CONSTANT / 4184.0, 'kcal / mol'),
                                 spectrum=am15g)
    edges = np.array([480.0, 560.0, 700.0])
    number_photons = np.array([1.0, 1.0, 2.0])
    assert calculate_efficiency(edges, number_photons, molar) \
        == pytest.approx(calculate_efficiency(edges, number_photons, per_event), rel=1e-12)


def test_wavelength_and_temperature_units_are_converted(am15g):
    kelvin = SolarToHydrogenModel(*WATER_SPLITTING_1_EV, spectrum=am15g)
    celsius = SolarToHydrogenModel(*WATER_SPLITTING_1_EV, temperature=Quantity(25.0, 'degC'), spectrum=am15g)
    in_nanometres = evaluate_solar_to_hydrogen_efficiency(kelvin, Quantity(np.array([556.3]), 'nm'), [4])
    in_micrometres = evaluate_solar_to_hydrogen_efficiency(celsius, Quantity(np.array([0.5563]), 'um'), [4])
    assert in_micrometres.efficiency.unit['%'] == pytest.approx(in_nanometres.efficiency.unit['%'], rel=1e-9)


def test_energy_must_be_a_quantity_of_energy():
    with pytest.raises(TypeError, match='must be a Quantity'):
        read_energy_per_event(1.0, 'gibbs_energy')
    with pytest.raises(ValueError, match='energy per event'):
        read_energy_per_event(Quantity(500.0, 'nm'), 'gibbs_energy')


def test_absorbers_must_match_their_photon_numbers(am15g):
    model = SolarToHydrogenModel(*WATER_SPLITTING_1_EV, spectrum=am15g)
    with pytest.raises(ValueError, match='one positive number'):
        evaluate_solar_to_hydrogen_efficiency(model, Quantity(np.array([500.0, 600.0]), 'nm'), [1, 1, 1])
    with pytest.raises(ValueError, match='one positive number'):
        maximize_solar_to_hydrogen_efficiency(model, [1, 0])
    with pytest.raises(TypeError, match='must be a Quantity'):
        evaluate_solar_to_hydrogen_efficiency(model, np.array([500.0]), [1])


# --- Plotting ------------------------------------------------------------------------

def test_scientific_notation_matches_the_original_labels():
    assert format_scientific_latex(3.6012e20, 2) == r'3.60\times 10^{20}'
    assert format_scientific_latex(1.5e-3, 1) == r'1.5\times 10^{-3}'


def test_spectrum_partition_labels_the_band_fluxes(am15g):
    model = SolarToHydrogenModel(*WATER_SPLITTING_17_5_KCAL, spectrum=am15g)
    result = evaluate_solar_to_hydrogen_efficiency(model, Quantity(np.array([491.6, 583.2, 748.6]), 'nm'),
                                                   [1, 1, 2])
    ax = plot_spectrum_partition(model, result, label_offsets=Quantity(np.array([0.0, -10.0, 10.0]), 'nm'))

    fluxes = calculate_band_photon_fluxes(np.array([491.6, 583.2, 748.6]), am15g.photon_flux_table)
    band_labels = [text.get_text() for text in ax.texts[:-1]]
    assert len(ax.collections) == 3
    assert all(f'{edge:.0f} nm' in label and format_scientific_latex(flux, 2) in label
               for edge, flux, label in zip([491.6, 583.2, 748.6], fluxes, band_labels))
    assert ax.texts[0].get_position()[0] == pytest.approx((280.0 + 491.6) / 2)
    assert ax.texts[1].get_position()[0] == pytest.approx((491.6 + 583.2) / 2 - 10.0)
    assert '28.96' in ax.texts[-1].get_text()
    plt.close('all')


def test_efficiency_plots_draw_the_computed_values():
    spectrum = flat_spectrum(5.0)
    model = SolarToHydrogenModel(Quantity(2.0, 'eV'), Quantity(0.5, 'eV'), spectrum=spectrum)
    first, second = Quantity(np.linspace(300.0, 500.0, 6), 'nm'), Quantity(np.linspace(400.0, 900.0, 5), 'nm')

    map_axes = plot_efficiency_map(model, first, second, [1, 1])
    np.testing.assert_array_equal(map_axes.collections[0].get_array(),
                                  calculate_efficiency_map(model, first, second, [1, 1]).unit['-'])

    curve_axes = plot_efficiency_curve(model, first, [2])
    expected = [calculate_efficiency(np.array([edge]), np.array([2.0]), model) for edge in first.unit['nm']]
    np.testing.assert_array_equal(curve_axes.lines[0].get_ydata(), expected)
    plt.close('all')
