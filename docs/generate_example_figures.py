"""Regenerate the example figures used by the README and the documentation.

Run from the repository root::

    python docs/generate_example_figures.py

Every figure is produced from the package itself, so a change in behaviour
shows up in the documentation the next time this is run. The figures are
committed, so a documentation build needs neither this script nor a display.
"""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from pyKES.database.database_experiments import Experiment
from pyKES.fitting_ODE import Fitting_Model, square_loss_time_series
from pyKES.reaction_model import Reaction_Model
from pyKES.utilities.calculate_absorption import (
    calculate_excitations_per_second_multi_competing_fast)
from pyKES.utilities.max_rate import extract_max_rate, plot_max_rate
from pyKES.utilities.unit_handler import Quantity

# --- Output ------------------------------------------------------------------

IMAGE_DIRECTORY = 'docs/_static/images'
FIGURE_DPI = 160

# --- Photophysical parameters of the illustrative network ---------------------

PHOTON_FLUX = 1e17          # photons cm^-2 s^-1
PATHLENGTH = 2.25           # cm
A_EXTINCTION_COEFFICIENT = 8500   # M^-1 cm^-1
B_EXTINCTION_COEFFICIENT = 5400   # M^-1 cm^-1
C_EXTINCTION_COEFFICIENT = 1000   # M^-1 cm^-1

# Time at which the photon budget is evaluated, well inside the phase where
# both chromophores are present in comparable amounts.
PROPAGATION_TIMEPOINT = 10.0

# --- Ru(bpy)3 / persulfate water oxidation ------------------------------------
# Rate constants fitted against measured O2 evolution; see the guide.

RU_RATE_CONSTANTS = {'k1': 9.995e-01,   # Ru(II) excitation
                     'k2': 9.886e-01,   # light-driven turnover of Ru(III)
                     'k3': 7.407e-03,   # dimerization of Ru(III)
                     'k4': 3.437e-03,   # dimer-catalysed loss of Ru(III)
                     'k5': 2.739e-02,   # H2O2 -> O2
                     'k6': 4.762e-03,   # decomposition of Ru(III)
                     'k7': 5.918e+01,   # oxidative quenching by persulfate
                     'k8': 1 / 650e-9}  # excited-state lifetime, 650 ns

RU_II_EXTINCTION_COEFFICIENT = 8500   # M^-1 cm^-1
RU_III_EXTINCTION_COEFFICIENT = 540   # M^-1 cm^-1

# --- Synthetic data used by the fitting example -------------------------------

TRUE_RATE_CONSTANTS = {'k1': 0.045, 'k2': 0.011}
FIT_BOUNDS = {'k1': (1e-3, 5e-1), 'k2': (1e-3, 5e-1)}
FIT_NOISE_FRACTION = 0.02   # relative to the initial concentration
FIT_SEED = 3


def save(figure, name):
    """
    Write a figure to the image directory and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    name : str
        File name, without directory or extension.

    Returns
    -------
    None
    """

    path = f'{IMAGE_DIRECTORY}/{name}.png'
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(figure)

    print(f'wrote {path}')


def build_photochemical_model():
    """
    Build the two-chromophore photochemical cascade used by two figures.

    [A] absorbs light and either relaxes or converts to [B]; [B] absorbs in
    turn and either relaxes, reverts to [A] or converts to the product [C].
    Both excitation steps carry a competitive-absorption multiplier, so [A] and
    [B] compete for the incident photons as their ratio changes.

    Returns
    -------
    Reaction_Model
        The solved model.
    """

    reactions = ['[A] > [A-excited], k1 ; hv_functionA',
                 '[A-excited] > [A], k2',
                 '[A-excited] > [B], k3',
                 '[B] > [A], k4',
                 '[B] > [B-excited], k5 ; hv_functionB',
                 '[B-excited] > [B], k6',
                 '[B-excited] > [C], k7']

    rate_constants = {'k1': 1, 'k2': 3e8, 'k3': 1e8, 'k4': 1e2,
                      'k5': 1, 'k6': 2e8, 'k7': 3.3e8}

    absorption_arguments = {
        'photon_flux': 'photon_flux',
        'concentration_[A]': '[A]',
        'concentration_[B]': '[B]',
        'concentration_[C]': '[C]',
        'extinction_coefficient_[A]': 'A_extinction_coefficient',
        'extinction_coefficient_[B]': 'B_extinction_coefficient',
        'extinction_coefficient_[C]': 'C_extinction_coefficient',
        'pathlength': 'pathlength'}

    other_multipliers = {
        'pathlength': PATHLENGTH,
        'photon_flux': PHOTON_FLUX,
        'A_extinction_coefficient': A_EXTINCTION_COEFFICIENT,
        'B_extinction_coefficient': B_EXTINCTION_COEFFICIENT,
        'C_extinction_coefficient': C_EXTINCTION_COEFFICIENT,
        'hv_functionA_species_of_interest': '[A]',
        'hv_functionB_species_of_interest': '[B]',
        'hv_functionA': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': absorption_arguments
                         | {'species_of_interest': 'hv_functionA_species_of_interest'}},
        'hv_functionB': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': absorption_arguments
                         | {'species_of_interest': 'hv_functionB_species_of_interest'}}}

    model = Reaction_Model(reaction_network=reactions,
                           rate_constants=rate_constants,
                           initial_conditions={'[A]': 10.0, '[B]': 0.0},
                           other_multipliers=other_multipliers,
                           times=np.linspace(0, 1000, 1000))

    model.solve_ode()

    return model


def figure_reaction_network():
    """
    Simulate Ru(bpy)3-photosensitized water oxidation and plot the traces.

    The standard persulfate-driven system: Ru(II) is excited, oxidatively
    quenched by persulfate to Ru(III), and Ru(III) either turns over to O2 or
    is lost to dimerization and decomposition. Both light-driven steps carry a
    competitive-absorption multiplier, so the strongly absorbing Ru(II) and the
    weakly absorbing Ru(III) compete for the incident photons as their ratio
    shifts during the run.

    The figure shows what makes the system worth modelling: O2 evolution slows
    down not because the substrate runs out, but because the photosensitizer is
    consumed by the two loss channels.

    Returns
    -------
    None
    """

    reactions = ['[RuII] > [RuII-ex], k1 ; hv_functionA',
                 '[RuII-ex] > [RuII], k8',
                 '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
                 '[RuIII] > [H2O2] + [RuII], k2 ; hv_functionB',
                 '2 [RuIII] > [Ru-Dimer], k3',
                 '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
                 '[H2O2] > [O2], k5',
                 '[RuIII] > [Inactive], k6']

    absorption_arguments = {
        'photon_flux': 'photon_flux',
        'concentration_[RuII]': '[RuII]',
        'concentration_[RuIII]': '[RuIII]',
        'extinction_coefficient_[RuII]': 'Ru_II_extinction_coefficient',
        'extinction_coefficient_[RuIII]': 'Ru_III_extinction_coefficient',
        'pathlength': 'pathlength'}

    other_multipliers = {
        'pathlength': PATHLENGTH,
        'photon_flux': PHOTON_FLUX,
        'Ru_II_extinction_coefficient': RU_II_EXTINCTION_COEFFICIENT,
        'Ru_III_extinction_coefficient': RU_III_EXTINCTION_COEFFICIENT,
        'hv_functionA_species_of_interest': '[RuII]',
        'hv_functionB_species_of_interest': '[RuIII]',
        'hv_functionA': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': absorption_arguments
                         | {'species_of_interest': 'hv_functionA_species_of_interest'}},
        'hv_functionB': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': absorption_arguments
                         | {'species_of_interest': 'hv_functionB_species_of_interest'}}}

    model = Reaction_Model(reaction_network=reactions,
                           rate_constants=RU_RATE_CONSTANTS,
                           initial_conditions={'[S2O8]': 6000, '[RuII]': 10},
                           other_multipliers=other_multipliers,
                           times=np.linspace(0, 300, 1000))

    model.solve_ode()

    figure, axis = plt.subplots(figsize=(7.5, 4.4))

    # The persulfate and sulfate traces are three orders of magnitude larger
    # than everything else and would flatten the catalyst speciation.
    model.plot_solution(exclude_species=['[S2O8]', '[SO4]', '[RuII-ex]'],
                        ax=axis)

    axis.set_xlabel('Time / s')
    axis.set_ylabel('Concentration / µM')
    axis.set_title('Ru(bpy)$_3$-photosensitized water oxidation')

    save(figure, 'reaction_network_simulation')


def figure_pathway_diagram(model):
    """
    Plot the photon budget of the cascade as a pathway diagram.

    Parameters
    ----------
    model : Reaction_Model
        A solved model, as returned by `build_photochemical_model`.

    Returns
    -------
    None
    """

    absorbing_species = {
        '[A]': {'excited_name': '[A-excited]',
                'extinction_coefficient': A_EXTINCTION_COEFFICIENT},
        '[B]': {'excited_name': '[B-excited]',
                'extinction_coefficient': B_EXTINCTION_COEFFICIENT},
        '[C]': {'excited_name': '[C-excited]',
                'extinction_coefficient': C_EXTINCTION_COEFFICIENT}}

    model.calculate_reaction_network_propopagation(
        timepoint=PROPAGATION_TIMEPOINT,
        absorbing_species_with_extinction_coefficients=absorbing_species,
        photon_flux=PHOTON_FLUX,
        pathlength=PATHLENGTH,
        concentration_unit='uM')

    figure, axis = plt.subplots(figsize=(8.0, 5.5))

    model.plot_reaction_network_propagation(ax=axis,
                                            value_key='log_value',
                                            forward_link_kwargs={'alpha': 0.6})

    save(figure, 'pathway_diagram')


def figure_max_rate():
    """
    Plot the max-rate diagnostic for a synthetic trace with a known rate.

    The trace carries the three disturbances the algorithm is built for: white
    noise, a slow baseline wave whose local slope rivals the true rate, and a
    bubble artifact whose instantaneous slope is far larger than either.

    Returns
    -------
    None
    """

    generator = np.random.default_rng(0)

    time_seconds = np.arange(0.0, 8000.0, 1.0)
    kinetics = 0.02 * np.clip(time_seconds - 1000.0, 0.0, None)
    baseline_wave = 3.0 * np.sin(2.0 * np.pi * time_seconds / 400.0)

    artifact = np.zeros_like(time_seconds)
    artifact[5000:5010] = np.linspace(0.0, 15.0, 10)
    artifact[5010:] = 15.0 * np.exp(
        -(time_seconds[5010:] - time_seconds[5010]) / 100.0)

    time = Quantity(time_seconds, 's')
    values = Quantity(kinetics + baseline_wave + artifact
                      + 0.2 * generator.standard_normal(len(time_seconds)), 'umol')

    result = extract_max_rate(time, values)

    figure, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    plot_max_rate(result, time, values, axes=axes)

    save(figure, 'max_rate_diagnostic')

    print(f"  max rate {result.max_rate.unit['umol / s']:.4f} umol/s "
          f"(true 0.0200), flags: {result.flags}")


def simulate_consecutive_reaction(rate_constants, initial_concentration, times):
    """
    Analytic solution of [A] → [B] → [C] for the fitting example.

    Used to manufacture the "measured" data, so that the fit has a ground truth
    it was not itself generated by.

    Parameters
    ----------
    rate_constants : dict
        ``'k1'`` and ``'k2'``, in s^-1.
    initial_concentration : float
        Concentration of [A] at t = 0, in µM.
    times : numpy.ndarray
        Time points in s.

    Returns
    -------
    numpy.ndarray
        Concentration of [B] at each time point, in µM.
    """

    k1, k2 = rate_constants['k1'], rate_constants['k2']

    return (initial_concentration * k1 / (k2 - k1)
            * (np.exp(-k1 * times) - np.exp(-k2 * times)))


def build_synthetic_experiments():
    """
    Create three synthetic experiments differing only in starting concentration.

    Returns
    -------
    list of Experiment
        Experiments carrying the noisy [B] trace under
        ``processed_data['intermediate']`` and the starting concentration under
        ``metadata['initial_concentration_uM']``.
    """

    generator = np.random.default_rng(FIT_SEED)
    times = np.linspace(0.0, 300.0, 120)

    experiments = []

    for index, (initial_concentration, color) in enumerate(
            [(10.0, '#1f4e79'), (6.0, '#c0504d'), (3.0, '#4f8a3d')]):

        clean = simulate_consecutive_reaction(TRUE_RATE_CONSTANTS,
                                              initial_concentration,
                                              times)
        noise = (FIT_NOISE_FRACTION * initial_concentration
                 * generator.standard_normal(len(times)))

        experiments.append(Experiment(
            experiment_name=f'run-{index + 1}',
            raw_data_file=f'synthetic-{index + 1}',
            color=color,
            group='synthetic',
            metadata={'initial_concentration_uM': initial_concentration},
            raw_data={},
            processed_data={'intermediate': {'x': times, 'y': clean + noise},
                            'times': times}))

    return experiments


def figure_fitting():
    """
    Fit a consecutive reaction to three synthetic experiments and plot it.

    The three runs share one reaction network and one pair of rate constants
    but start from different concentrations, which is the case the fitting
    interface is built around: the initial condition is a *path* into each
    experiment rather than a fixed number.

    Returns
    -------
    None
    """

    model = Fitting_Model(['[A] > [B], k1', '[B] > [C], k2'])

    model.experiments = build_synthetic_experiments()
    model.rate_constants_to_optimize = FIT_BOUNDS
    model.data_to_be_fitted = {'[B]': {'x': 'processed_data/intermediate/x',
                                       'y': 'processed_data/intermediate/y'}}
    model.initial_conditions = {'[A]': 'metadata/initial_concentration_uM'}
    model.times = {'times': 'processed_data/times'}
    model.loss_function = square_loss_time_series

    model.optimize(workers=1, disp=False, print_results=False)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))

    model.visualize_optimization_results(ax=axis)

    axis.set_xlabel('Time / s')
    axis.set_ylabel('[B] / µM')
    axis.set_title('Global fit of [A] → [B] → [C] across three experiments')

    save(figure, 'fitting_result')

    fitted = dict(zip(model.rate_constants_to_optimize.keys(), model.result.x))
    print(f"  fitted {fitted}, true {TRUE_RATE_CONSTANTS}")


def main():
    """
    Regenerate every example figure.

    Returns
    -------
    None
    """

    figure_reaction_network()
    figure_pathway_diagram(build_photochemical_model())
    figure_max_rate()
    figure_fitting()


if __name__ == '__main__':
    main()
