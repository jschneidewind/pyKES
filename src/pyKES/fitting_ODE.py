"""Fit the rate constants of a reaction network to experimental data.

`Fitting_Model` holds a reaction network whose rate constants are unknown,
together with the experiments it should reproduce, and searches for the
constants that minimize the disagreement. `objective_function` is the bridge
between the two: for a trial set of rate constants it integrates the network
once per experiment and accumulates the loss.

Two things make the fit work across a whole dataset rather than one curve:

* **Per-experiment conditions are resolved from the experiments themselves.**
  Initial conditions, multipliers and time grids are given as *paths* into an
  `pyKES.database.database_experiments.Experiment` (for example
  ``'experiment_metadata.ru_concentration_uM'``), resolved by
  `pyKES.utilities.resolve_attributes.resolve_experiment_attributes`. One model
  definition therefore covers experiments run at different concentrations,
  light intensities and durations.
* **Experiments can be weighted.** An entry of ``experiments`` may be a bare
  experiment or an ``(experiment, weight)`` tuple, so a noisy or atypical run
  can be down-weighted instead of dropped.

The loss functions at the top of the module select *what* is compared. Fitting
the derivative (`square_loss_ydiff`) rather than the integrated trace is often
the better choice for evolved-gas measurements, where a constant offset in the
accumulated amount says nothing about the kinetics.
"""

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pprint import pprint

from pyKES.reaction_ODE import solve_ode_system, parse_reactions, calculate_excitations_per_second_competing
from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.utilities.resolve_attributes import resolve_experiment_attributes
from pyKES.utilities.make_json_serializable import make_json_serializable

# The module namespace also holds SciPy's `minimize`, `differential_evolution`
# and `dual_annealing`, whose names are generic enough to be worth keeping out
# of a star import — and out of the generated API reference.
__all__ = [
    'square_loss_time_series',
    'square_loss_time_series_normalized',
    'square_loss_max_rate_ydiff',
    'square_loss_ydiff',
    'Fitting_Model',
    'objective_function',
    'test_function',
]


def square_loss_time_series(model_data, experimental_data, **kwargs):
    '''
    Calculate the square loss between model data and experimental data for time series fitting.

    Parameters
    ----------
    model_data : array-like
        The model data to be compared against experimental data.
    experimental_data : dict   
        Dictionary containing experimental data with keys 'x' and 'y'.
    kwargs : dict, optional
        Additional keyword arguments (not used in this function).

    Returns
    -------
    float
        The sum of squared differences between model data and experimental data.
    model_data : array-like
        The model data used for the calculation.
    '''

    return np.sum((np.array(model_data) - np.array(experimental_data['y'])) ** 2), model_data

def square_loss_time_series_normalized(model_data, experimental_data, **kwargs):
    '''
    Calculate the normalized square loss between model data and experimental data for time series fitting.

    Parameters
    ----------
    model_data : array-like
        The model data to be compared against experimental data.
    experimental_data : dict   
        Dictionary containing experimental data with keys 'x' and 'y'.
    kwargs : dict, optional
        Additional keyword arguments (not used in this function).

    Returns
    -------
    float
        The normalized sum of squared differences between model data and experimental data.
    model_data : array-like
        The model data used for the calculation.
    '''

    raw_square_loss, model_data = square_loss_time_series(model_data, experimental_data, **kwargs)

    length = len(experimental_data['y'])
    average_magnitude = np.mean(np.abs(experimental_data['y']))

    normalized_error = raw_square_loss / (length * (average_magnitude ** 2 + 1e-12)) # Divide by mean squared magnitude for correct units (since raw_square_loss is squared)

    return normalized_error, model_data

def square_loss_max_rate_ydiff(model_data, experimental_data, times, **kwargs):
    '''
    Calculate the square loss between the maximum rate of change of model data and experimental data.
    
    Parameters
    ----------
    model_data : array-like
        The model data to be compared against experimental data.
    experimental_data : dict  
        Dictionary containing experimental data with keys 'y'.
    times : array-like
        Time points corresponding to the model data.
    kwargs : dict, optional
        Additional keyword arguments (not used in this function).

    Returns
    -------
    float
        The sum of squared differences between model data and experimental data.
    model_data_max_rate_ydiff : array-like
        The model data used for the calculation.
    '''

    model_data_ydiff = np.diff(model_data) / np.diff(times)

    model_data_max_rate_ydiff = np.amax(model_data_ydiff) 

    return (model_data_max_rate_ydiff - experimental_data['y']) ** 2, model_data_max_rate_ydiff

def square_loss_ydiff(model_data, experimental_data, times, **kwargs):
    '''
    Calculate the square loss between the rate of change of model data and experimental data.
    Parameters
    ----------
    model_data : array-like
        The model data to be compared against experimental data.
    experimental_data : dict
        Dictionary containing experimental data with keys 'y'.
    times : array-like
        Time points corresponding to the model data.
    kwargs : dict, optional
        Additional keyword arguments (not used in this function).

    Returns
    -------
    float
        The sum of squared differences between model data and experimental data.
    model_data_ydiff : array-like
        The model data used for the calculation.     
    '''

    model_data_ydiff = np.diff(model_data) / np.diff(times)

    return np.sum((model_data_ydiff - experimental_data['y']) ** 2), model_data_ydiff

class Fitting_Model:
    """
    A model for fitting kinetic reaction networks to experimental data using optimization algorithms.
    
    This class handles the setup and optimization of kinetic models by defining reaction networks,
    rate constants, experimental conditions, and loss functions. It supports multiple optimization
    methods including differential evolution, dual annealing, and local minimization.
    
    Parameters
    ----------
    reaction_network : list
        List of reaction strings defining the kinetic model. Each reaction should be in the format:
        '[Reactants] > [Products], rate_constant ; multipliers'
        Example: ['[RuII] + [S2O8] > [RuIII] + [SO4], k1 ; hv_functionA']
    
    Attributes
    ----------
    reaction_network : list
        The input reaction network
    fixed_rate_constants : dict
        Dictionary of rate constants with fixed values (not optimized)
    rate_constants_to_optimize : dict
        Dictionary mapping rate constant names to optimization bounds (min, max)
    data_to_be_fitted : dict
        Dictionary specifying which experimental data to fit for each species
    initial_conditions : dict
        Dictionary mapping species to initial concentration attribute paths
    other_multipliers : dict
        Dictionary of additional parameters (e.g., light intensity, concentrations)
    times : dict
        Dictionary specifying time points for ODE integration
    experiments : list
        List of experiment objects or (experiment, weight) tuples for fitting
    loss_function : callable
        Function to calculate loss between model and experimental data
    x0 : array-like, optional
        Initial guess for optimization parameters
    parsed_reactions : list
        Parsed reaction network (set during initialization)
    species : list
        List of all species in the reaction network (set during initialization)
    result : OptimizeResult
        Optimization result object (set after calling optimization methods)
    
    Examples
    --------
    >>> reactions = ['[RuII] + [S2O8] > [RuIII] + [SO4], k1 ; hv_functionA',
    ...              '[RuIII] > [H2O2] + [RuII], k2 ; hv_function_B']
    >>> model = Fitting_Model(reactions)
    >>> model.rate_constants_to_optimize = {'k1': (0.1, 1.0), 'k2': (0.1, 1.0)}
    >>> model.data_to_be_fitted = {'[O2]': {'x': 'time_series_data.x_diff',
    ...                                     'y': 'time_series_data.y_diff'}}
    >>> model.optimize()
    """

    def __init__(self, reaction_network: list, **kwargs):
        """
        Initialize the fitting model and parse its reaction network.

        Parameters
        ----------
        reaction_network : list of str
            Reactions in the string format of
            `pyKES.reaction_ODE.parse_reactions`.
        **kwargs
            ``fixed_rate_constants``, ``rate_constants_to_optimize``,
            ``data_to_be_fitted``, ``initial_conditions``,
            ``other_multipliers``, ``times``, ``experiments``,
            ``loss_function`` and ``x0``, as documented on the class. Each
            defaults to an empty container or None, so a model is normally
            built by assigning to the attributes after construction.

        Returns
        -------
        None
        """

        self.reaction_network = reaction_network
        self.fixed_rate_constants: dict = kwargs.get('fixed_rate_constants', {})
        self.rate_constants_to_optimize: dict = kwargs.get('rate_constants_to_optimize', {})
        self.data_to_be_fitted: dict = kwargs.get('data_to_be_fitted', {})
        self.initial_conditions: dict = kwargs.get('initial_conditions', {})
        self.other_multipliers: dict = kwargs.get('other_multipliers', {})
        self.times: dict = kwargs.get('times', {})
        self.experiments: list = kwargs.get('experiments', [])
        self.loss_function = kwargs.get('loss_function', None)
        self.x0 = kwargs.get('x0', None)

        self.parsed_reactions, self.species = parse_reactions(self.reaction_network)

    def optimize(self, 
                 workers = -1, 
                 disp = True, 
                 print_results = True):
        """
        Fit the rate constants by differential evolution.

        The default optimizer of this module. Rate constants of a
        photochemical network span many orders of magnitude and the loss
        surface has many local minima, so a global search is needed; a local
        method such as `minimize` is best used afterwards to polish the
        result.

        Parameters
        ----------
        workers : int, optional
            Processes used for the population, as in
            `scipy.optimize.differential_evolution`. ``-1`` uses every
            available core. Requires the loss function and every multiplier
            function to be importable at module level, since the population is
            evaluated in subprocesses.
        disp : bool, optional
            Print the best loss of every generation while the fit runs.
        print_results : bool, optional
            Print the optimizer result together with the fitted and fixed rate
            constants when the fit has finished.

        Returns
        -------
        None : None
            The result is stored as ``self.result``; the fitted values are in
            ``self.result.x``, in the order of ``rate_constants_to_optimize``.

        Notes
        -----
        Updating is deferred rather than immediate, which is what allows the
        population to be evaluated in parallel.
        """

        bounds = list(self.rate_constants_to_optimize.values())

        self.result = differential_evolution(
            objective_function,
            bounds = bounds,
            args = (self,),
            workers = workers,
            disp = disp,
            updating = 'deferred',
            x0 = self.x0)

        if print_results:
            print(self.result)
            print('----------------------------')
            print('Optimized rate constants:')
            pprint(dict(zip(self.rate_constants_to_optimize.keys(), self.result.x)))
            print('Fixed rate constants:')
            pprint(self.fixed_rate_constants)

    def optimize_dual_annealing(self):
        """
        Fit the rate constants by dual annealing.

        An alternative global optimizer, single-process and often better at
        escaping a narrow local minimum than differential evolution, at the
        cost of not using multiple cores.

        Returns
        -------
        None : None
            The result is stored as ``self.result`` and printed.
        """

        bounds = list(self.rate_constants_to_optimize.values())

        self.result = dual_annealing(
            objective_function,
            bounds= bounds,
            args = (self,)
        )

        print(self.result)

    def minimize(self, x0, method = 'L-BFGS-B'):
        """
        Refine the rate constants by local minimization.

        Intended as a polishing step after a global search, started from the
        values that search returned. Note that the bounds in
        ``rate_constants_to_optimize`` are *not* applied here, so a local run
        may leave the range the global search was confined to.

        Parameters
        ----------
        x0 : array_like
            Starting values, in the order of ``rate_constants_to_optimize``.
        method : str, optional
            Any method accepted by `scipy.optimize.minimize`.

        Returns
        -------
        None : None
            The result is stored as ``self.result`` and printed.
        """

        self.result = minimize(
            objective_function,
            method = method,
            x0 = x0,
            args = (self,)
            )

        print(self.result)

    def visualize_optimization_results(self, ax = None):
        """
        Plot fitted curves against the experimental data they were fitted to.

        Every experiment is drawn in its own color, the measurement as points
        and the model as a line, so systematic misfits stand out per experiment
        rather than being hidden in the total loss.

        Requires one of the optimization methods to have been run.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes drawn on. A new figure is created when omitted.

        Returns
        -------
        None
        """

        error, model_results = objective_function(self.result.x, self, return_full = True)

        if ax is None:
            fig, ax = plt.subplots()

        for experiment in self.experiments:
            if isinstance(experiment, tuple): # If experiment is a tuple, unpack it
                experiment, weight = experiment
            
            # Get experimental data
            experimental_data = resolve_experiment_attributes(
                                    self.data_to_be_fitted, 
                                    experiment,
                                    mode = 'semi-strict')
            
            for species, data in experimental_data.items():
                model_data = model_results[experiment.experiment_name][species]
                
                ax.scatter(data['x'], data['y'], color = experiment.color,
                           s = 2)
                ax.plot(data['x'], model_data, color = experiment.color)
                ax.plot([], [], color = experiment.color, 
                        marker='o', linestyle='-', 
                        label=f'{species} - {experiment.experiment_name}')
                
        ax.legend()

    def add_fit_results_to_database(self, parent_database: ExperimentalDataset):
        """
        Add fit results and parameters to the database object.
        
        This method:
        1. Adds fit results to each experiment's processed_data dict in the database
        2. Adds Fitting_Model parameters to database.processing_parameters
        
        The fit results are stored in processed_data as:
        - f'{species}_experimental': {'x': ..., 'y': ...}
        - f'{species}_fit': {'x': ..., 'y': ...}
        
        Raises
        ------
        AttributeError
            If optimization has not been run (self.result does not exist).
        
        Examples
        --------
        >>> database = ExperimentalDataset.load_from_hdf5('data.h5')
        >>> model = Fitting_Model(reactions)
        >>> model.optimize()
        >>> model.add_fit_results_to_database(parent_datase = database)
        >>> model.database.save_to_hdf5('fit_results.h5')
        """
        if not hasattr(self, 'result'):
            raise AttributeError(
                "Optimization has not been run yet. Please call optimize(), "
                "optimize_dual_annealing(), or minimize() first."
            )

        error, model_results = objective_function(self.result.x, self, return_full = True)

        experiment_list = []
        experiment_weights = {}    

        for experiment in self.experiments:
            if isinstance(experiment, tuple): # If experiment is a tuple, unpack it
                experiment, weight = experiment
            else:
                weight = 1.0

            exp_name = experiment.experiment_name

            experiment_list.append(exp_name)
            experiment_weights[exp_name] = weight
            
            # Get experimental data
            experimental_data = resolve_experiment_attributes(
                                    self.data_to_be_fitted, 
                                    experiment,
                                    mode = 'semi-strict')
            
            for species, data in experimental_data.items():
                model_data = model_results[exp_name][species]

                parent_database.experiments[exp_name].processed_data[f'{species}_experimental'] = {
                    'x': data['x'],
                    'y': data['y']}
                
                parent_database.experiments[exp_name].processed_data[f'{species}_fit'] = {
                    'x': data['x'],
                    'y': model_data}

        # Get optimized rate constants as dict
        optimized_rate_constants = dict(
            zip(self.rate_constants_to_optimize.keys(), self.result.x)
        )
        
        # Get loss function name
        loss_function_name = (
            self.loss_function.__name__ if hasattr(self.loss_function, '__name__') 
            else str(self.loss_function)
        )
        
        # Add Fitting_Model parameters to processing_parameters
        if 'fitting_model' not in parent_database.processing_parameters:
            parent_database.processing_parameters['fitting_model'] = {}
        
        parent_database.processing_parameters['fitting_model'] = {
            'reaction_network': self.reaction_network,
            'parsed_reactions': self.parsed_reactions,
            'species': self.species,
            'fixed_rate_constants': self.fixed_rate_constants,
            'rate_constants_to_optimize_bounds': self.rate_constants_to_optimize,
            'optimized_rate_constants': optimized_rate_constants,
            'data_to_be_fitted': self.data_to_be_fitted,
            'initial_conditions': self.initial_conditions,
            'other_multipliers': make_json_serializable(self.other_multipliers),
            'times': self.times,
            'experiments': experiment_list,
            'loss_function': loss_function_name,
            'optimization_result': {
                'success': self.result.success if hasattr(self.result, 'success') else None,
                'final_error': float(self.result.fun),
                'n_iterations': int(self.result.nit) if hasattr(self.result, 'nit') else None,
                'n_function_evaluations': int(self.result.nfev) if hasattr(self.result, 'nfev') else None,
                'message': self.result.message if hasattr(self.result, 'message') else None
            }
        }

        self.database = parent_database
        
        print(f"Fit results added to database for {len(experiment_list)} experiments.")
    
def objective_function(rate_constants_to_optimize, 
                       model: Fitting_Model, 
                       return_full = False):
    '''
    Calculate the objective function for kinetic model optimization.
    
    This function evaluates the total weighted error between model predictions and experimental 
    data across all experiments. It solves the ODE system for each experiment using the provided 
    rate constants and compares the results to experimental data using the specified loss function.
    
    Parameters
    ----------
    rate_constants_to_optimize : array-like
        Array of rate constant values to be optimized, corresponding to the keys in 
        model.rate_constants_to_optimize in the same order.
    model : Fitting_Model
        The kinetic model object containing reaction network, experimental data, initial 
        conditions, and optimization parameters.
    return_full : bool or str, default False
        Controls the return format:
        - False: Return only total error
        - True: Return total error and transformed model data for each experiment/species
        - 'All': Return total error, transformed model data, and full time series data
    
    Returns
    -------
    total_error : float
        The total weighted error across all experiments and species.
    full_output : dict, optional
        Dictionary mapping experiment names to species data. Only returned if 
        return_full is True or 'All'. Structure: {experiment_name: {species: model_data_transformed}}
    time_series : dict, optional
        Dictionary mapping experiment names to full ODE solution arrays. Only returned if 
        return_full is 'All'. Structure: {experiment_name: solution_array}
    
    Notes
    -----
    The function performs the following steps for each experiment:

    1. Combines optimized rate constants with fixed rate constants.
    2. Resolves experiment-specific attributes (initial conditions, multipliers,
       times, data). ``data_to_be_fitted`` and ``initial_conditions`` are
       resolved in 'semi-strict' mode, meaning at least one entry must resolve.
    3. Solves the ODE system using the reaction network and rate constants.
    4. Calculates the error between model predictions and experimental data
       using the loss function.
    5. Accumulates weighted errors across all experiments.
    
    Experiments can be provided as single objects (weight=1.0) or as (experiment, weight) tuples
    to allow differential weighting of experiments in the optimization.
    
    Examples
    --------
    >>> # Basic usage during optimization
    >>> error = objective_function([0.1, 0.5, 0.2], model)
    >>> 
    >>> # Get detailed results for analysis
    >>> error, results = objective_function([0.1, 0.5, 0.2], model, return_full=True)
    >>> 
    >>> # Get complete output including time series
    >>> error, results, time_series = objective_function([0.1, 0.5, 0.2], model, return_full='All')
    '''

    rate_constants = dict(zip(model.rate_constants_to_optimize.keys(), rate_constants_to_optimize))
    rate_constants |= model.fixed_rate_constants

    total_error = 0.0 
    full_output = {}
    time_series = {}

    for experiment_entry in model.experiments:

        if isinstance(experiment_entry, tuple):
            experiment, weight = experiment_entry
        else:
            experiment, weight = experiment_entry, 1.0 # Default weight of 1.0 if not specified

        experiment_name = experiment.experiment_name
        full_output[experiment_name] = {}

        # Resolve the rate constants, initial conditions, other multipliers, and times and data to be fitted
        initial_conditions = resolve_experiment_attributes(model.initial_conditions, 
                                                           experiment, 
                                                           mode = 'semi-strict')
        other_multipliers = resolve_experiment_attributes(model.other_multipliers, 
                                                          experiment,
                                                          mode = 'strict')
        times = resolve_experiment_attributes(model.times, 
                                              experiment,
                                              mode = 'strict')
        data_to_be_fitted = resolve_experiment_attributes(model.data_to_be_fitted, 
                                                          experiment,
                                                          mode = 'semi-strict')

        # Solve the ODE system
        model_result = solve_ode_system(model.parsed_reactions,
                                        model.species,
                                        rate_constants,
                                        initial_conditions,
                                        times['times'],
                                        other_multipliers)
        time_series[experiment_name] = model_result
               
        # Calculate the error between the model and the data for each species to be fitted
        experiment_error = 0.0

        for species, data in data_to_be_fitted.items():
            idx = model.species.index(species)
            model_data = model_result[:,idx]

            error, model_data_transformed = model.loss_function(model_data, data, times = times['times'])
            
            full_output[experiment_name][species] = model_data_transformed

            experiment_error += error
        
        total_error += experiment_error * weight

    if return_full is True:
        return total_error, full_output
    elif return_full == 'All':
        return total_error, full_output, time_series
    else:
        return total_error


def test_function(dataset): 
    """
    Fit the Ru(bpy)3 / persulfate network to a measured dataset.

    Demonstrates the intended workflow end to end: load an HDF5 dataset,
    declare which rate constants are fixed and which are searched within
    bounds, point the initial conditions and the photon flux at per-experiment
    metadata, and fit the O2 evolution rate across sixteen experiments at once.

    Parameters
    ----------
    dataset : str
        Path to an HDF5 dataset written by
        `pyKES.database.database_experiments.ExperimentalDataset.save_to_hdf5`.

    Returns
    -------
    None
        Runs the optimization and shows the resulting figure.
    """

    dataset = ExperimentalDataset.load_from_hdf5(dataset)
    dataset.update_reaction_data()

    model = Fitting_Model(['[RuII] > [RuII-ex], k1 ; hv_functionA',
                            '[RuII-ex] > [RuII], k8',
                            '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
                            '[RuIII] > [H2O2] + [RuII], k2 ; hv_function_B',
                            '2 [RuIII] > [Ru-Dimer], k3',
                            '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
                            '[H2O2] > [O2], k5',
                            '[RuIII] > [Inactive], k6'])
                           
    model.experiments = [dataset.experiments['MRG-059-ZO-2-1'],
                         dataset.experiments['MRG-059-ZN-7-1'],
                         dataset.experiments['MRG-059-ZN-4-1'],
                         dataset.experiments['MRG-059-ZN-3-1'],
                         (dataset.experiments['MRG-059-ZN-2-1'], 1),
                         dataset.experiments['MRG-059-ZN-1-1'],
                         dataset.experiments['MRG-059-ZO-1-1'],
                         dataset.experiments['MRG-059-ZO-8-1'],
                         (dataset.experiments['MRG-059-ZN-10-1'], 1),
                         (dataset.experiments['MRG-059-ZN-9-1'], 1),
                         (dataset.experiments['MRG-059-ZN-8-1'], 1),
                         (dataset.experiments['MRG-059-ZO-9-1'], 1),
                         dataset.experiments['MRG-059-ZO-3-1'],
                         (dataset.experiments['MRG-059-ZN-14-1'], 0.1),
                         dataset.experiments['MRG-059-ZN-13-1'],
                         dataset.experiments['MRG-059-ZN-11-1']]
    
    model.fixed_rate_constants = {
        'k8': 1/650e-9
    }
    model.rate_constants_to_optimize = {'k1': (1E-1, 1E-0),
                                        'k2': (1E-1, 1E-0),
                                        'k3': (1E-3, 1E-1),
                                        'k4': (1E-3, 1E-1),
                                        'k5': (1E-3, 5E-1),
                                        'k6': (1E-3, 5E-1),
                                        'k7': (1E+0, 6E+1)}  
    
    model.data_to_be_fitted = {
        '[O2]': {'x': 'time_series_data.x_diff',
                 'y': 'time_series_data.y_diff'}
        }
    
    model.initial_conditions = {
            '[S2O8]': 'experiment_metadata.oxidant_concentration_uM',
            '[RuII]': 'experiment_metadata.ru_concentration_uM'
        }
        
    model.other_multipliers = {
        'pathlength': 2.25,
        'photon_flux': 'experiment_metadata.photon_flux',
        'Ru_II_extinction_coefficient': 8500,
        'Ru_III_extinction_coefficient': 540,
        'hv_functionA': {
            'function': calculate_excitations_per_second_competing,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_A': '[RuII]',
                'concentration_B': '[RuIII]',
                'extinction_coefficient_A': 'Ru_II_extinction_coefficient',
                'extinction_coefficient_B': 'Ru_III_extinction_coefficient',
                'pathlength': 'pathlength'
            }
        },
        'hv_function_B': {
            'function': calculate_excitations_per_second_competing,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_A': '[RuIII]',
                'concentration_B': '[RuII]',
                'extinction_coefficient_A': 'Ru_III_extinction_coefficient',
                'extinction_coefficient_B': 'Ru_II_extinction_coefficient',
                'pathlength': 'pathlength'
            }
        }
    }

    model.times = {
            'times': 'time_series_data.time_reaction'
        }

    model.loss_function = square_loss_ydiff

    # Uncomment the following line to run the optimization
    model.optimize()



    # Optimized parameters
    # model.result = SimpleNamespace()
    # model.result.x = np.array([9.995e-01,
    #                     9.886e-01,
    #                     7.407e-03,
    #                     3.437e-03,
    #                     2.739e-02,
    #                     4.762e-03,
    #                     5.918e+01])



    plt.show()

if __name__ == '__main__':
    test_function('/Users/jacob/Documents/Water_Splitting/Projects/HTE_Photocatalysis/HTE_Streamlit_App/data/250608_HTE.h5')
