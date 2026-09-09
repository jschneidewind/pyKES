"""Object-oriented front end to simulation and pathway analysis.

`Reaction_Model` bundles a reaction network, its rate constants, initial
conditions and multipliers into one object, and exposes the four things
normally done with them in sequence: integrate the network
(`Reaction_Model.solve_ode`), plot the concentration traces
(`Reaction_Model.plot_solution`), trace where the absorbed photons end up at
a chosen time (`Reaction_Model.calculate_reaction_network_propopagation`) and
draw that as a pathway diagram
(`Reaction_Model.plot_reaction_network_propagation`).

It holds every intermediate result as an attribute rather than returning it,
which is what makes it convenient in a notebook: the parsed network, the
solution array, the concentrations at the analysed time point and the
transformed pathway data all stay available for inspection afterwards.

Use `pyKES.reaction_ODE` directly when the functional interface is enough, and
`pyKES.fitting_ODE.Fitting_Model` when the rate constants are to be fitted
rather than given.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyKES.reaction_ODE import parse_reactions, solve_ode_system, plot_solution
from pyKES.pathways.pathways import calculate_reaction_network_propagation
from pyKES.pathways.transform_pathways_data import transform_data_for_plotting
from pyKES.plotting.plotting_pathways_transformed import plot_pathway_bars
from pyKES.utilities.calculate_absorption import calculate_excitations_per_second_multi_competing_fast
from pyKES.utilities.find_nearest import find_nearest

class Reaction_Model:
    """
    A reaction network together with everything needed to simulate it.

    Parameters
    ----------
    reaction_network : list of str, optional
        Reactions in the string format of `pyKES.reaction_ODE.parse_reactions`.
    rate_constants : dict, optional
        Mapping of rate constant identifiers to their values.
    initial_conditions : dict, optional
        Mapping of species names to their concentrations at t = 0. Species left
        out start at zero.
    other_multipliers : dict, optional
        Mapping of multiplier identifiers to values or function specifications,
        for example light-absorption terms; see
        `pyKES.reaction_ODE.resolve_other_multipliers`.
    times : array_like, optional
        Time points the solution is reported at.

    Attributes
    ----------
    parsed_reactions : list of dict
        The parsed network, filled in on construction.
    species : list of str
        Sorted species of the network, fixing the column order of `solution`.
    solution : numpy.ndarray
        Concentration array of shape ``(len(times), len(species))``, available
        after `solve_ode`.
    propagation_results : dict
        Nested pathway tree, available after
        `calculate_reaction_network_propopagation`.
    transformed_propagation_data : dict
        Node/link representation of that tree, available after
        `plot_reaction_network_propagation`.

    Examples
    --------
    >>> model = Reaction_Model(reaction_network=['[A] > [B], k1'],
    ...                        rate_constants={'k1': 0.1},
    ...                        initial_conditions={'[A]': 1.0},
    ...                        times=np.linspace(0, 60, 200))
    >>> model.solve_ode()
    >>> model.plot_solution()
    """

    def __init__(self, **kwargs):
        '''
        Initialize the model and parse its reaction network.

        Parameters
        ----------
        **kwargs
            ``reaction_network``, ``rate_constants``, ``initial_conditions``,
            ``other_multipliers`` and ``times``, as documented on the class.
            Each defaults to an empty container, so a model can be built
            piecewise by assigning to the attributes afterwards.

        Returns
        -------
        None
        '''
        self.reaction_network: list = kwargs.get('reaction_network', [])
        self.rate_constants: dict = kwargs.get('rate_constants', {})
        self.initial_conditions: dict = kwargs.get('initial_conditions', {})
        self.other_multipliers: dict = kwargs.get('other_multipliers', {})
        self.times: dict = kwargs.get('times', {})

        self.parsed_reactions, self.species = parse_reactions(self.reaction_network)

    def solve_ode(self):
        '''
        Integrate the network and store the result on the model.

        Returns
        -------
        None : None
            The solution array is stored as ``self.solution``.
        '''

        self.solution = solve_ode_system(
            self.parsed_reactions,
            self.species,
            self.rate_constants,
            self.initial_conditions,
            self.times,
            self.other_multipliers,)
        
    def plot_solution(self, exclude_species = [], ax = None):
        '''
        Plot the concentration-time traces of the solved network.

        Parameters
        ----------
        exclude_species : list of str, optional
            Species left out of the figure, typically reagents in large excess
            whose trace would flatten everything else.
        ax : matplotlib.axes.Axes, optional
            Axes drawn on. A new figure is created when omitted.

        Returns
        -------
        None
        '''

        plot_solution(self.species,
                      self.times, 
                      self.solution, 
                      exclude_species=exclude_species, ax=ax)


    def calculate_reaction_network_propopagation(self,
                                                 timepoint: float,
                                                 absorbing_species_with_extinction_coefficients: dict,
                                                 photon_flux: float,
                                                 pathlength: float,
                                                 concentration_unit: str):
        """
        Trace where the absorbed photons end up, at one point in time.

        The network is frozen at the requested time: the concentrations there
        fix the branching ratio of every species, and the absorbed photons are
        followed through those branches until each pathway terminates. The
        result answers what an integrated trace cannot — which fraction of the
        absorbed light reaches the product and which fraction is lost, and to
        what.

        Requires `solve_ode` to have been run.

        Parameters
        ----------
        timepoint : float
            Time the analysis is performed at. The nearest available point of
            ``self.times`` is used.
        absorbing_species_with_extinction_coefficients : dict
            Mapping of every light-absorbing ground state to
            ``{'excited_name': str, 'extinction_coefficient': float}``, the
            latter in M^-1 cm^-1. Species listed here compete for the incident
            photons.
        photon_flux : float
            Photon flux in photons cm^-2 s^-1.
        pathlength : float
            Optical path length of the sample in cm.
        concentration_unit : str
            Unit the concentrations of the network are expressed in, e.g.
            ``'uM'``. Needed to turn them into absorbances.

        Returns
        -------
        None : None
            The pathway tree is stored as ``self.propagation_results``; the
            time index, the picked solution and the concentrations at that time
            are stored alongside it.
        """

        # Storing photon flux and pathlength
        self.photon_flux = photon_flux
        self.pathlength = pathlength    

        # Picking the ODE solution at the specified timepoint
        self.timepoint_idx = find_nearest(self.times, timepoint)[0]
        self.picked_solution = self.solution[self.timepoint_idx]
        self.concentrations_at_timepoint = {species: self.picked_solution[i] for i, species in enumerate(self.species)}

        # Setting up absorbing species and their extinction coefficients dicts
        self.absorbing_species = {species: value['excited_name'] for species, value in absorbing_species_with_extinction_coefficients.items()}
        self.extinction_coefficients = {species: value['extinction_coefficient'] for species, value in absorbing_species_with_extinction_coefficients.items()}
  
        # Calculating the reaction network propagation
        self.propagation_results = calculate_reaction_network_propagation(
                    concentrations = self.concentrations_at_timepoint,
                    parsed_reactions = self.parsed_reactions,
                    rate_constants = self.rate_constants,
                    absorbing_species = self.absorbing_species,
                    extinction_coefficients = self.extinction_coefficients,
                    photon_flux = self.photon_flux,
                    pathlength = self.pathlength,
                    concentration_unit = concentration_unit,
                    other_multipliers = self.other_multipliers,)
    
    def plot_reaction_network_propagation(self, 
                                          ax = None,
                                          value_key = 'log_value',
                                          fanning_factor = 0.7,
                                          assumed_branching_degree = 1.7,
                                          excluded_nodes = [],
                                          excluded_links = [],
                                          forward_link_kwargs = {},
                                          backward_link_kwargs = {},
                                          **kwargs
                                          ):
        """
        Draw the pathway diagram of a computed network propagation.

        Requires `calculate_reaction_network_propopagation` to have been run.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes drawn on. A new figure is created when omitted.
        value_key : {'log_value', 'value'}, optional
            Quantity the bar heights encode. Photon budgets span several orders
            of magnitude, so the log-normalized value keeps minor pathways
            visible; ``'value'`` draws true proportions instead.
        fanning_factor : float, optional
            Vertical space each node gives its children. Larger values spread
            branches further apart.
        assumed_branching_degree : float, optional
            Assumed number of children per node, used to widen the layout of
            early levels so that later ones still fit.
        excluded_nodes, excluded_links : list, optional
            Node names and ``(source, target)`` pairs omitted from the figure.
        forward_link_kwargs, backward_link_kwargs : dict, optional
            Matplotlib keyword arguments for forward-running and looping
            (back-reaction) bands respectively.
        **kwargs
            Passed on to `pyKES.plotting.plotting_pathways_transformed.plot_pathway_bars`.

        Returns
        -------
        None : None
            The layout is stored as ``self.transformed_propagation_data``.
        """

        self.transformed_propagation_data = transform_data_for_plotting(
            self.propagation_results,
            value_key = value_key,
            fanning_factor = fanning_factor,
            assumed_branching_degree = assumed_branching_degree,)
        
        plot_pathway_bars(
            self.transformed_propagation_data,
            ax = ax,
            excluded_nodes = excluded_nodes,
            excluded_links = excluded_links,
            forward_link_kwargs = forward_link_kwargs,
            backward_link_kwargs = backward_link_kwargs,
            **kwargs)
        




def full_testing():
    """
    Demonstrate the full model on a three-state photochemical cascade.

    A absorbs light and either relaxes or converts to B; B absorbs in turn and
    either relaxes, reverts to A or converts to C. Both the concentration
    traces and the pathway diagram of the absorbed photons are drawn, which is
    the pairing the class exists for.

    Returns
    -------
    None
        Shows two matplotlib figures.
    """

    PHOTON_FLUX = 1e17 # photons/cm2/s
    PATHLENGTH = 2.25 # cm

    A_EXTINCTION_COEFFICIENT = 8500 # M^-1 cm^-1
    B_EXTINCTION_COEFFICIENT = 5400 # M^-1 cm^-1
    C_EXTINCTION_COEFFICIENT = 1000 # M^-1 cm^-1

    reactions = ['[A] > [A-excited], k1 ; hv_functionA',
                 '[A-excited] > [A], k2',
                 '[A-excited] > [B], k3',
                 '[B] > [A], k4',
                 '[B] > [B-excited], k5 ; hv_functionB',
                 '[B-excited] > [B], k6',
                 '[B-excited] > [C], k7',]
    
    rate_constants = {'k1': 1,
                      'k2': 3e8,
                      'k3': 1e8,
                      'k4': 1e2,
                      'k5': 1,
                      'k6': 2e8,
                      'k7': 3.3e8,
                      'k10': 1e11,
                      'k11': 1e11,
                      'k12': 1e-1,}
    
    initial_conditions = {'[A]': 10.0, # concentration in uM
                          '[B]': 0} # concentration in uM 
    
    other_multipliers = {
        'pathlength': PATHLENGTH,
        'photon_flux': PHOTON_FLUX,
        'A_extinction_coefficient': A_EXTINCTION_COEFFICIENT,
        'B_extinction_coefficient': B_EXTINCTION_COEFFICIENT,
        'C_extinction_coefficient': C_EXTINCTION_COEFFICIENT,
        'hv_functionA_species_of_interest': '[A]',
        'hv_functionB_species_of_interest': '[B]',
        'hv_functionC_species_of_interest': '[C]',
        'hv_functionA': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_[A]': '[A]',
                'concentration_[B]': '[B]',
                'concentration_[C]': '[C]',
                'extinction_coefficient_[A]': 'A_extinction_coefficient',
                'extinction_coefficient_[B]': 'B_extinction_coefficient',
                'extinction_coefficient_[C]': 'C_extinction_coefficient',
                'pathlength': 'pathlength',
                'species_of_interest': 'hv_functionA_species_of_interest',
            }
        },
        'hv_functionB': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_[A]': '[A]',
                'concentration_[B]': '[B]',
                'concentration_[C]': '[C]',
                'extinction_coefficient_[A]': 'A_extinction_coefficient',
                'extinction_coefficient_[B]': 'B_extinction_coefficient',
                'extinction_coefficient_[C]': 'C_extinction_coefficient',
                'pathlength': 'pathlength',
                'species_of_interest': 'hv_functionB_species_of_interest',
            }
        },
        'hv_functionC': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_[A]': '[A]',
                'concentration_[B]': '[B]',
                'concentration_[C]': '[C]',
                'extinction_coefficient_[A]': 'A_extinction_coefficient',
                'extinction_coefficient_[B]': 'B_extinction_coefficient',
                'extinction_coefficient_[C]': 'C_extinction_coefficient',
                'pathlength': 'pathlength',
                'species_of_interest': 'hv_functionC_species_of_interest',
            }
        },
    }

    absorbing_species_with_extinction_coefficients = {'[A]': {
                                                        'excited_name': '[A-excited]', 
                                                        'extinction_coefficient': A_EXTINCTION_COEFFICIENT
                                                            },
                                                      '[B]': {
                                                        'excited_name': '[B-excited]',
                                                        'extinction_coefficient': B_EXTINCTION_COEFFICIENT
                                                        },
                                                      '[C]': {
                                                        'excited_name': '[C-excited]', 
                                                        'extinction_coefficient': C_EXTINCTION_COEFFICIENT}}
    
    times = np.linspace(0, 1000, 1000)

    model = Reaction_Model(reaction_network = reactions,
                           rate_constants = rate_constants,
                           initial_conditions = initial_conditions,
                           other_multipliers = other_multipliers,
                           times = times,)
    
    model.solve_ode()

    model.calculate_reaction_network_propopagation(
        timepoint = 10,
        absorbing_species_with_extinction_coefficients = absorbing_species_with_extinction_coefficients,
        photon_flux = PHOTON_FLUX,
        pathlength = PATHLENGTH,
        concentration_unit = 'uM',)
    
    model.plot_solution()
    model.plot_reaction_network_propagation(forward_link_kwargs = {'alpha': 0.6},)

    plt.show()
    


def testing():
    """
    Simulate the Ru(bpy)3 / persulfate water-oxidation network.

    The same network as `pyKES.reaction_ODE.test_function`, built through
    `Reaction_Model` instead of the functional interface.

    Returns
    -------
    None
        Shows the concentration-time plot.
    """

    reactions = ['[RuII] > [RuII-ex], k1 ; hv_functionA',
                 '[RuII-ex] > [RuII], k8',
                 '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
                 '[RuIII] > [H2O2] + [RuII], k2 ; hv_functionB',
                 '2 [RuIII] > [Ru-Dimer], k3',
                 '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
                 '[H2O2] > [O2], k5',
                 '[RuIII] > [Inactive], k6']
    
    rate_constants = {'k1': 9.995e-01,
                      'k2': 9.886e-01,
                      'k3': 7.407e-03,
                      'k4': 3.437e-03,
                      'k5': 2.739e-02,
                      'k6': 4.762e-03,
                      'k7': 5.918e+01,
                      'k8': 1/650e-9}
    
    initial_conditions =  {'[S2O8]': 6000,
                           '[RuII]': 10}
    
    other_multipliers = {
        'pathlength': 2.25,
        'photon_flux': 1e17,
        'Ru_II_extinction_coefficient': 8500,
        'Ru_III_extinction_coefficient': 540,
        'hv_functionA_species_of_interest': '[RuII]',
        'hv_functionB_species_of_interest': '[RuIII]',
        'hv_functionA': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_[RuII]': '[RuII]',
                'concentration_[RuIII]': '[RuIII]',
                'extinction_coefficient_[RuII]': 'Ru_II_extinction_coefficient',
                'extinction_coefficient_[RuIII]': 'Ru_III_extinction_coefficient',
                'pathlength': 'pathlength',
                'species_of_interest': 'hv_functionA_species_of_interest',
            }
        },
        'hv_functionB': {
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_[RuIII]': '[RuIII]',
                'concentration_[RuII]': '[RuII]',
                'extinction_coefficient_[RuIII]': 'Ru_III_extinction_coefficient',
                'extinction_coefficient_[RuII]': 'Ru_II_extinction_coefficient',
                'pathlength': 'pathlength',
                'species_of_interest': 'hv_functionB_species_of_interest',
            }
        }
    }

    times = np.linspace(0, 300, 1000) 

    absorbing_species_with_extinction_coefficients = {'[A]': {
                                                 'excited_name': '[A-excited]', 
                                                 'extinction_coefficient': 1000
                                                        },
                                                      '[B]': {
                                                'excited_name': '[B-excited]',
                                                'extinction_coefficient': 2000
                                                        },
                                                        '[C]': {
                                                'excited_name': '[C-excited]', 
                                                'extinction_coefficient': 3000}}

    model = Reaction_Model(reaction_network = reactions,
                           rate_constants = rate_constants,
                           initial_conditions = initial_conditions,
                           other_multipliers = other_multipliers,
                           times = times)
    
    model.solve_ode()
    model.plot_solution(exclude_species = ['[S2O8]', '[SO4]'])

    plt.show()






if __name__ == "__main__":
    #testing()
    full_testing()   


