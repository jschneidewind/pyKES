import matplotlib.pyplot as plt
import numpy as np

from pyKES.reaction_ODE import parse_reactions, solve_ode_system, plot_solution
from pyKES.pathways.pathways import calculate_reaction_network_propagation
from pyKES.pathways.transform_pathways_data import transform_data_for_plotting
from pyKES.plotting.plotting_pathways_transformed import plot_pathway_bars
from pyKES.utilities.calculate_absorption import calculate_excitations_per_second_multi_competing_fast
from pyKES.utilities.find_nearest import find_nearest

class Reaction_Model:

    def __init__(self, **kwargs):
        '''
        Initilize the Reaction_Model class.
        '''
        self.reaction_network: list = kwargs.get('reaction_network', [])
        self.rate_constants: dict = kwargs.get('rate_constants', {})
        self.initial_conditions: dict = kwargs.get('initial_conditions', {})
        self.other_multipliers: dict = kwargs.get('other_multipliers', {})
        self.times: dict = kwargs.get('times', {})

        self.parsed_reactions, self.species = parse_reactions(self.reaction_network)

    def solve_ode(self):
        '''
        Solve the ODE system for the reaction model.
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
        Plot the solution of the ODE system.
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


