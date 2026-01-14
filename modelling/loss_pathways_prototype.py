import numpy as np
import matplotlib.pyplot as plt
import pprint as pp

from pyKES.reaction_ODE import parse_reactions, solve_ode_system, plot_solution
from pyKES.utilities.calculate_absorption import calculate_excitations_per_second_multi_competing, calculate_excitations_per_second_multi_competing_fast

def main():

    PHOTON_FLUX = 1e17
    PATHLENGTH = 2.25
    A_EXTINCTION_COEFFICIENT = 8500
    B_EXTINCTION_COEFFICIENT = 5400
    C_EXTINCTION_COEFFICIENT = 1000

    reactions = ['[A] > [B], q1 ; hv_functionA',
                 '[B] > [A], k1',
                 '[B] > [C], q2 ; hv_functionB',]
    
    rate_constants = {'q1': 0.3,
                      'q2': 0.6,
                      'k1': 1e2}
    
    initial_conditions = {'[A]': 10.0} 

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
        }
    }

    times = np.linspace(0, 1000, 1000)

    parsed_reactions, species = parse_reactions(reactions)
    
    pp.pprint(parsed_reactions)

    solution = solve_ode_system(parsed_reactions, 
                                species, 
                                rate_constants,
                                initial_conditions,
                                times,
                                other_multipliers)
    
    colors = {'[A]': 'blue',
              '[B]': 'orange',
              '[C]': 'green'}
    
    extinction_coefficients = {'[A]': A_EXTINCTION_COEFFICIENT,
                               '[B]': B_EXTINCTION_COEFFICIENT,
                               '[C]': C_EXTINCTION_COEFFICIENT}

    for counter, row in enumerate(solution):

        concentrations = {'[A]': row[species.index('[A]')],
                          '[B]': row[species.index('[B]')],
                          '[C]': row[species.index('[C]')]}

        for single_species in species:

            excitations_per_second = calculate_excitations_per_second_multi_competing_fast(
                photon_flux = PHOTON_FLUX,
                pathlength = PATHLENGTH,
                species_of_interest = single_species,
                extinction_coefficients = extinction_coefficients,
                concentrations = concentrations
            )

            plt.plot(counter, excitations_per_second, '.', color = colors[single_species])






    
    picked_solution = solution[10]

    print(picked_solution)

    concentrations = {'[A]': picked_solution[species.index('[A]')],
                      '[B]': picked_solution[species.index('[B]')],
                      '[C]': picked_solution[species.index('[C]')]}
    


    excitations_per_species_dict, absorbed_dict = calculate_excitations_per_second_multi_competing(
                                                species_of_interest = ['A'],
                                                photon_flux = PHOTON_FLUX,
                                                pathlength = PATHLENGTH,
                                                concentrations = concentrations,
                                                extinction_coefficients = extinction_coefficients,
                                                return_full = True,)
    

    pp.pprint(absorbed_dict)

    print(sum(absorbed_dict.values()))


    

    
    plot_solution(species, times, solution)
    plt.show()




if __name__ == '__main__':
    main()
