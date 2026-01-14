import numpy as np
import matplotlib.pyplot as plt
import pprint as pp

from pyKES.reaction_ODE import parse_reactions, solve_ode_system, plot_solution
from pyKES.pathways.pathways import calculate_reaction_pathway_proportions, calculate_reaction_network_propagation
from pyKES.utilities.calculate_absorption import calculate_excitations_per_second_multi_competing, calculate_excitations_per_second_multi_competing_fast, AVOGADRO_NUMBER

def main():

    PHOTON_FLUX = 1e17
    PATHLENGTH = 2.25
    A_EXTINCTION_COEFFICIENT = 8500
    B_EXTINCTION_COEFFICIENT = 5400
    C_EXTINCTION_COEFFICIENT = 1000

    # Used mechanism
    reactions = ['[A] > [A-excited], k1 ; hv_functionA',
                 '[A-excited] > [A], k2',
                 '[A-excited] > [B], k3',
                 '[B] > [A], k4',
                 '[B] > [B-excited], k5 ; hv_functionB',
                 '[B-excited] > [B], k6',
                 '[B-excited] > [C], k7',]

    # Added C excitation
    # reactions = ['[A] > [A-excited], k1 ; hv_functionA',
    #              '[A-excited] > [A], k2',
    #              '[A-excited] > [B], k3',
    #              '[B] > [A], k4',
    #              '[B] > [B-excited], k5 ; hv_functionB',
    #              '[B-excited] > [B], k6',
    #              '[B-excited] > [C], k7',
    #              '[C] > [C-excited], k1 ; hv_functionC',
    #              '[C-excited] > [C], k10',]    



    # reactions = ['[A] > [A-excited], k1 ; hv_functionA',
    #              '[A-excited] > [A], k10',
    #              '[A-excited] > [X], k2',
    #              '[B] > [B-excited], k5 ; hv_functionB',
    #              '[B-excited] > [B], k11',
    #              '[B-excited] > [X], k7',
    #              '[X] > [C], k12',
    #             ]

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

    # other_multipliers = {
    #     'pathlength': PATHLENGTH,
    #     'photon_flux': PHOTON_FLUX,
    #     'A_extinction_coefficient': A_EXTINCTION_COEFFICIENT,
    #     'B_extinction_coefficient': B_EXTINCTION_COEFFICIENT,
    #     'C_extinction_coefficient': C_EXTINCTION_COEFFICIENT,
    #     'hv_functionA_species_of_interest': '[A]',
    #     'hv_functionB_species_of_interest': '[B]',
    #     'hv_functionA': {
    #         'function': calculate_excitations_per_second_multi_competing_fast,
    #         'arguments': {
    #             'photon_flux': 'photon_flux',
    #             'concentration_[A]': '[A]',
    #             'concentration_[B]': '[B]',
    #             'concentration_[C]': '[C]',
    #             'extinction_coefficient_[A]': 'A_extinction_coefficient',
    #             'extinction_coefficient_[B]': 'B_extinction_coefficient',
    #             'extinction_coefficient_[C]': 'C_extinction_coefficient',
    #             'pathlength': 'pathlength',
    #             'species_of_interest': 'hv_functionA_species_of_interest',
    #         }
    #     },
    #     'hv_functionB': {
    #         'function': calculate_excitations_per_second_multi_competing_fast,
    #         'arguments': {
    #             'photon_flux': 'photon_flux',
    #             'concentration_[A]': '[A]',
    #             'concentration_[B]': '[B]',
    #             'concentration_[C]': '[C]',
    #             'extinction_coefficient_[A]': 'A_extinction_coefficient',
    #             'extinction_coefficient_[B]': 'B_extinction_coefficient',
    #             'extinction_coefficient_[C]': 'C_extinction_coefficient',
    #             'pathlength': 'pathlength',
    #             'species_of_interest': 'hv_functionB_species_of_interest',
    #         }
    #     }
    # }

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

    times = np.linspace(0, 1000, 1000)

    parsed_reactions, species = parse_reactions(reactions)
    
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

    
    picked_solution = solution[10]



    concentrations = {'[A]': picked_solution[species.index('[A]')],
                      '[B]': picked_solution[species.index('[B]')],
                      '[C]': picked_solution[species.index('[C]')]}
    
    concentrations_full = {species: picked_solution[i] for i, species in enumerate(species)}

    excitations_per_species_dict, absorbed_dict = calculate_excitations_per_second_multi_competing(
                                                species_of_interest = ['A'],
                                                photon_flux = PHOTON_FLUX,
                                                pathlength = PATHLENGTH,
                                                concentrations = concentrations,
                                                extinction_coefficients = extinction_coefficients,
                                                return_full = True,)
    

    pp.pprint(absorbed_dict)
    print(sum(absorbed_dict.values()))

    pathways_A_excited = calculate_reaction_pathway_proportions(
        selected_species = '[A-excited]',
        concentrations = concentrations_full,
        parsed_reactions = parsed_reactions,
        rate_constants = rate_constants,
        other_multipliers = other_multipliers,
    )

    pathways_B_excited = calculate_reaction_pathway_proportions(
        selected_species = '[B-excited]',
        concentrations = concentrations_full,
        parsed_reactions = parsed_reactions,
        rate_constants = rate_constants,
        other_multipliers = other_multipliers,
    )

    pp.pprint(pathways_A_excited)
    pp.pprint(pathways_B_excited)

    absorbing_species = {'[A]': '[A-excited]',
                         '[B]': '[B-excited]',
                         '[C]': '[C-excited]'}

    propagation_results = calculate_reaction_network_propagation(
        concentrations = concentrations_full,
        parsed_reactions = parsed_reactions,
        rate_constants = rate_constants,
        absorbing_species=absorbing_species,
        extinction_coefficients=extinction_coefficients,
        photon_flux = PHOTON_FLUX,
        pathlength = PATHLENGTH,
        other_multipliers = other_multipliers,
    )

    pp.pprint(propagation_results)

    #### ----------------------------------------------------------
    #### Comparison of analytical solution with propagation result

    concentrations_t0 = solution[10]
    concentrations_t1 = solution[11]

    concentrations_delta = concentrations_t1 - concentrations_t0
    concentrations_delta_dict = {species[i]: concentrations_delta[i] for i in range(len(species))}

    pp.pprint(concentrations_delta_dict)
    
    print(concentrations_delta_dict['[A]'] + concentrations_delta_dict['[B]'])

    time_delta = times[11] - times[10]

    number_of_photons = PHOTON_FLUX * time_delta # photons per cm2 (assuming 1 cm2 area for simplicity)
    umol_photons = (number_of_photons / AVOGADRO_NUMBER) * 1e6

    volume_L = 1 * PATHLENGTH / 1000  # cm3 to L
    umol_formed_C = concentrations_delta_dict['[C]'] * volume_L

    
    C_formed_via_A = propagation_results['Light absorption']['[A]']['[A-excited]']['[B]']['[B-excited]']['[C]']['amount_formed']
    C_formed_via_B = propagation_results['Light absorption']['[B]']['[B-excited]']['[C]']['amount_formed']

    # For [X] pathway
    # C_formed_via_A = propagation_results['Light absorption']['[A]']['[A-excited]']['[X]']['[C]']['amount_formed']
    # C_formed_via_B = propagation_results['Light absorption']['[B]']['[B-excited]']['[X]']['[C]']['amount_formed']

    print('--- Detailed pathway contributions to C formation ---')
    print('C formed via A pathway: ', C_formed_via_A)
    print('C formed via B pathway: ', C_formed_via_B)

    C_formed = C_formed_via_A + C_formed_via_B
    
    umol_formed_C_predicted = C_formed * umol_photons


    print('--- Comparison of analytical vs predicted from propagation ---')
    print('Prediction total:       ', umol_formed_C_predicted)
    print('Prediction via A:       ', C_formed_via_A * umol_photons)
    print('Prediction via B:       ', C_formed_via_B * umol_photons)
    print('Analytical:             ', umol_formed_C)
    
    # Result 08.01.2026
    # Approximation by propagation only holds if steps after excitation are fast
    # See above, if rate constant for [X] > [C] is slow, deviation starts to appear







    plot_solution(species, times, solution)
    plt.show()




if __name__ == '__main__':
    main()
