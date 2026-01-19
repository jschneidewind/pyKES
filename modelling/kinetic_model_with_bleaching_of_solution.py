import numpy as np 
import matplotlib.pyplot as plt

from pyKES.reaction_model import Reaction_Model
from pyKES.utilities.calculate_absorption import calculate_excitations_per_second_multi_competing_fast

def main():

    PHOTON_FLUX = 1e17 # photons/s/cm2
    PATHLENGTH = 1.0   # cm

    A_EXTINCTION_COEFFICIENT = 5000  # M^-1 cm^-1
    B_EXTINCTION_COEFFICIENT = 5000  # M^-1 cm^-1

    reactions = ['[A] > [B], k1 ; hv_functionA']
    #reactions = ['2 [A] > [B], k1']


    rate_constants = {'k1': 0.08}  # 1/s

    initial_conditions = {'[A]': 10, '[B]': 0.0}  # uM

    times = np.linspace(0, 1000, 10000)

    other_multipliers = {
        'pathlength': PATHLENGTH,
        'photon_flux': PHOTON_FLUX,
        'concentration_unit': 'mM',
        'A_extinction_coefficient': A_EXTINCTION_COEFFICIENT,
        'B_extinction_coefficient': B_EXTINCTION_COEFFICIENT,
        'hv_functionA_species_of_interest': '[A]',
        'hv_functionA': { 
            'function': calculate_excitations_per_second_multi_competing_fast,
            'arguments': {
                'photon_flux': 'photon_flux',
                'pathlength': 'pathlength',
                'concentration_unit': 'concentration_unit',
                'species_of_interest': 'hv_functionA_species_of_interest',
                'concentration_[A]': '[A]',
                'concentration_[B]': '[B]',
                'extinction_coefficient_[A]': 'A_extinction_coefficient',
                'extinction_coefficient_[B]': 'B_extinction_coefficient',
            },

        }        
    }



    model = Reaction_Model(
            reaction_network = reactions,
            rate_constants = rate_constants,
            initial_conditions = initial_conditions,
            times = times,
            other_multipliers = other_multipliers,
        )

    model.solve_ode()
    model.plot_solution()

    initial_concentrations = np.logspace(np.log10(0.01), np.log10(10), 50)
    max_rates = []


    for initial_concentration in initial_concentrations:
        initial_conditions = {'[A]': initial_concentration, '[B]': 0.0}  # uM

        model = Reaction_Model(
            reaction_network = reactions,
            rate_constants = rate_constants,
            initial_conditions = initial_conditions,
            times = times,
            other_multipliers = other_multipliers,
        )

        model.solve_ode()

        # Get max rate of formation of [B]
        concentration_B = model.solution[:, 1]
        time_points = model.times
        rates_B = np.gradient(concentration_B, time_points)
        max_rate_B = np.max(rates_B)
        max_rates.append(max_rate_B)

    #plt.plot(initial_concentrations, max_rates, marker='o')



    excitations_per_second = calculate_excitations_per_second_multi_competing_fast(
        photon_flux = PHOTON_FLUX,
        pathlength = PATHLENGTH,
        concentration_unit = 'mM',
        species_of_interest = '[A]',
        concentrations = {
            '[A]': 0.01,
            '[B]': 9.99,
        },
        extinction_coefficients = {
            '[A]': A_EXTINCTION_COEFFICIENT,
            '[B]': B_EXTINCTION_COEFFICIENT,
        },)
    
    print(excitations_per_second)
    



    #model.plot_solution()

    plt.show()




    


if __name__ == "__main__":
    main()