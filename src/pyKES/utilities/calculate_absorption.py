import numpy as np
import cProfile 
import pstats
import pprint as pp

AVOGADRO_NUMBER = 6.022e23  # Avogadro's number in mol^-1

def calculate_excitations_per_second(photon_flux = None, 
                                    concentration = None, 
                                    extinction_coefficient = None,
                                    pathlength = None):
    """
    Calculate the number of excitations per Ru per second based on photon flux and concentration.

    Parameters
    ----------
    photon_flux : float
        Photon flux in photons cm^-2 s^-1.
    concentration : float
        Concentration of the species in micromolar (uM).
    extinction_coefficient : float
        Extinction coefficient of species in M^-1 cm^-1.
    pathlength : float
        Path length of the sample in cm (e.g., the distance light travels through the sample).  

    Returns
    -------
    float
        Number of excitations per Ru per second.
    """

    concentration_M = concentration * 1e-6  # Convert concentration from uM to M
    volume_L = (pathlength * 1) / 1000 # Assuming a unit area (1 cm2) for simplicity, converting from cm3 to L
    photon_flux_mol = photon_flux / AVOGADRO_NUMBER  # Convert photon flux to mol/s
    
    absorbance = concentration_M * extinction_coefficient * pathlength  # Calculation of absorbance using Beer-Lambert law
    absorbed_fraction = 1 - 10**(-absorbance)  # Fraction of photons absorbed

    excitations_per_Ru = (photon_flux_mol * absorbed_fraction) / (volume_L * concentration_M)

    return excitations_per_Ru

def calculate_excitations_per_second_competing(photon_flux,
                                               concentration_A,
                                               concentration_B,
                                               extinction_coefficient_A,
                                               extinction_coefficient_B,
                                               pathlength):
    '''
    Calculate the number of excitations per A per second for two competing species A and B.
    
    Parameters
    ----------
    photon_flux : float
        Photon flux in photons cm^-2 s^-1.
    concentration_A : float
        Concentration of species A in micromolar (uM).
    concentration_B : float
        Concentration of species B in micromolar (uM).
    extinction_coefficient_A : float
        Extinction coefficient of species A in M^-1 cm^-1.
    extinction_coefficient_B : float
        Extinction coefficient of species B in M^-1 cm^-1.
    pathlength : float
        Path length of the sample in cm (e.g., the distance light travels through the sample.

    Returns
    -------
    float
        Number of excitations per A per second.
        '''
    
    concentration_A_M = concentration_A * 1e-6  # Convert from uM to M
    concentration_B_M = concentration_B * 1e-6  # Convert from uM to M
    volume_L = (pathlength * 1) / 1000 # Assuming a unit area (1 cm2) for simplicity, converting from cm3 to L
    photon_flux_mol = photon_flux / AVOGADRO_NUMBER  # Convert photon flux to mol/s

    mu_A = concentration_A_M * extinction_coefficient_A
    mu_B = concentration_B_M * extinction_coefficient_B

    if mu_A + mu_B > 0:
        fractional_absorbance_A = mu_A / (mu_A + mu_B)  # Fraction of total absorbance due to A
    else:
        fractional_absorbance_A = 0

    absorbance_total = (mu_A + mu_B) * pathlength  # Total absorbance
    absorbed_fraction = 1 - 10**(-absorbance_total)  # Fraction of photons absorbed

    if concentration_A_M > 0:
        excitations_per_A = (photon_flux_mol * absorbed_fraction * fractional_absorbance_A) / (volume_L * concentration_A_M)
    else:
        excitations_per_A = 0

    return excitations_per_A

def calculate_excitations_per_second_multi_competing_fast(species_of_interest,
                                                          photon_flux,
                                                          pathlength,
                                                          concentration_unit = 'uM',
                                                          concentrations=None,
                                                          extinction_coefficients=None,
                                                          **kwargs):
    """
    Calculate the number of excitations per species per second for multiple competing species.
    
    This is a fast implementation using pure Python dictionaries (no NumPy overhead).
    Supports flexible input via pre-structured dictionaries, prefixed kwargs, or both.
    
    Parameters
    ----------
    species_of_interest : str
        The species for which to calculate excitations.
    photon_flux : float
        Photon flux in photons cm^-2 s^-1.
    pathlength : float
        Path length of the sample in cm.
    concentrations : dict, optional
        Dictionary mapping species names to concentrations in micromolar (uM).
    extinction_coefficients : dict, optional
        Dictionary mapping species names to extinction coefficients in M^-1 cm^-1.
    **kwargs : additional keyword arguments
        Additional concentrations can be passed as ``concentration_<species>=value``.
        Additional extinction coefficients as ``extinction_coefficient_<species>=value``.
        These are merged into the respective dictionaries (kwargs override existing keys).
    
    Returns
    -------
    float
        Number of excitations per species of interest per second.
    
    Examples
    --------
    Using only dictionaries:
    
    >>> concentrations = {'A': 10, 'B': 20}
    >>> extinction_coefficients = {'A': 5000, 'B': 300}
    >>> calculate_excitations_per_second_multi_competing_fast(
    ...     'A', photon_flux=1e17, pathlength=2.5,
    ...     concentrations=concentrations,
    ...     extinction_coefficients=extinction_coefficients
    ... )
    
    Using only kwargs:
    
    >>> calculate_excitations_per_second_multi_competing_fast(
    ...     'A', photon_flux=1e17, pathlength=2.5,
    ...     concentration_A=10, concentration_B=20,
    ...     extinction_coefficient_A=5000, extinction_coefficient_B=300
    ... )
    
    Mixed usage (base dicts extended with kwargs):
    
    >>> calculate_excitations_per_second_multi_competing_fast(
    ...     'A', photon_flux=1e17, pathlength=2.5,
    ...     concentrations={'A': 10, 'B': 20},
    ...     extinction_coefficients={'A': 5000, 'B': 300},
    ...     concentration_C=30,
    ...     extinction_coefficient_C=1500
    ... )
    """
    # Concentration conversion factors, from supplied unit to M
    concentration_conversion_factors = {
        'uM': 1e-6,  # micromolar to M
        'mM': 1e-3,  # millimolar to M
        'M': 1.0      # molar to M
    }
    if concentration_unit not in concentration_conversion_factors:
        raise ValueError(f"Unsupported concentration unit '{concentration_unit}'. Supported units: {list(concentration_conversion_factors.keys())}")

    conversion_factor = concentration_conversion_factors[concentration_unit]

    # Initialize dicts if not provided
    concentrations = dict(concentrations) if concentrations else {}
    extinction_coefficients = dict(extinction_coefficients) if extinction_coefficients else {}
    
    # Extract and merge kwargs by prefix
    for key, value in kwargs.items():
        if key.startswith('concentration_'):
            species = key[len('concentration_'):]  # Strip prefix
            concentrations[species] = value
        elif key.startswith('extinction_coefficient_'):
            species = key[len('extinction_coefficient_'):]  # Strip prefix
            extinction_coefficients[species] = value
    
    volume_L = (pathlength * 1) / 1000 # Assuming a unit area (1 cm2) for simplicity, converting from cm3 to L
    photon_flux_mol = photon_flux / AVOGADRO_NUMBER  # Convert photon flux to mol/s

    concentrations_M = {species: conc * conversion_factor for species, conc in concentrations.items()}  # Convert to M
    mu_values = {species: concentrations_M[species] * extinction_coefficients[species] 
                 for species in concentrations.keys()}

    mu_total = sum(mu_values.values())

    if mu_total > 0:
        fractional_mu = mu_values[species_of_interest] / mu_total
    else:
        fractional_mu = 0
    
    absorbance_total = mu_total * pathlength
    absorbed_fraction = 1 - 10**(-absorbance_total)  # Fraction of photons absorbed

    if concentrations_M[species_of_interest] > 0:
        excitations_per_species = (photon_flux_mol * absorbed_fraction * fractional_mu) / (volume_L * concentrations_M[species_of_interest])
    else:
        excitations_per_species = 0
    
    return excitations_per_species


def calculate_excitations_per_second_multi_competing(species_of_interest,
                                                     photon_flux,
                                                     concentrations,
                                                     extinction_coefficients,
                                                     pathlength,
                                                     concentration_unit = 'uM',
                                                     return_full = False):
    
    """
    Calculate the number of excitations per species per second for multiple competing species.
    
    This implementation uses NumPy arrays for vectorized calculations, which is
    beneficial for large numbers of species or when full absorption data is needed.
    
    Parameters
    ----------
    species_of_interest : str
        The species for which to calculate excitations.
    photon_flux : float
        Photon flux in photons cm^-2 s^-1.
    concentrations : dict
        Dictionary mapping species names to concentrations in micromolar (uM).
    extinction_coefficients : dict
        Dictionary mapping species names to extinction coefficients in M^-1 cm^-1.
        Must contain the same keys as `concentrations`.
    pathlength : float
        Path length of the sample in cm.
    return_full : bool, optional
        If False (default), returns only the excitations for the species of interest.
        If True, returns detailed absorption data for all species.
    
    Returns
    -------
    float or tuple
        If ``return_full=False``:
            Number of excitations per species of interest per second.
        If ``return_full=True``:
            A tuple containing:
            - excitations_per_species_dict : dict
                Dictionary mapping species names to excitations per second.
            - absorbed_dict : dict
                Dictionary mapping species names to fraction of light absorbed,
                plus a 'transmitted' key with the fraction of light transmitted.
    
    Raises
    ------
    ValueError
        If `concentrations` and `extinction_coefficients` do not have the same keys.
    
    Examples
    --------
    Basic usage:
    
    >>> concentrations = {'A': 10, 'B': 20, 'C': 5}
    >>> extinction_coefficients = {'A': 5000, 'B': 300, 'C': 1500}
    >>> excitations = calculate_excitations_per_second_multi_competing(
    ...     'A', photon_flux=1e17,
    ...     concentrations=concentrations,
    ...     extinction_coefficients=extinction_coefficients,
    ...     pathlength=2.5
    ... )
    
    Getting full absorption data:
    
    >>> excitations_dict, absorbed_dict = calculate_excitations_per_second_multi_competing(
    ...     'A', photon_flux=1e17,
    ...     concentrations=concentrations,
    ...     extinction_coefficients=extinction_coefficients,
    ...     pathlength=2.5,
    ...     return_full=True
    ... )
    >>> absorbed_dict['transmitted']  # Fraction of light transmitted through sample
    """

    # Concentration conversion factors, from supplied unit to M
    concentration_conversion_factors = {
        'uM': 1e-6,  # micromolar to M
        'mM': 1e-3,  # millimolar to M
        'M': 1.0      # molar to M
    }
    if concentration_unit not in concentration_conversion_factors:
        raise ValueError(f"Unsupported concentration unit '{concentration_unit}'. Supported units: {list(concentration_conversion_factors.keys())}")

    conversion_factor = concentration_conversion_factors[concentration_unit]

    # Calculate volume and photon flux in mol/s
    volume_L = (pathlength * 1) / 1000 # Assuming a unit area (1 cm2) for simplicity, converting from cm3 to L  
    photon_flux_mol = photon_flux / AVOGADRO_NUMBER  # Convert photon flux to mol/s

    # Validate that both dictionaries have the same species
    if set(concentrations.keys()) != set(extinction_coefficients.keys()):
        raise ValueError("Concentrations and extinction coefficients must have the same species.")

    # Create consistent ordering and index mapping
    species_list = sorted(concentrations.keys())
    species_to_index = {species: i for i, species in enumerate(species_list)}

    # Convert to numpy arrays with consistent order
    concentrations_array = np.array([concentrations[species] for species in species_list])
    concentrations_array_M = concentrations_array * conversion_factor # Convert to M
    extinction_coefficients_array = np.array([extinction_coefficients[species] for species in species_list])

    # Calculate absorption
    mu_values = concentrations_array_M * extinction_coefficients_array
    mu_total = np.sum(mu_values)

    fractional_mu = mu_values / mu_total if mu_total > 0 else np.zeros_like(mu_values) # avoid division by zero

    absorbance_total = mu_total * pathlength
    transmitted = 10**(-absorbance_total) # Fraction of photons transmitted
    absorbed_fraction = 1 - transmitted  # Fraction of photons absorbed

    absorbed_by_species = absorbed_fraction * fractional_mu

    # Calculate excitations per species
    mol_photons_absorbed_by_species = photon_flux_mol * absorbed_by_species
    mol_species = volume_L * concentrations_array_M

    excitations_per_species = np.divide(mol_photons_absorbed_by_species, 
                                        mol_species, 
                                        out=np.zeros_like(mol_photons_absorbed_by_species), 
                                        where=mol_species != 0) # avoid division by zero for species with zero concentration

    if return_full is False:
        return excitations_per_species[species_to_index[species_of_interest]]
    
    else:
        excitations_per_species_dict = dict(zip(species_list, excitations_per_species))
        absorbed_dict = dict(zip(species_list, absorbed_by_species))
        absorbed_dict['transmitted'] = transmitted
 
        return excitations_per_species_dict, absorbed_dict




def test_function():

    photon_flux = 1000e17  # photons cm^-2 s^-1


    # construct a concentrations dict with 100 entries, using letters A to Z repeatedly (appending 1, 2, 3 etc.)
    # import string
    # letters = string.ascii_uppercase
    # concentrations = {f"{letters[i % 26]}{i // 26 + 1}": i + 1 for i in range(100)}

    # concentrations['A'] = 10
    # concentrations['B'] = 20

    # # construct a fitting extinction_coefficients dict with 100 entries, using letters A to Z repeatedly (appending 1, 2, 3 etc.)
    # extinction_coefficients = {f"{letters[i % 26]}{i // 26 + 1}": (i + 1) * 100 for i in range(100)}
    # extinction_coefficients['A'] = 5000
    # extinction_coefficients['B'] = 300

    concentrations = {'A': 1,
                      'B': 2}
    
    extinction_coefficients = {'A': 5000,
                               'B': 300}

    # concentrations = {'A': 1,
    #                   'B': 2,
    #                   'C': 3,
    #                   'D': 4,
    #                   'E': 5,
    #                   'F': 6,
    #                   'G': 7,
    #                   'H': 8,
    #                   'I': 9,
    #                   'J': 10}
    
    # extinction_coefficients = {'A': 5000,
    #                            'B': 300,
    #                            'C': 1500,
    #                            'D': 2500,
    #                            'E': 4000,
    #                            'F': 600,
    #                            'G': 700,
    #                            'H': 800,
    #                            'I': 900,
    #                            'J': 1000}
    
    pathlength = 2.5

    from time import time

    start = time()
    for _ in range(100000):
         calculate_excitations_per_second_multi_competing('A',
                                                     photon_flux,
                                                     concentrations,
                                                     extinction_coefficients,
                                                     pathlength)
    end = time()
    print(f"Time for calculations: {end - start} seconds")


    # with cProfile.Profile() as pr:
    #     for _ in range(100000):
    #          calculate_excitations_per_second_multi_competing('A',
    #                                                      photon_flux,
    #                                                      concentrations,
    #                                                      extinction_coefficients,
    #                                                      pathlength)
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats(filename = "needs_profiling.prof")


    start = time()
    for _ in range(100000):
            calculate_excitations_per_second_competing(photon_flux,
                                                concentrations['A'],
                                                concentrations['B'],
                                                extinction_coefficients['A'],
                                                extinction_coefficients['B'],
                                                pathlength) 
    end = time()
    print(f"Time for calculations: {end - start} seconds")

    start = time()
    for _ in range(100000):
         calculate_excitations_per_second_multi_competing_fast('A',
                                                     photon_flux,
                                                     pathlength,
                                                     concentrations=concentrations,
                                                     extinction_coefficients=extinction_coefficients)
    end = time()
    print(f"Time for calculations: {end - start} seconds")

    print(calculate_excitations_per_second_multi_competing('A',
                                                     photon_flux,
                                                     concentrations,
                                                     extinction_coefficients,
                                                     pathlength))
    
    print(calculate_excitations_per_second_competing(photon_flux,
                                               concentrations['A'],
                                               concentrations['B'],
                                               extinction_coefficients['A'],
                                               extinction_coefficients['B'],
                                               pathlength))
    
    print(calculate_excitations_per_second_multi_competing_fast('A',
                                                                photon_flux,
                                                                pathlength,
                                                                concentrations = concentrations,
                                                                extinction_coefficients = extinction_coefficients,
                                                                concentration_HEYA = 50,
                                                                extinction_coefficient_HEYA = 2000
                                                           ))

    


    excitations_per_species, absorbed_dict = calculate_excitations_per_second_multi_competing('A',
                                                     photon_flux,
                                                     concentrations,
                                                     extinction_coefficients,
                                                     pathlength,
                                                     return_full=True)
    
    #pp.pprint(absorbed_dict)



    
    


if __name__ == "__main__":
    test_function()