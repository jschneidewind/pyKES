import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def parse_reactions(reactions):
    '''
    Parse a list of chemical reaction strings into structured dictionaries.
    
    This function processes reaction strings with the format:
    "[A] + 2 [B] > [C] + [D], k1 ; hv1, sigma1"
    
    Where:
    - Chemical species are enclosed in square brackets
    - Stoichiometric coefficients can be integers or decimals (e.g., 0.5 [A])
    - Reactants and products are separated by ">"
    - The rate constant identifier follows after a comma
    - Optional additional parameters follow a semicolon and are comma-separated
    
    Parameters
    ----------
    reactions : list of str
        List of reaction strings to parse.
        
    Returns
    -------
    tuple
        A tuple containing:
        - parsed_reactions: list of dict
            Each dictionary contains:
            * 'reactants': dict mapping species to stoichiometric coefficients
            * 'products': dict mapping species to stoichiometric coefficients
            * 'rate_constant': str, identifier of the rate constant
            * 'other_multipliers': list of str, optional parameters for the reaction
        - sorted_species: list
            Alphabetically sorted list of all unique chemical species in the reaction network
            
    Examples
    --------
    >>> reactions = ['[A] + 2 [B] > [C], k1', '[C] > [A] + [B], k2 ; hv1, sigma1']
    >>> parsed, species = parse_reactions(reactions)
    >>> parsed
    [{'reactants': {'[A]': 1.0, '[B]': 2.0}, 'products': {'[C]': 1.0}, 
      'rate_constant': 'k1', 'other_multipliers': []},
     {'reactants': {'[C]': 1.0}, 'products': {'[A]': 1.0, '[B]': 1.0}, 
      'rate_constant': 'k2', 'other_multipliers': ['hv1', 'sigma1']}]
    >>> species
    ['[A]', '[B]', '[C]']'''

    parsed_reactions = []
    species_set = set()
    
    for reaction in reactions:
        reaction_dict = {'reactants': {}, 'products': {}, 'rate_constant': '', 'other_multipliers': []}
        
        # Split the reaction into main components
        reaction_part, rate_part = reaction.split(',', 1)
        rate_details = [x.strip() for x in rate_part.split(';')]

        reaction_dict['rate_constant'] = rate_details[0]  # First element is the rate constant
       
        if len(rate_details) > 1:
            reaction_dict['other_multipliers'] = [item.strip() for item in rate_details[1].split(',')]
        
        # Split reactants and products
        reactants_str, products_str = reaction_part.split('>')
        
        def parse_species(side):
            species_count = defaultdict(float)
            species_matches = re.findall(r'(?:([\d\.]+)\s*)?(\[[^\]]+\])', side)

            for count, species in species_matches:
                count = float(count) if count else 1.0
                species_count[species] += count
                species_set.add(species)
            
            return dict(species_count)
        
        reaction_dict['reactants'] = parse_species(reactants_str)
        reaction_dict['products'] = parse_species(products_str)
        
        parsed_reactions.append(reaction_dict)
    
    return parsed_reactions, sorted(species_set)

def build_ode_system(parsed_reactions, species, rate_constants, other_multipliers = {}):
    """
    Build the system of ordinary differential equations.
    
    Parameters:
    -----------
    parsed_reactions : list
        List of dictionaries with keys 'reactants', 'products', 'rate_constant', and optional 'photon_flux', 'sigma'
    species : list
        Sorted list of all unique chemical species
    rate_constants : dict
        Dictionary mapping rate constant identifiers to values
    other_multipliers : dict, optional
        Dictionary mapping other multipliers to values
        
    Returns:
    --------
    function
        A function that computes the derivatives for each species.
    """
    
    def ode_system(y, t):
        """
        Compute derivatives for each species.
        
        Parameters:
        -----------
        y : array-like
            Current concentrations of each species.
        t : float
            Current time (not used explicitly in autonomous systems).
            
        Returns:
        --------
        array-like
            Derivatives for each species.
        """
        dydt = np.zeros(len(species))
        
        # Create a dictionary mapping species to their current concentrations
        conc = {spec: y[i] for i, spec in enumerate(species)}
        
        # Compute contribution from each reaction
        for reaction in parsed_reactions:

            # Start with base rate constant
            rate = rate_constants[reaction['rate_constant']]

            # Apply other multipliers if present
            for multiplier in reaction['other_multipliers']:
                mult = other_multipliers[multiplier]

                # If the multiplier is a function, resolve its arguments, call it, and multiply the rate
                if isinstance(mult, dict) and 'function' in mult:
                    arguments = mult['arguments']  
                    # resolve sources into keyword args
                    kwargs = {}
                    for parameter, source in arguments.items():
                        if source in conc:
                            kwargs[parameter] = conc[source]
                        elif source in other_multipliers and not isinstance(other_multipliers[source], dict):
                            kwargs[parameter] = other_multipliers[source]
                        elif source in rate_constants:
                            kwargs[parameter] = rate_constants[source]
                        else:
                            raise KeyError(f"Cannot resolve argument source '{source}' for multiplier '{multiplier}'")
                    rate *= mult['function'](**kwargs)

                # If the multiplier is a number, multiply directly
                else:
                    rate *= mult
            
            # Calculate concentration-dependent rate
            for reactant, stoich in reaction['reactants'].items():
                rate *= conc[reactant] ** stoich
            
            # Update derivatives for reactants (consumption)
            for reactant, stoich in reaction['reactants'].items():
                idx = species.index(reactant)
                dydt[idx] -= stoich * rate
            
            # Update derivatives for products (production)
            for product, stoich in reaction['products'].items():
                idx = species.index(product)
                dydt[idx] += stoich * rate
        
        return dydt
    
    return ode_system

def solve_ode_system(parsed_reactions, 
                     species, 
                     rate_constants, 
                     initial_conditions, 
                     times, 
                     other_multipliers = {}):
    """
    Solve the system of ODEs.
    
    Parameters:
    -----------
    parsed_reactions : list
        List of dictionaries with keys 'reactants', 'products', 'rate_constant', and optional 'photon_flux', 'sigma'
    species : list
        Sorted list of all unique chemical species
    rate_constants : dict
        Dictionary mapping rate constant identifiers to values
    initial_conditions : dict
        Dictionary mapping species to their initial concentrations
    times : array-like
        Time points at which to solve the ODEs
    other_multipliers : dict, optional
        Dictionary mapping other multipliers to values
        
    Returns:
    --------
    array-like
        Solution array with shape (len(times), len(species)).
    """

    y0 = np.zeros(len(species))  # Initial concentrations default to zero

    for spec, conc in initial_conditions.items():
        if spec in species:
            idx = species.index(spec)
            y0[idx] = conc
        else:
            print(f'Warning: {spec} not in species list')
    
    # Build ODE system
    ode_system = build_ode_system(parsed_reactions, species, rate_constants, other_multipliers)
    
    # Solve ODEs
    solution = odeint(ode_system, y0, times, 
                        rtol=1e-8, atol=1e-10,  
                        mxstep=5000)           
    
    return solution

def plot_solution(species, times, solution, exclude_species = [], ax = None):
    """
    Plot the solution.
    
    Parameters:
    -----------
    species : list
        Sorted list of all unique chemical species
    times : array-like
        Time points at which the ODEs were solved
    solution : array-like
        Solution array with shape (len(times), len(species))
    exclude_species : list, optional
        List of species names to exclude from plotting. Default is None.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    
    for i, spec in enumerate(species):
        if spec not in exclude_species:
            ax.plot(times, solution[:, i], label=spec)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title('Chemical Reaction Network Dynamics')
    ax.legend()
    ax.grid(True)

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

    AVOGADRO_NUMBER = 6.022e23  # Avogadro's number in mol^-1

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
    
    AVOGADRO_NUMBER = 6.022e23  # Avogadro's number in mol^-1

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