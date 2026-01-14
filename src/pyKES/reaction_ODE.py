import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from pyKES.utilities.calculate_absorption import calculate_excitations_per_second_competing, calculate_excitations_per_second_multi_competing_fast

def parse_species(side, species_set):
    """
    Parse one side of a chemical reaction string to extract species and their stoichiometric coefficients.
    
    This function processes a string representing either the reactants or products
    side of a chemical reaction, extracting species names (enclosed in square brackets)
    and their stoichiometric coefficients.
    
    Parameters
    ----------
    side : str
        A string representing one side of a reaction equation.
        Species should be enclosed in square brackets, optionally preceded by
        a stoichiometric coefficient (integer or decimal).
        Example: "2 [A] + 0.5 [B]"
    species_set : set
        A set to which discovered species names will be added.
        This set is modified in place and also returned.
        
    Returns
    -------
    tuple
        A tuple containing:
        - species_count : dict
            Dictionary mapping species names (str) to their stoichiometric
            coefficients (float). Repeated species are summed.
        - species_set : set
            The updated set containing all unique species names encountered.
            
    Examples
    --------
    >>> species_set = set()
    >>> species_count, species_set = parse_species("2 [A] + [B]", species_set)
    >>> species_count
    {'[A]': 2.0, '[B]': 1.0}
    >>> species_set
    {'[A]', '[B]'}
    
    >>> species_set = {'[C]'}
    >>> species_count, species_set = parse_species("[A] + [A]", species_set)
    >>> species_count
    {'[A]': 2.0}
    >>> species_set
    {'[A]', '[C]'}
    """

    species_count = defaultdict(float)
    species_matches = re.findall(r'(?:([\d\.]+)\s*)?(\[[^\]]+\])', side)

    for count, species in species_matches:
        count = float(count) if count else 1.0
        species_count[species] += count
        species_set.add(species)
    
    return dict(species_count), species_set


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
                
        reaction_dict['reactants'], species_set = parse_species(reactants_str, species_set)
        reaction_dict['products'], species_set = parse_species(products_str, species_set)
        
        parsed_reactions.append(reaction_dict)
    
    return parsed_reactions, sorted(species_set)


def resolve_other_multipliers(other_multipliers,
                              multiplier, 
                              concentrations, 
                              rate_constants):
    """
    Resolve a multiplier value by looking it up and optionally calling it as a function.

    This function handles two types of multipliers:
    1. Simple numeric values that are returned directly
    2. Function-based multipliers defined as dictionaries with a 'function' key and
    an 'arguments' dictionary that maps parameter names to source identifiers

    For function-based multipliers, argument sources are resolved in the following
    priority order: current concentrations, other multipliers (non-dict values),
    then rate constants.

    Parameters
    ----------
    other_multipliers : dict
        Dictionary containing all multiplier definitions. Values can be either
        numeric (float/int) or dictionaries with keys:
        * 'function': callable, the function to invoke
        * 'arguments': dict mapping parameter names to source identifiers
    multiplier : str
        The key identifying which multiplier to resolve from `other_multipliers`.
    concentrations : dict
        Dictionary mapping species names (str) to their current concentrations (float).
    rate_constants : dict
        Dictionary mapping rate constant identifiers (str) to their values (float).

    Returns
    -------
    float
        The resolved multiplier value, either retrieved directly or computed
        by calling the specified function with resolved arguments.

    Raises
    ------
    KeyError
        If an argument source string cannot be found in `concentrations`, `other_multipliers`,
        or `rate_constants`.

    Examples
    --------
    >>> other_multipliers = {'photon_flux': 1e17, 
                             'hv_func': {
    ...                             'function': lambda x, y: x * y,
    ...                             'arguments': {'x': 'photon_flux', 
                                                  'y': '[RuII]'}
    ...                                   }
                                }
    >>> concentrations = {'[RuII]': 10.0}
    >>> resolve_other_multipliers(other_multipliers, 'photon_flux', concentrations, {})
    1e17
    >>> resolve_other_multipliers(other_multipliers, 'hv_func', concentrations, {})
    1e18
    """

    multiplier_resolved = other_multipliers[multiplier]

    # If the multiplier is a function, resolve its arguments, call it, and multiply the rate
    if isinstance(multiplier_resolved, dict) and 'function' in multiplier_resolved:
        arguments = multiplier_resolved['arguments']  

        # resolve sources into keyword args
        kwargs = {}

        for parameter, source in arguments.items():
            if source in concentrations:
                kwargs[parameter] = concentrations[source]
            elif source in other_multipliers and not isinstance(other_multipliers[source], dict):
                kwargs[parameter] = other_multipliers[source]
            elif source in rate_constants:
                kwargs[parameter] = rate_constants[source]
            else:
                raise KeyError(f"Cannot resolve argument source '{source}' for multiplier '{multiplier}'")
            
        multiplier_value = multiplier_resolved['function'](**kwargs)

    # If the multiplier is a number, multiply directly
    else:
        multiplier_value = multiplier_resolved

    return multiplier_value


def calculate_reaction_rate(reaction, 
                            concentrations, 
                            rate_constants, 
                            other_multipliers = {}):
    """
    Calculate the rate of a single reaction given current concentrations.
    
    This function computes the reaction rate by multiplying the base rate constant
    with any additional multipliers and the concentration-dependent terms from
    the reactants.
    
    Parameters
    ----------
    reaction : dict
        A parsed reaction dictionary containing:
        - 'rate_constant': str, identifier for the rate constant
        - 'reactants': dict mapping species names to stoichiometric coefficients
        - 'other_multipliers': list of str, identifiers for additional multipliers
    concentrations : dict
        Dictionary mapping species names (str) to their current concentrations (float).
    rate_constants : dict
        Dictionary mapping rate constant identifiers (str) to their values (float).
    other_multipliers : dict, optional
        Dictionary mapping multiplier identifiers to values or function definitions.
        See `resolve_other_multipliers` for the expected format of function-based
        multipliers.
        
    Returns
    -------
    float
        The calculated reaction rate.
        
    Examples
    --------
    >>> reaction = {'reactants': {'[A]': 1.0, '[B]': 2.0}, 
    ...             'rate_constant': 'k1', 
    ...             'other_multipliers': []}
    >>> concentrations = {'[A]': 1.0, '[B]': 2.0}
    >>> rate_constants = {'k1': 0.5}
    >>> calculate_reaction_rate(reaction, concentrations, rate_constants)
    2.0  # 0.5 * 1.0^1 * 2.0^2 = 0.5 * 1 * 4 = 2.0
    """
    # Start with base rate constant
    rate = rate_constants[reaction['rate_constant']]

    # Apply other multipliers if present
    for multiplier in reaction['other_multipliers']:
        multiplier_value = resolve_other_multipliers(other_multipliers,
                                                     multiplier,
                                                     concentrations,
                                                     rate_constants)
        rate *= multiplier_value
    
    # Calculate concentration-dependent rate
    for reactant, stoichiometry in reaction['reactants'].items():
        rate *= concentrations[reactant] ** stoichiometry
    
    return rate

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
        concentrations = {species: y[i] for i, species in enumerate(species)}
        
        # Compute contribution from each reaction
        for reaction in parsed_reactions:
            # Calculate rate using the extracted function
            rate = calculate_reaction_rate(reaction, 
                                           concentrations, 
                                           rate_constants, 
                                           other_multipliers)
            
            # Update derivatives for reactants (consumption)
            for reactant, stoichiometry in reaction['reactants'].items():
                idx = species.index(reactant)
                dydt[idx] -= stoichiometry * rate
            
            # Update derivatives for products (production)
            for product, stoichiometry in reaction['products'].items():
                idx = species.index(product)
                dydt[idx] += stoichiometry * rate
        
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

    # Set initial concentrations
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



def test_function():
    
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

    parsed_reactions, species = parse_reactions(reactions)

    solution = solve_ode_system(parsed_reactions, 
                                species, 
                                rate_constants,
                                initial_conditions,
                                times,
                                other_multipliers)
    
    plot_solution(species, times, solution, exclude_species = ['[S2O8]', '[SO4]'])

    print(solution[-1][species.index('[O2]')])

    plt.show()


    
    

if __name__ == "__main__":
    test_function()