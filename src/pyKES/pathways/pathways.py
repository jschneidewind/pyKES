import pprint as pp

from pyKES.reaction_ODE import calculate_reaction_rate, parse_reactions
from pyKES.utilities.calculate_absorption import calculate_excitations_per_second_multi_competing

def calculate_reaction_pathway_proportions(selected_species,
                                           concentrations,
                                           parsed_reactions,
                                           rate_constants,
                                           other_multipliers = {}):
    """
    Calculate the proportion/probability of a species reacting via each pathway.
    
    For a given species, this function identifies all reactions where it participates
    as a reactant and calculates the relative probability (proportion) that the species
    will react via each pathway. The proportions are normalized to sum to 1.
    
    Parameters
    ----------
    selected_species : str
        The species name (e.g., '[A]') for which to calculate reaction pathway proportions.
    concentrations : dict
        Dictionary mapping species to their concentrations.
    parsed_reactions : list
        List of parsed reaction dictionaries as returned by `parse_reactions`.
    rate_constants : dict
        Dictionary mapping rate constant identifiers (str) to their values (float).
    other_multipliers : dict, optional
        Dictionary mapping multiplier identifiers to values or function definitions.
        
    Returns
    -------
    dict
        Dictionary mapping reaction indices (int) to their proportions (float).
        Only reactions with non-zero rates are included. Returns an empty dict
        if the species does not participate as a reactant in any reaction or
        if all reaction rates are zero.
        
    Notes
    -----
    The rate calculation includes the full concentration dependence of all reactants,
    including the selected species itself. Since all rates involving the selected
    species will have its concentration as a factor, this cancels out during
    normalization, giving the correct relative proportions.
    
    Examples
    --------
    >>> reactions = ['[A] > [B], k1', '[A] + [C] > [D], k2']
    >>> parsed_reactions, species = parse_reactions(reactions)
    >>> rate_constants = {'k1': 1.0, 'k2': 2.0}
    >>> concentrations = np.array([1.0, 0.0, 5.0, 0.0])  # [A]=1, [B]=0, [C]=5, [D]=0
    >>> result = calculate_reaction_pathway_proportions(
    ...     '[A]', concentrations, species, parsed_reactions, rate_constants
    ... )
    >>> result
    {0: 0.09090909090909091, 1: 0.9090909090909091}  # k1*[A]=1, k2*[A]*[C]=10
    """

    reaction_rates = {}
    
    # Find all reactions where the selected species is a reactant and calculate rates
    for idx, reaction in enumerate(parsed_reactions):
        if selected_species in reaction['reactants']:
            rate = calculate_reaction_rate(reaction, 
                                           concentrations, 
                                           rate_constants, 
                                           other_multipliers)
            if rate > 0:
                reaction_rates[idx] = rate
    
    # If no reactions or all rates are zero, return empty dict
    if not reaction_rates:
        return {}
        
    # Normalize rates to get proportions
    total_rate = sum(reaction_rates.values())
    proportions = {idx: rate / total_rate for idx, rate in reaction_rates.items()}
    
    return proportions

def merge_propagation_trees(tree1, tree2):
    """
    Merge two propagation trees for the same species.
    
    When a species can be formed via multiple reaction pathways, this function
    merges their propagation trees by summing amounts and recursively merging
    sub-trees.
    
    Parameters
    ----------
    tree1 : dict
        First propagation tree.
    tree2 : dict
        Second propagation tree.
        
    Returns
    -------
    dict
        Merged propagation tree with combined amounts.
    """
    merged = {}
    
    # Sum the amounts
    merged['amount_formed'] = tree1.get('amount_formed', 0) + tree2.get('amount_formed', 0)
    
    # Get all species keys (excluding 'amount_formed')
    all_keys = set(tree1.keys()) | set(tree2.keys())
    all_keys.discard('amount_formed')
    
    # Recursively merge sub-trees
    for key in all_keys:
        if key in tree1 and key in tree2:
            merged[key] = merge_propagation_trees(tree1[key], tree2[key])
        elif key in tree1:
            merged[key] = tree1[key]
        else:
            merged[key] = tree2[key]
    
    return merged

def propagate_species(species,
                      amount,
                      concentrations,
                      parsed_reactions,
                      rate_constants,
                      other_multipliers,
                      ancestor_chain):
    """
    Recursively propagate a species through the reaction network.
    
    This function calculates how a species reacts further and recursively
    propagates each product through the network until termination conditions
    are met.
    
    Parameters
    ----------
    species : str
        The species name to propagate (e.g., '[A-excited]').
    amount : float
        The amount/proportion of this species being propagated.
    concentrations : dict
        Dictionary mapping species names to their concentrations.
    parsed_reactions : list
        List of parsed reaction dictionaries as returned by `parse_reactions`.
    rate_constants : dict
        Dictionary mapping rate constant identifiers to their values.
    other_multipliers : dict
        Dictionary mapping multiplier identifiers to values or function definitions.
    ancestor_chain : set
        Set of species names that have appeared in the current propagation path.
        Used to detect and prevent circular reaction sequences.
        
    Returns
    -------
    dict
        Nested dictionary containing the propagation tree for this species.
        Returns an empty dict if the species doesn't react further or
        would create a cycle.
    """
    result = {'amount_formed': amount}
    
    # Check for cycle - if this species already appeared upstream, stop propagation
    if species in ancestor_chain:
        return result
    
    # Get reaction pathway proportions for this species
    pathway_proportions = calculate_reaction_pathway_proportions(
        selected_species=species,
        concentrations=concentrations,
        parsed_reactions=parsed_reactions,
        rate_constants=rate_constants,
        other_multipliers=other_multipliers
    )
    
    # If no reactions, this is a terminal species
    if not pathway_proportions:
        return result
    
    # Add current species to ancestor chain for recursive calls
    new_ancestor_chain = ancestor_chain | {species}
    
    # Process each reaction pathway
    for reaction_idx, proportion in pathway_proportions.items():
        reaction = parsed_reactions[reaction_idx]
        
        # Get products and their stoichiometries
        for product, stoichiometry in reaction['products'].items():
            product_amount = amount * proportion * stoichiometry
            
            # Recursively propagate this product
            product_tree = propagate_species(
                species=product,
                amount=product_amount,
                concentrations=concentrations,
                parsed_reactions=parsed_reactions,
                rate_constants=rate_constants,
                other_multipliers=other_multipliers,
                ancestor_chain=new_ancestor_chain
            )
            
            # Merge product tree into result
            # If product already exists (from different reaction pathways)
            if product in result:
                result[product] = merge_propagation_trees(result[product], product_tree)
            else:
                result[product] = product_tree
    
    return result

def calculate_reaction_network_propagation(concentrations,
                                           parsed_reactions,
                                           rate_constants,
                                           absorbing_species,
                                           extinction_coefficients,
                                           photon_flux,
                                           pathlength,
                                           other_multipliers={}):
    """
    Calculate the full propagation through a photochemical reaction network.
    
    This function models how light absorption leads to excited state formation
    and subsequent reactions through the network. It calculates:
    1. Light absorption fractions for each absorbing species
    2. Formation of excited states from light absorption
    3. Recursive propagation of products through the reaction network
    
    Parameters
    ----------
    concentrations : dict
        Dictionary mapping species names to their concentrations in micromolar (uM).
        Should contain all species in the reaction network.
    parsed_reactions : list
        List of parsed reaction dictionaries as returned by `parse_reactions`.
    rate_constants : dict
        Dictionary mapping rate constant identifiers to their values.
    absorbing_species : dict
        Dictionary mapping ground state species names to their excited state names.
        Example: {'[A]': '[A-excited]', '[B]': '[B-excited]'}
    extinction_coefficients : dict
        Dictionary mapping absorbing species names to their extinction coefficients
        in M^-1 cm^-1. Keys should match the keys in `absorbing_species`.
        Example: {'[A]': 8500, '[B]': 5400}
    photon_flux : float
        Photon flux in photons cm^-2 s^-1.
    pathlength : float
        Path length of the sample in cm.
    other_multipliers : dict, optional
        Dictionary mapping multiplier identifiers to values or function definitions.
        
    Returns
    -------
    dict
        Nested dictionary containing the full propagation tree starting from
        light absorption. Structure:
        {
            'Light absorption': {
                '[A]': {
                    'absorbed': 0.3,
                    '[A-excited]': {
                        'amount_formed': 0.3,
                        '[B]': {
                            'amount_formed': 0.15,
                            ...
                        },
                        ...
                    }
                },
                '[B]': {...},
                'transmitted': 0.1
            }
        }
        
    Examples
    --------
    >>> reactions = ['[A] > [A-excited], k1 ; hv_functionA',
    ...              '[A-excited] > [A], k2',
    ...              '[A-excited] > [B], k3']
    >>> parsed_reactions, species = parse_reactions(reactions)
    >>> concentrations = {'[A]': 10.0, '[A-excited]': 0.0, '[B]': 0.0}
    >>> absorbing_species = {'[A]': '[A-excited]'}
    >>> extinction_coefficients = {'[A]': 8500}
    >>> result = calculate_reaction_network_propagation(
    ...     concentrations=concentrations,
    ...     parsed_reactions=parsed_reactions,
    ...     rate_constants={'k1': 1, 'k2': 3e8, 'k3': 1e8},
    ...     absorbing_species=absorbing_species,
    ...     extinction_coefficients=extinction_coefficients,
    ...     photon_flux=1e17,
    ...     pathlength=2.25
    ... )
    """
    # Filter concentrations to only include absorbing species
    absorbing_concentrations = {
        species: concentrations.get(species, 0) 
        for species in absorbing_species.keys()
    }
    
    # Calculate light absorption using the multi-competing function
    _, absorbed_dict = calculate_excitations_per_second_multi_competing(
        species_of_interest=list(absorbing_species.keys())[0],  # Required but we use return_full
        photon_flux=photon_flux,
        concentrations=absorbing_concentrations,
        extinction_coefficients=extinction_coefficients,
        pathlength=pathlength,
        return_full=True
    )
    
    # Build the result structure
    result = {'Light absorption': {}}
    
    # Process each absorbing species
    for ground_state, excited_state in absorbing_species.items():
        absorbed_fraction = absorbed_dict.get(ground_state, 0)
        
        if absorbed_fraction > 0:
            # Propagate from the excited state
            excited_tree = propagate_species(
                species=excited_state,
                amount=absorbed_fraction,
                concentrations=concentrations,
                parsed_reactions=parsed_reactions,
                rate_constants=rate_constants,
                other_multipliers=other_multipliers,
                ancestor_chain={ground_state}  # Include ground state to prevent immediate back-reaction counting as cycle
            )
            
            result['Light absorption'][ground_state] = {
                'absorbed': absorbed_fraction,
                excited_state: excited_tree
            }
    
    # Add transmitted fraction
    result['Light absorption']['transmitted'] = absorbed_dict.get('transmitted', 0)
    
    return result


def testing_function():

    reactions = ['[A] > [B], k1',
                 '[A] + [C] > [D], k2',
                 '[A] + [A] > [E], k3']
    
    parsed_reactions, species = parse_reactions(reactions)

    concentrations = {'[A]': 0.1,
                      '[C]': 0.5}
    
    rate_constants = {'k1': 1.0,
                      'k2': 2.0,
                      'k3': 10.0}
    
    proportions = calculate_reaction_pathway_proportions('[A]',
                                                         concentrations,
                                                         parsed_reactions,
                                                         rate_constants)
    
    print(parsed_reactions[1])
    
    print(proportions)

def testing_propagation():

    PHOTON_FLUX = 1e17
    PATHLENGTH = 2.25

    A_EXTINCTION_COEFFICIENT = 8500
    B_EXTINCTION_COEFFICIENT = 5400
    C_EXTINCTION_COEFFICIENT = 1000

    extinction_coefficients = {'[A]': A_EXTINCTION_COEFFICIENT,
                               '[B]': B_EXTINCTION_COEFFICIENT,
                               '[C]': C_EXTINCTION_COEFFICIENT}

    reactions = ['[A] > [A-excited], k1',
                 '[A-excited] > [A], k2',
                 '[A-excited] > [B], k3',
                 '[B] > [A], k4',
                 '[B] > [B-excited], k5',
                 '[B-excited] > [B], k6',
                 '[B-excited] > [C], k7',]
    
    parsed_reactions, species = parse_reactions(reactions)
    
    rate_constants = {'k1': 1,
                      'k2': 3e8,
                      'k3': 1e8,
                      'k4': 1e2,
                      'k5': 1,
                      'k6': 2e8,
                      'k7': 3.3e8}
    
    concentrations = {'[A]': 5.0,
                      '[A-excited]': 0.001,
                      '[B]': 5.0,
                      '[B-excited]': 0.001,
                      '[C]': 1.0}
    
    absorbing_species = {'[A]': '[A-excited]',
                         '[B]': '[B-excited]',
                         '[C]': '[C-excited]'}
    
    excitations, absorbed = calculate_excitations_per_second_multi_competing(
        species_of_interest='[A]',
        photon_flux=PHOTON_FLUX,
        concentrations={k: concentrations[k] for k in absorbing_species.keys()},
        extinction_coefficients=extinction_coefficients,
        pathlength=PATHLENGTH,
        return_full=True
    )

    proportions = calculate_reaction_pathway_proportions('[B]',
                                                         concentrations,
                                                         parsed_reactions,
                                                         rate_constants)
    
    pp.pprint(proportions)

    network_propagation = calculate_reaction_network_propagation(
        concentrations=concentrations,
        parsed_reactions=parsed_reactions,
        rate_constants=rate_constants,
        absorbing_species=absorbing_species,
        extinction_coefficients=extinction_coefficients,
        photon_flux=PHOTON_FLUX,
        pathlength=PATHLENGTH)
    
    pp.pprint(network_propagation)
                        
def test_propagate_species_function():

    # Merging works correctly when comparing these two mechanism (two
    # pathways that both lead to [B])
    # reactions = ['[A] > [B] + 2 [D], k1',
    #              '[A] + [A] > [C], k2',
    #              ]
    # reactions = ['[A] > [B] + 2 [D], k1',
    #              '[A] + [A] > [B], k2',
    #              ]

    # Works correctly, [D] is formed at different points, but this is handled correctly (not merged)
    # reactions = ['[A] > [B] + 2 [D], k1',
    #              '[B] > [D], k2',
    #              ]
    
    # Additional test, works correctly, not merged
    # reactions = ['[A] > [B], k1',
    #              '[A] > [C], k2',
    #              '[B] > [D], k3',
    #              '[C] > [D], k4',
    #              '[D] > [F], k5',
    #             ]
    
    # Two pathways to [B], merging makes sense:
    # recursive merging is needed to updates amounts downstream correctly
    # when additional [B] is formed from [A] via second pathway
    # since algorithm goes depth-first
    reactions = ['[A] > [B] + [Marker-0], k1',
                 '[A] + [B] > 2 [B] + [Marker-1], k2',
                 '[B] > [C] + [Marker-2], k3',
                 '[B] > [D] + [Marker-3], k4',
                 '[C] > [E] + [Marker-4], k5',
                ]
                
    
    parsed_reactions, species = parse_reactions(reactions)

    concentrations = {'[A]': 1.0,
                      '[B]': 1.0,
                      '[C]': 1e-9,
                      '[D]': 1e-9}
    
    rate_constants = {'k1': 1.0,
                      'k2': 1.0,
                      'k3': 1.0,
                      'k4': 1.0,
                      'k5': 1.0}


    propagation_tree = propagate_species(
        species='[A]',
        amount=1.0,
        concentrations=concentrations,
        parsed_reactions=parsed_reactions,
        rate_constants=rate_constants,
        other_multipliers={},
        ancestor_chain=set()
    )

    pp.pprint(propagation_tree)





  



if __name__ == '__main__':
    #testing_function()
    testing_propagation()
    #test_propagate_species_function()