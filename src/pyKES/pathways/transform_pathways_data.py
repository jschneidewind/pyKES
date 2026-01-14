import numpy as np
import pprint as pp
from functools import partial

from pyKES.plotting.plotting_pathways_transformed import plot_pathway_bars

def floor_to_magnitude(value):
    """Round down to the next lowest order of magnitude."""
    return 10 ** np.floor(np.log10(value))

def get_all_values(data):
    """Yield all numeric values from a nested dictionary."""
    for value in data.values():
        if isinstance(value, dict):
            yield from get_all_values(value)
        elif isinstance(value, (int, float, np.floating)):
            yield value

def transform_log_normalized(value, min_expected=0.001, max_expected=1.0):
    """Normalize log-transformed values to [0, 1] range."""

    log_val = np.log10(value)
    log_min = np.log10(min_expected)
    log_max = np.log10(max_expected)
    return (log_val - log_min) / (log_max - log_min)

def adjust_fanning_factor(level, fanning_factor, assumed_branching_degree, max_level):

    multiplier = assumed_branching_degree ** (max_level - level - 1)
    fanning_factor_adjusted = fanning_factor * multiplier

    return fanning_factor_adjusted

def check_in_history(name: str, history: list) -> tuple[bool, str | None]:
    """
    Check if a name appears in the history list.
    
    Parameters
    ----------
    name : str
        The name to search for (e.g., '[B]').
    history : list
        List of history entries with depth suffix (e.g., ['[A]_1', '[B]_3']).
    
    Returns
    -------
    tuple[bool, str | None]
        (True, last_matching_entry) if found, (False, None) otherwise.
    """
    last_match = None
    
    for item in history:
        # Extract the species name in square brackets
        base_name = item.split('~')[0]

        if base_name == name:
            last_match = item
    
    if last_match is not None:
        return True, last_match
    
    return False, None

def transform_pathways_data(data, 
                            parent = None,
                            depth_level = 0,
                            history = None,
                            min_expected = None):
    
    if history is None:
        history = []

    nodes = {}
    links = []

    # Auto-compute min_expected on first call (depth_level == 0)
    if min_expected is None and depth_level == 0:
        min_expected = min(get_all_values(data))
        min_expected = floor_to_magnitude(min_expected)

    log_normalize = partial(transform_log_normalized, min_expected=min_expected)

    metadata_keys = {'amount_formed', 'absorbed'}

    for key, value in data.items():

        # Skip metadata keys
        if key in metadata_keys:
            continue

        # Create unique node ID
        key_id = f'{key}~{depth_level}/{len(nodes)}'

        # Special handling for root node
        if key == 'Light absorption':
            nodes[key_id] = {'name': 'Photons', 
                             'value': 1, 
                             'log_value': log_normalize(1),
                             'level': depth_level,
                             'history': history,
                             'parent': None,
                             'is_terminal': False}

        # Special handling for transmitted node
        elif key == 'transmitted':
            amount_formed = value

            nodes[key_id] = {'name': 'Transmitted', 
                             'value': value, 
                             'log_value': log_normalize(value),
                             'level': depth_level,
                             'history': history,
                             'parent': parent,
                             'is_terminal': True}

        # General case for node handling
        else:
            amount_formed = value.get('amount_formed', 0) or value.get('absorbed', 0)

            is_terminal = not any(
                isinstance(v, dict) and k not in metadata_keys 
                for k, v in value.items()
            ) if isinstance(value, dict) else True

            nodes[key_id] = {'name': key, 
                             'value': amount_formed, 
                             'log_value': log_normalize(amount_formed),
                             'level': depth_level,
                             'history': history,
                             'parent': parent,
                             'is_terminal': is_terminal}
            
        # Create link from parent to current node
        if parent is not None:            
            links.append({'source': parent,
                          'target': key_id,
                          'value': amount_formed,
                          'log_value': log_normalize(amount_formed)})
        
        # Recurse into children
        if isinstance(value, dict):
            child_history = history + [key_id]
            child_result = transform_pathways_data(value,
                                                   parent = key_id,
                                                   depth_level = depth_level + 1,
                                                   history = child_history,
                                                   min_expected = min_expected)
            nodes.update(child_result['nodes'])
            links.extend(child_result['links'])

    result = {'nodes': nodes, 'links': links, 'min_expected': min_expected}

    # Set root_id on first call
    if depth_level == 0:
        root_id = next(iter(nodes.keys()))
        result['root_id'] = root_id
    
    return result

def add_sibling_order(data, value_key='value'):
    """
    Add sibling order information to all nodes in the pathway data.
    
    Groups nodes by their parent and assigns order_idx (sorted by value_key
    in descending order) and num_siblings to each node.
    
    Parameters
    ----------
    data : dict
        The pathway data dictionary containing 'nodes' and 'links'.
    value_key : str, optional
        The key to use for sorting siblings. Default is 'value'.
    
    Returns
    -------
    dict
        The modified data dictionary with order_idx and num_siblings added to nodes.
    """
    # Group all nodes by their parent (across all levels)
    siblings_by_parent = {}
    
    for node_id, node in data['nodes'].items():
        parent = node['parent']
        if parent not in siblings_by_parent:
            siblings_by_parent[parent] = []
        siblings_by_parent[parent].append((node_id, node))
    
    # Process each sibling group
    for parent, siblings in siblings_by_parent.items():
        # Sort siblings by value_key in descending order
        siblings_sorted = sorted(siblings, key=lambda x: x[1][value_key], reverse=True)
        
        num_siblings = len(siblings_sorted)
        
        # Assign order_idx and num_siblings to each node
        for order_idx, (node_id, node) in enumerate(siblings_sorted):
            node['order_idx'] = order_idx
            node['num_siblings'] = num_siblings
    
    return data

def _update_sibling_order_after_removal(data, removed_node_ids):
    """
    Update order_idx for siblings after nodes have been removed.
    
    Removed nodes are treated as if they were at the bottom of the sibling order.
    Remaining siblings are re-indexed starting from 0 while preserving their
    relative order. num_siblings is NOT decremented (memory of deleted nodes).
    
    Parameters
    ----------
    data : dict
        The pathway data dictionary.
    removed_node_ids : set
        Set of node IDs that were removed.
    """
    # Group remaining nodes by parent
    siblings_by_parent = {}
    
    for node_id, node in data['nodes'].items():
        parent = node['parent']
        if parent not in siblings_by_parent:
            siblings_by_parent[parent] = []
        siblings_by_parent[parent].append((node_id, node))
    
    # Re-index each sibling group (preserving relative order by current order_idx)
    for parent, siblings in siblings_by_parent.items():
        # Sort by existing order_idx to preserve relative order
        siblings_sorted = sorted(siblings, key=lambda x: x[1]['order_idx'])
        
        # Re-assign order_idx starting from 0
        for new_idx, (node_id, node) in enumerate(siblings_sorted):
            node['order_idx'] = new_idx

def post_process_pathways_data(data):
    """
    Post-process pathway data by consolidating terminal nodes.
    
    Terminal nodes that cycle back to earlier occurrences are redirected,
    and remaining terminal nodes are consolidated by species name.
    Updates sibling order_idx after removing cycling terminal nodes.
    
    Parameters
    ----------
    data : dict
        The pathway data dictionary containing 'nodes' and 'links'.
    
    Returns
    -------
    dict
        The modified data dictionary with consolidated terminal nodes.
    """
    # Dictionary to track consolidated terminal nodes by species name
    # {species_name: {'node': consolidated_node, 'original_ids': [list of original node ids]}}
    terminal_species = {}
    
    # Track removed node IDs for sibling order update
    removed_cycling_node_ids = set()

    # Iterate over a copy of keys since we'll be deleting during iteration
    for node_id in list(data['nodes'].keys()):
        node = data['nodes'][node_id]

        # Skipping over non-terminal nodes
        if not node['is_terminal']:
            continue

        is_in_history, last_match = check_in_history(node['name'], node['history'])

        # If previous occurance of species is found in history,
        # redirect links to that node instead
        if is_in_history:
            for link in data['links']:
                if link['target'] == node_id:
                    link['target'] = last_match
            # Track this node for sibling order update
            removed_cycling_node_ids.add(node_id)

        # Otherwise, consolidate terminal nodes by species
        else:
            species_name = node['name']
            
            if species_name not in terminal_species:
                # Create a new consolidated node for this species
                consolidated_id = f'{species_name}~terminal'
                terminal_species[species_name] = {
                    'node': {
                        'name': species_name,
                        'value': node['value'],
                        'log_value': None, # will be set later based on sum of regular value
                        'level': None, # Will be set after node removal
                        'history': [],
                        'is_terminal': True,
                        'original_ids': [node_id],
                        'parent': [node['parent']],
                    },
                    'consolidated_id': consolidated_id
                }
            else:
                # Add value to existing consolidated node
                terminal_species[species_name]['node']['value'] += node['value']
                terminal_species[species_name]['node']['original_ids'].append(node_id)
                terminal_species[species_name]['node']['parent'].append(node['parent'])
            
        # Remove terminal node directly
        del data['nodes'][node_id]

    # Redirect links from original terminal nodes to consolidated nodes
    for species_name, species_data in terminal_species.items():
        for link in data['links']:
            if link['target'] in species_data['node']['original_ids']:
                link['target'] = species_data['consolidated_id']
    
    # Update sibling order_idx after removing cycling terminal nodes
    if removed_cycling_node_ids:
        _update_sibling_order_after_removal(data, removed_cycling_node_ids)

    # Get maximum level after node removal
    max_level = max(node['level'] for node in data['nodes'].values())
    data['max_level'] = max_level + 1 # Highest level + 1 to consider terminal nodes (below)

    # Add consolidated terminal nodes to the data
    for species_data in terminal_species.values():
        species_data['node']['log_value'] = transform_log_normalized(
            species_data['node']['value'],
            min_expected=data['min_expected'])
        
        species_data['node']['level'] = max_level + 1
        data['nodes'][species_data['consolidated_id']] = species_data['node']

    return data
    
def compute_child_y_coordinate(y_window: tuple, num_siblings: int, order_idx: int) -> float:
    """
    Compute the y-coordinate of a child node based on its position among siblings.
    
    Parameters
    ----------
    y_window : tuple
        (lower_edge, upper_edge) of the parent's window.
    num_siblings : int
        Total number of siblings in the group (including this node).
    order_idx : int
        Position of this node among siblings (0 = first/top, sorted by value descending).
    
    Returns
    -------
    float
        The y-coordinate for this child node.
    """
    lower_edge, upper_edge = y_window
    
    if num_siblings == 1:
        # Single child: center of window (same as parent y_coord)
        return (lower_edge + upper_edge) / 2
    
    # Multiple siblings: distribute from upper_edge (idx=0) to lower_edge (idx=num_siblings-1)
    fraction = order_idx / (num_siblings - 1)
    return upper_edge - fraction * (upper_edge - lower_edge)

def compute_y_window(y_coord, level, fanning_factor, assumed_branching_degree, max_level, value):

    window_size = adjust_fanning_factor(level, fanning_factor, assumed_branching_degree, max_level) 
    window_size *= value

    lower_edge = y_coord - window_size / 2
    upper_edge = y_coord + window_size / 2

    return (lower_edge, upper_edge)

def computate_y_coordinates(data, level, fanning_factor, assumed_branching_degree,
                            value_key):
    
    for node_id, node in data['nodes'].items():
        if node['level'] != level:
            continue

        parent_window = data['nodes'][node['parent']]['y_window']

        # Compute y_coord for this node
        node['y_coord'] = compute_child_y_coordinate(
            parent_window,
            node['num_siblings'],
            node['order_idx']
        )

        node['y_min'] = node['y_coord'] - node[value_key] / 2
        node['y_max'] = node['y_coord'] + node[value_key] / 2

        node['y_window'] = compute_y_window(
            node['y_coord'],
            node['level'],
            fanning_factor,
            assumed_branching_degree,
            data['max_level'],
            node[value_key]
        )

def compute_y_coordinates_terminal_nodes(data, value_key):

    for node_id, node in data['nodes'].items():
        if not node['is_terminal']:
            continue
        
        parent_y_coords = []

        for parent in node['parent']:
            parent_y_coords.append(data['nodes'][parent]['y_coord'])
        
        # get smallest value from parent y coords
        node['y_coord'] = min(parent_y_coords)

        node['y_min'] = node['y_coord'] - node[value_key] / 2
        node['y_max'] = node['y_coord'] + node[value_key] / 2

def add_y_coordinates(data, fanning_factor, assumed_branching_degree,
                      value_key='log_value'):
    
    # Add coordinates for root node
    root_node = data['nodes'][data['root_id']]

    root_node['y_coord'] = 0
    root_node['y_min'] = root_node['y_coord'] - root_node[value_key] / 2
    root_node['y_max'] = root_node['y_coord'] + root_node[value_key] / 2

    root_node['y_window'] = compute_y_window(
        root_node['y_coord'],
        root_node['level'],
        fanning_factor,
        assumed_branching_degree,
        data['max_level'], 
        root_node[value_key]
    )

    # get all levels in data as a list (excluding root and terminal level)
    levels = sorted(set(node['level'] for node in data['nodes'].values()) - {0, data['max_level']})

    # Iterate through levels and compute y coordinates
    for level in levels: 
        computate_y_coordinates(data, level, fanning_factor, assumed_branching_degree,
                               value_key)

    compute_y_coordinates_terminal_nodes(data, value_key)

    return data

def add_link_starting_values(data):
    """
    Add starting_value to each link based on source node ordering.
    
    For each source node, outgoing links are sorted by their target's y_coord
    (highest first). The starting_value represents the vertical position on
    the source node where each link originates, counting down from 1.
    
    Parameters
    ----------
    data : dict
        The pathway data dictionary containing 'nodes' and 'links'.
    
    Returns
    -------
    dict
        The modified data dictionary with starting_value added to each link.
    """
    # Group links by source node
    links_by_source = {}
    
    for link in data['links']:
        source = link['source']
        if source not in links_by_source:
            links_by_source[source] = []
        links_by_source[source].append(link)
    
    # Process each source node's outgoing links
    for source_id, outgoing_links in links_by_source.items():
        source_node = data['nodes'][source_id]
        
        # Sort links by target's y_coord (highest first)
        outgoing_links_sorted = sorted(
            outgoing_links,
            key=lambda link: data['nodes'][link['target']]['y_coord'],
            reverse=True
        )
        
        # Assign starting_value to each link
        cumulative_fraction = 0.0
        
        for link in outgoing_links_sorted:
            # Starting value counts down from 1
            link['starting_value'] = 1.0 - cumulative_fraction
            
            # Calculate this link's source value fraction
            source_value_fraction = link['value'] / source_node['value']
            cumulative_fraction += source_value_fraction

    # Special case: for terminal nodes as target, set starting_value to 1.0
    for links in data['links']:
        if data['nodes'][links['target']]['is_terminal'] and data['nodes'][links['source']]['level'] != 0:
            links['starting_value'] = 1.0
    
    return data

def add_link_ending_values(data):
    """
    Add ending_value to links targeting terminal nodes based on target node ordering.
    
    For each terminal target node, incoming links are sorted by their source's y_coord
    (highest first). The ending_value represents the vertical position on
    the target node where each link terminates, counting down from 1.
    
    For non-terminal targets, ending_value is set to 1.0 (default behavior).
    
    Parameters
    ----------
    data : dict
        The pathway data dictionary containing 'nodes' and 'links'.
    
    Returns
    -------
    dict
        The modified data dictionary with ending_value added to each link.
    """
    # Group links by target node
    links_by_target = {}
    
    for link in data['links']:
        target = link['target']
        if target not in links_by_target:
            links_by_target[target] = []
        links_by_target[target].append(link)
    
    # Process each target node's incoming links
    for target_id, incoming_links in links_by_target.items():
        target_node = data['nodes'][target_id]
        
        # Only compute ending values for terminal nodes
        if not target_node['is_terminal']:
            # For non-terminal nodes, set default ending_value
            for link in incoming_links:
                link['ending_value'] = 1.0
            continue
        
        # Sort links by source's y_coord (highest first)
        incoming_links_sorted = sorted(
            incoming_links,
            key=lambda link: data['nodes'][link['source']]['y_coord'],
            reverse=True
        )
        
        # Assign ending_value to each link
        cumulative_fraction = 0.0
        
        for link in incoming_links_sorted:
            # Ending value counts down from 1
            link['ending_value'] = 1.0 - cumulative_fraction
            
            # Calculate this link's target value fraction
            target_value_fraction = link['value'] / target_node['value']
            cumulative_fraction += target_value_fraction
    
    return data

def process_links(data, value_key):

    for link in data['links']:
        source_node = data['nodes'][link['source']]
        target_node = data['nodes'][link['target']]

        start_value = source_node['y_min'] + link['starting_value'] * source_node[value_key]
        end_value = target_node['y_min'] + link['ending_value'] * target_node[value_key]

        source_value_fraction = link['value'] / source_node['value']
        source_value_adjusted = source_node[value_key] * source_value_fraction

        target_value_fraction = link['value'] / target_node['value']
        target_value_adjusted = target_node[value_key] * target_value_fraction

        # Process looping links
        if target_node['level'] < source_node['level']:
            link['looping'] = True

            link['coordinates'] = {
                'source': {'x': source_node['level'], 
                        'y': (source_node['y_min'], source_node['y_min'] + source_value_adjusted)},
                'target': {'x': target_node['level'], 
                        'y': (target_node['y_min'], target_node['y_min'] + target_value_adjusted)},        
            }

        # Process non-looping links
        else:
            link['looping'] = False

            link['coordinates'] = {
                'source': {'x': source_node['level'], 
                        'y': (start_value - source_value_adjusted, start_value)},
                'target': {'x': target_node['level'],    
                           'y': (end_value - target_value_adjusted, end_value)}    
            }

    return data

def transform_data_for_plotting(data, value_key='value',
                                fanning_factor=0.7,
                                assumed_branching_degree=1.7):
    
    # Transform nested data into links/nodes structure
    results = transform_pathways_data(data)

    # Add sibling information to nodes, including order and number of siblings
    results = add_sibling_order(results, value_key=value_key)

    # Post-process to consolidate terminal nodes and handle cycles
    results = post_process_pathways_data(results)

    # Add y coordinates to nodes
    results = add_y_coordinates(results, fanning_factor=fanning_factor, assumed_branching_degree=assumed_branching_degree,
                                value_key=value_key)
    
    # Add link starting and ending values
    results = add_link_starting_values(results)
    results = add_link_ending_values(results)

    # Process links to add coordinates and looping info
    results = process_links(results, value_key)
    
    return results

        
def main():
 
    # With C excitation
    test_data = {'Light absorption': {'[A]': {'[A-excited]': {'[A]': {'amount_formed': np.float64(0.25101450169405526)},
                                              '[B]': {'[A]': {'amount_formed': np.float64(0.0822802686631513)},
                                                      '[B-excited]': {'[B]': {'amount_formed': np.float64(0.000524993170390105)},
                                                                      '[C]': {'[C-excited]': {'[C]': {'amount_formed': np.float64(0.0008662387311436734)},
                                                                                              'amount_formed': np.float64(0.0008662387311436734)},
                                                                              'amount_formed': np.float64(0.0008662387311436734)},
                                                                      'amount_formed': np.float64(0.0013912319015337786)},
                                                      'amount_formed': np.float64(0.08367150056468509)},
                                              'amount_formed': np.float64(0.33468600225874034)},
                              'absorbed': np.float64(0.33468600225874034)},
                      '[B]': {'[B-excited]': {'[B]': {'amount_formed': np.float64(0.0005283380534483106)},
                                              '[C]': {'[C-excited]': {'[C]': {'amount_formed': np.float64(0.0008717577881897126)},
                                                                      'amount_formed': np.float64(0.0008717577881897126)},
                                                      'amount_formed': np.float64(0.0008717577881897126)},
                                              'amount_formed': np.float64(0.0014000958416380235)},
                              'absorbed': np.float64(0.0014000958416380235)},
                      '[C]': {'[C-excited]': {'[C]': {'amount_formed': np.float64(0.002792020891002747)},
                                              'amount_formed': np.float64(0.002792020891002747)},
                              'absorbed': np.float64(0.002792020891002747)},
                      'transmitted': np.float64(0.6611218810086189)}}

    # Without C excitation
    test_data = {'Light absorption': {'[A]': {'[A-excited]': {'[A]': {'amount_formed': np.float64(0.25101450170151896)},
                                              '[B]': {'[A]': {'amount_formed': np.float64(0.08228026866560609)},
                                                      '[B-excited]': {'[B]': {'amount_formed': np.float64(0.0005249931704026031)},
                                                                      '[C]': {'amount_formed': np.float64(0.0008662387311642952)},
                                                                      'amount_formed': np.float64(0.0013912319015668981)},
                                                      'amount_formed': np.float64(0.08367150056717299)},
                                              'amount_formed': np.float64(0.33468600226869194)},
                              'absorbed': np.float64(0.33468600226869194)},
                      '[B]': {'[B-excited]': {'[B]': {'amount_formed': np.float64(0.0005283380534604106)},
                                              '[C]': {'amount_formed': np.float64(0.0008717577882096776)},
                                              'amount_formed': np.float64(0.001400095841670088)},
                              'absorbed': np.float64(0.001400095841670088)},
                      '[C]': {'[C-excited]': {'amount_formed': np.float64(0.0027920208895789896)},
                              'absorbed': np.float64(0.0027920208895789896)},
                      'transmitted': np.float64(0.661121881000059)}}
    
    results = transform_data_for_plotting(test_data,
                                        value_key='log_value',
                                        fanning_factor=0.7,
                                        assumed_branching_degree=1.7)

    pp.pprint(results)

    plot_pathway_bars(results)
            
if __name__ == '__main__':
    main()  