def resolve_path_slash(path_string, obj):
        """
        Resolve a slash-separated path through object attributes and dict keys.
        
        Handles dict keys that contain '/' by trying the longest possible match first.
        For example, 'metadata/Catalyst loading [wt% Rh/Cr]' will correctly resolve
        to obj.metadata['Catalyst loading [wt% Rh/Cr]'] even though the key contains '/'.
        """
        current = obj
        components = path_string.split('/')
        
        i = 0
        while i < len(components):
            resolved = False
            
            # Try progressively longer key combinations (greedy approach)
            # This handles dict keys that contain '/'
            for j in range(len(components), i, -1):
                potential_key = '/'.join(components[i:j])
                
                # Try as attribute first
                if hasattr(current, potential_key):
                    current = getattr(current, potential_key)
                    i = j
                    resolved = True
                    break
                # Then try as dict key
                elif isinstance(current, dict) and potential_key in current:
                    current = current[potential_key]
                    i = j
                    resolved = True
                    break
            
            if not resolved:
                # Could not resolve any component starting at position i
                attempted_keys = ['/'.join(components[i:j]) for j in range(len(components), i, -1)]
                raise ValueError(
                    f"Cannot resolve path '{path_string}' at component '{components[i]}'. "
                    f"Tried keys: {attempted_keys}. "
                    f"Available: {dir(current) if hasattr(current, '__dir__') else list(current.keys()) if isinstance(current, dict) else 'N/A'}"
                )
        
        return current  


def resolve_experiment_attributes(template_dict, experiment, mode='strict'):
    """
    Recursively resolve attribute paths from experiment object.
    
    Parameters
    ----------
    template_dict : dict
        Dictionary with string paths to resolve (e.g., 'processed_data/time_s')
        or nested dictionaries containing such paths.
    experiment : Experiment
        Experiment object containing the data to resolve.
    mode : {'strict', 'semi-strict', 'permissive'}, default 'permissive'
        Resolution mode:
        - 'strict': All paths must resolve successfully, raises error otherwise
        - 'semi-strict': At least one top-level entry must resolve, raises error if none do
        - 'permissive': Skip entries that cannot be resolved, return what's available
        
    Returns
    -------
    dict
        Dictionary with resolved values. In permissive mode, excludes entries that 
        couldn't be fully resolved. In strict/semi-strict modes, contains all requested
        entries or raises an error.
        
    Raises
    ------
    ValueError
        In 'strict' mode: if any path cannot be resolved
        In 'semi-strict' mode: if no top-level entries can be resolved
        
    Examples
    --------
    >>> # Permissive mode (default) - skip missing data
    >>> data_template = {
    ...     '[O2]': {'x': 'processed_data/time_s', 'y': 'processed_data/O2_conc'},
    ...     '[F]': {'x': 'processed_data/time_s', 'y': 'processed_data/F_conc'}
    ... }
    >>> resolved = resolve_experiment_attributes(data_template, experiment)
    >>> # Only returns entries where all nested values resolved
    
    >>> # Strict mode - all must resolve
    >>> resolved = resolve_experiment_attributes(data_template, experiment, mode='strict')
    >>> # Raises ValueError if any path fails
    
    >>> # Semi-strict mode - at least one entry must work
    >>> resolved = resolve_experiment_attributes(data_template, experiment, mode='semi-strict')
    >>> # Raises ValueError if all entries fail, returns partial results otherwise
    """

    result_dict = {}
    failed_keys = []
    
    for key, value in template_dict.items():
        try:
            # Check if the value is a nested dictionary (but not a function spec)
            if isinstance(value, dict) and 'function' not in value:
                # Recursively resolve the nested dictionary with same mode
                resolved_nested = resolve_experiment_attributes(value, experiment, mode=mode)
                
                # Only include if ALL nested values were resolved (same keys as template)
                if set(resolved_nested.keys()) == set(value.keys()):
                    result_dict[key] = resolved_nested
                elif mode == 'strict':
                    raise ValueError(f"Failed to resolve nested dictionary for key '{key}'")
                else:
                    failed_keys.append(key)
                    
            elif isinstance(value, str): 
                if '/' in value:
                    # Resolve path string (only resolved if value contains '/')
                    result_dict[key] = resolve_path_slash(value, experiment)
                else:
                    # Otherwise pass the string value through unchanged
                    result_dict[key] = value
                
            else:
                # For other types (e.g., numbers, lists), just copy the value
                result_dict[key] = value
                
        except (ValueError, KeyError, AttributeError) as e:
            if mode == 'strict':
                raise ValueError(f"Failed to resolve '{key}': {str(e)}") from e
            else:
                failed_keys.append(key)
                continue
    
    # Semi-strict mode: ensure at least one entry resolved
    if mode == 'semi-strict' and not result_dict:
        raise ValueError(
            f"Semi-strict mode: No entries could be resolved. "
            f"Failed keys: {failed_keys}"
        )
    
    return result_dict


def testing():

    class Experiment:
        def __init__(self):
            self.metadata = {
                'Catalyst loading [wt% Rh/Cr]': 5.0,
                'Temperature_C': 350,
                'Nested': {
                    'Level1': {
                        'Loading Rh/Cr': 'Deep/Value'
                    }
                }
            }
    
    testing_exp = Experiment()

    template = {
        'loading': 'metadata/Catalyst loading [wt% Rh/Cr]',
        'temp': 'metadata/Temperature_C',
        'missing': 'metadata/Nonexistent Key',
        'nested_value': 'metadata/Nested/Level1/Loading Rh/Cr',
        'should_be_included': '[B]'
    }

    print(resolve_experiment_attributes(template, testing_exp, mode='permissive'))
    

if __name__ == "__main__":
    testing()