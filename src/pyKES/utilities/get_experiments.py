from typing import Dict
import numpy as np

def get_experiments_by_metadata(data_dict, 
                                **metadata_criteria) -> Dict:
    """
    Return experiments that match all specified metadata criteria.
    
    Args:
        dataset: The ExperimentalDataset to search
        **metadata_criteria: Key-value pairs that must match in experiment metadata
        
    Returns:
        Dict of experiment_name -> Experiment for matching experiments
        
    Example:
        matching = get_experiments_by_metadata(dataset, type='intensity', intensity=0.5)
    """
    matching_experiments = {}
    
    for name, experiment in data_dict.items():
        # Check if all criteria match
        if all(experiment.metadata.get(key) == value for key, value in metadata_criteria.items()):
            matching_experiments[name] = experiment
            
    return matching_experiments

def get_unique_metadata_values(experiment_group: Dict, 
                                metadata_key: str) -> list:
    """
    Get all unique values for a specific metadata key across a group of experiments.
    
    Args:
        experiment_group: Dict of experiment_name -> Experiment
        metadata_key: The metadata key to extract unique values for
        
    Returns:
        Sorted np.arraynd of unique values for that metadata key (excluding None)
        
    Example:
        unique_starts = get_unique_metadata_values(group, 'start')
    """
    unique_values = set()
    
    for experiment in experiment_group.values():
        if metadata_key in experiment.metadata:
            value = experiment.metadata[metadata_key]
            if value is not None:
                unique_values.add(value)
                
    return np.asarray(sorted(unique_values))