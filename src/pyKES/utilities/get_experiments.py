"""Select experiments from a dataset by their metadata."""

from typing import Dict
import numpy as np

def get_experiments_by_metadata(data_dict, 
                                **metadata_criteria) -> Dict:
    """
    Return the experiments whose metadata matches every given criterion.

    Parameters
    ----------
    data_dict : dict
        Mapping of experiment name to `Experiment`, typically
        ``dataset.experiments``.
    **metadata_criteria
        Metadata keys and the values they must have. An experiment matches only
        if all of them agree; an experiment lacking a key never matches.

    Returns
    -------
    dict
        The matching subset, in the same ``name -> Experiment`` form, so the
        result can be filtered again.

    Examples
    --------
    >>> matching = get_experiments_by_metadata(dataset.experiments,
    ...                                        type='intensity', intensity=0.5)
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
    Collect the distinct values a metadata key takes across experiments.

    Useful for discovering the axis of a series — the light intensities a set of
    experiments was run at, say — without having to state it in advance.

    Parameters
    ----------
    experiment_group : dict
        Mapping of experiment name to `Experiment`.
    metadata_key : str
        Metadata key read from each experiment. Experiments lacking it, or
        holding None under it, are skipped.

    Returns
    -------
    numpy.ndarray
        The distinct values, sorted.

    Examples
    --------
    >>> intensities = get_unique_metadata_values(group, 'intensity')
    """
    unique_values = set()
    
    for experiment in experiment_group.values():
        if metadata_key in experiment.metadata:
            value = experiment.metadata[metadata_key]
            if value is not None:
                unique_values.add(value)
                
    return np.asarray(sorted(unique_values))