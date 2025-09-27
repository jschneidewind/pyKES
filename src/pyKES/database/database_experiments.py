import pandas as pd
import numpy as np
import h5py
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Union
import json
import pickle


def import_overview_excel(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name = sheet_name)
    return df

@dataclass
class Experiment:
    """Store data and metadata for a single experiment."""
    experiment_name: str
    raw_data_file: str
    color: str
    metadata: Dict[str, any]
    raw_data: Dict[str, any]
    processed_data: Dict[str, any]

def save_nested_dict_to_hdf5(group, data_dict, prefix=""):
    """
    Recursively save nested dictionaries to HDF5 group.
    Handles numpy arrays, basic Python types, and nested structures.
    """
    for key, value in data_dict.items():
        full_key = f"{prefix}/{key}" if prefix else key
        
        if isinstance(value, np.ndarray):
            # Save numpy arrays directly
            group.create_dataset(full_key, data=value)
            
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            save_nested_dict_to_hdf5(group, value, full_key)
            
        elif isinstance(value, (str, int, float, bool)):
            # Save basic types as attributes or small datasets
            if isinstance(value, str):
                # Handle strings (need special encoding for HDF5)
                group.create_dataset(full_key, data=value.encode('utf-8'))
            else:
                group.create_dataset(full_key, data=value)
                
        elif isinstance(value, (list, tuple)):
            # Try to convert to numpy array, fallback to JSON
            try:
                arr = np.array(value)
                group.create_dataset(full_key, data=arr)
            except:
                # If can't convert to array, store as JSON string
                json_str = json.dumps(value)
                group.create_dataset(full_key, data=json_str.encode('utf-8'))
                group[full_key].attrs['type'] = 'json'
                
        else:
            # For other types, use JSON serialization
            try:
                json_str = json.dumps(value)
                group.create_dataset(full_key, data=json_str.encode('utf-8'))
                group[full_key].attrs['type'] = 'json'
            except:
                # Last resort: pickle (less portable but handles everything)
                pickled_data = pickle.dumps(value)
                group.create_dataset(full_key, data=np.frombuffer(pickled_data, dtype=np.uint8))
                group[full_key].attrs['type'] = 'pickle'

def load_nested_dict_from_hdf5(group, prefix=""):
    """
    Recursively load nested dictionaries from HDF5 group.
    """
    result = {}
    
    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Remove prefix from name
            key = name[len(prefix):].lstrip('/') if prefix else name
            
            # Handle different data types
            if obj.attrs.get('type') == 'json':
                # JSON-encoded data
                json_str = obj[()].decode('utf-8')
                value = json.loads(json_str)
            elif obj.attrs.get('type') == 'pickle':
                # Pickled data
                pickled_bytes = obj[()].tobytes()
                value = pickle.loads(pickled_bytes)
            else:
                # Regular data (numpy arrays, numbers, strings)
                value = obj[()]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
            
            # Build nested dictionary structure
            keys = key.split('/')
            current_dict = result
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            current_dict[keys[-1]] = value
    
    group.visititems(visit_func)
    return result

@dataclass
class ExperimentalDataset:
    """
    """

    experiments: Dict[str, 'Experiment'] = field(default_factory=dict)
    overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def add_experiment(self, experimental_data: 'Experiment'):
        """Add an experiment to the dataset"""
        self.experiments[experimental_data.experiment_name] = experimental_data

    # def update_metadata(self, file_name):

    #     new_overview_df = import_overview_excel(file_name)
    #     self.overview_df = new_overview_df

    #     for experiment_name, experiment_data in self.experiments.items():
    #         new_experimental_metadata = get_experimental_metadata(experiment_name, self.overview_df)

    #         experiment_data.experiment_metadata = ExperimentMetadata(**new_experimental_metadata, 
    #                                         color = get_experiment_color(new_experimental_metadata['experiment_name']))
            
    #         self.insert_experiment_results_in_df(experiment_data)

    def save_to_hdf5(self, filename: str):
        """
        Save experiments to HDF5 file.
        """
        if not self.overview_df.empty:
            self.overview_df.to_hdf(filename, key='overview_df', mode='w', format='table')

        with h5py.File(filename, 'a') as f:
            for exp_name, experiment in self.experiments.items():

                if exp_name in f:
                    print(f"Experiment {exp_name} already exists. Overwriting...")
                    del f[exp_name]

                # Create a group for each experiment
                exp_grp = f.create_group(exp_name)

                exp_grp.attrs['experiment_name'] = experiment.experiment_name
                exp_grp.attrs['raw_data_file'] = experiment.raw_data_file
                exp_grp.attrs['color'] = experiment.color

                # Save nested dictionaries in separate groups
                if experiment.raw_data:
                    raw_data_group = exp_grp.create_group('raw_data')
                    save_nested_dict_to_hdf5(raw_data_group, experiment.raw_data)
                
                if experiment.metadata:
                    metadata_group = exp_grp.create_group('metadata')
                    save_nested_dict_to_hdf5(metadata_group, experiment.metadata)
                    
                if experiment.processed_data:
                    processed_data_group = exp_grp.create_group('processed_data')
                    save_nested_dict_to_hdf5(processed_data_group, experiment.processed_data)

                print(f"Experiment {exp_name} added successfully.")
                
    @classmethod
    def load_from_hdf5(cls, filename: str):
        """
        Load experiments from HDF5 file.
        """

        dataset = cls()

        try:
            dataset.overview_df = pd.read_hdf(filename, key='overview_df')
        except (KeyError, ValueError):
            print("No overview DataFrame found in file")
            dataset.overview_df = pd.DataFrame()

        with h5py.File(filename, 'r') as f:
            for exp_name in f.keys():
                if exp_name == 'overview_df':  # Skip the overview_df group
                    continue

                exp_group = f[exp_name]
                
                # Load simple attributes from the experiment group
                experiment_name = exp_group.attrs['experiment_name']
                raw_data_file = exp_group.attrs['raw_data_file'] 
                color = exp_group.attrs['color']
                
                # Load nested dictionaries
                raw_data = load_nested_dict_from_hdf5(exp_group['raw_data']) if 'raw_data' in exp_group else {}
                metadata = load_nested_dict_from_hdf5(exp_group['metadata']) if 'metadata' in exp_group else {}
                processed_data = load_nested_dict_from_hdf5(exp_group['processed_data']) if 'processed_data' in exp_group else {}
                
                single_experiment =  Experiment(
                    experiment_name=experiment_name,
                    raw_data_file=raw_data_file,
                    color=color,
                    metadata=metadata,
                    raw_data=raw_data,
                    processed_data=processed_data
                )

                dataset.add_experiment(single_experiment)

        return dataset
    
    def list_experiments(self) -> List[str]:
        """
        List all experiments in the dataset
        
        Returns:
            List[str]: A sorted list of experiment names
        """
        return sorted(self.experiments.keys())

    def print_experiments(self):
        """
        Print all experiments in the dataset in a formatted way
        """
        if not self.experiments:
            print("No experiments in dataset")
            return
            
        print(f"Dataset contains {len(self.experiments)} experiments:")
        for i, name in enumerate(self.list_experiments(), 1):
            print(f"{i}. {name}")


def usage_example():

    dataset = ExperimentalDataset()

    # Add an experiment
    exp1 = Experiment(
        experiment_name="Exp1",
        raw_data_file="data/exp1.h5",
        color="blue",
        metadata={"temperature": 300, "pressure": 101325},
        raw_data={"current": np.array([0, 1, 2]), "voltage": np.array([0, 0.5, 1])},
        processed_data={"baseline_corrected": {
            "efficiency": np.array([0.9, 0.95, 0.98]),
            'fit_parameters': {
                "a": 0.1,
                "b": 0.2,
                }       
            }
        }
    )
    
    dataset.add_experiment(exp1)

    # Save to HDF5
    dataset.save_to_hdf5("experiments.h5")

    # Load from HDF5
    loaded_dataset = ExperimentalDataset.load_from_hdf5("experiments.h5")
    loaded_dataset.print_experiments()

    print(loaded_dataset.experiments['Exp1'].processed_data['baseline_corrected']['fit_parameters'])

if __name__ == '__main__':
    usage_example()



