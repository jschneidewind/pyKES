import pandas as pd
import numpy as np
import h5py
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Union, Optional
import json
import pickle
from io import StringIO

# Bump when the on-disk layout changes in a way older readers cannot ignore
# (renamed/removed groups, changed required attributes). Purely additive
# changes to processing_parameters or per-experiment dicts do not require
# a bump.
SCHEMA_VERSION = "1.0"


def import_overview_excel(file_name, 
                          sheet_name,
                          dtype = None):
    
    '''
    Import the overview Excel sheet as a DataFrame.
    '''

    df = pd.read_excel(file_name, 
                       sheet_name = sheet_name, 
                       dtype = dtype)

    return df

@dataclass
class Experiment:
    """Store data and metadata for a single experiment."""
    experiment_name: str
    raw_data_file: str
    color: str
    group: str
    metadata: Dict[str, any]
    raw_data: Dict[str, any]
    processed_data: Dict[str, any]

def save_nested_dict_to_hdf5(group, data_dict, prefix=""):
    """
    Recursively save nested dictionaries to HDF5 group.
    Handles numpy arrays, basic Python types, and nested structures.
    
    Note: Replaces '/' in keys with '__SLASH__' to avoid HDF5 path conflicts.
    """
    for key, value in data_dict.items():
        # Replace '/' in keys to avoid HDF5 path interpretation issues
        safe_key = key.replace('/', '__SLASH__')
        full_key = f"{prefix}/{safe_key}" if prefix else safe_key
        
        if isinstance(value, np.ndarray):
            # Save numpy arrays directly
            group.create_dataset(full_key, data=value)
            
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            save_nested_dict_to_hdf5(group, value, full_key)
            
        elif isinstance(value, (str, int, float, bool, np.bool_)):
            # Save basic types as datasets
            if isinstance(value, str):
                # Handle strings (need special encoding for HDF5)
                group.create_dataset(full_key, data=value.encode('utf-8'))
            elif isinstance(value, (bool, np.bool_)):
                # Convert bool to int for HDF5 compatibility
                group.create_dataset(full_key, data=int(value))
                group[full_key].attrs['type'] = 'bool'
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
    
    Note: Restores '__SLASH__' in keys back to '/' after loading.
    """
    result = {}
    
    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Remove prefix from name
            key = name[len(prefix):].lstrip('/') if prefix else name
            
            # Handle different data types
            if obj.attrs.get('type') == 'bool':
                # Restore boolean type
                value = bool(obj[()])
            elif obj.attrs.get('type') == 'json':
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
                # Convert numpy scalar types to native Python types
                elif isinstance(value, (np.integer, np.floating)):
                    value = value.item()
            
            # Build nested dictionary structure
            keys = key.split('/')
            # Restore '/' characters in keys
            keys = [k.replace('__SLASH__', '/') for k in keys]
            
            current_dict = result
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            current_dict[keys[-1]] = value
    
    group.visititems(visit_func)
    return result


def write_df_to_hdf(h5_file: h5py.File, df: pd.DataFrame, key: str = 'overview_df') -> None:
    """Write a DataFrame to HDF5 with h5py-native serialization.

    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 file handle.
    df : pd.DataFrame
        DataFrame to serialize.
    key : str, default='overview_df'
        Group name under which the DataFrame payload is stored.

    Returns
    -------
    None
    """
    if key in h5_file:
        del h5_file[key]

    df_group = h5_file.create_group(key)
    payload = df.to_json(orient='split', date_format='iso')
    df_group.create_dataset('json', data=np.bytes_(payload))
    df_group.attrs['serialization_format'] = 'pandas_json_split'


def read_df_from_hdf(h5_file: h5py.File, key: str = 'overview_df') -> pd.DataFrame:
    """Read a DataFrame previously written by ``write_df_to_hdf``.

    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 file handle.
    key : str, default='overview_df'
        Group name from which the DataFrame payload is loaded.

    Returns
    -------
    pd.DataFrame
        Deserialized DataFrame. Returns an empty DataFrame if key is absent.
    """
    if key not in h5_file:
        return pd.DataFrame()

    df_group = h5_file[key]
    if 'json' not in df_group:
        print("overview_df exists but is not in h5py JSON format; returning empty DataFrame")
        return pd.DataFrame()

    raw_payload = df_group['json'][()]

    if isinstance(raw_payload, bytes):
        payload = raw_payload.decode('utf-8')
    else:
        payload = str(raw_payload)

    return pd.read_json(StringIO(payload), orient='split')

@dataclass
class ExperimentalDataset:
    """
    """

    experiments: Dict[str, 'Experiment'] = field(default_factory=dict)
    overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    plotting_instruction: Dict[str, Any] = field(default_factory=dict)
    group_mapping: Dict[str, Any] = field(default_factory=dict)
    processing_parameters: Dict[str, Any] = field(default_factory=dict)
    schema_version: Optional[str] = None

    def add_experiment(self, experimental_data: 'Experiment'):
        """Add an experiment to the dataset"""
        self.experiments[experimental_data.experiment_name] = experimental_data

    def update_overview_df(self,
                        incoming_df: pd.DataFrame,
                        key_column: str) -> None:
        """
        Merge an incoming overview DataFrame into the existing overview_df.

        Parameters
        ----------
        incoming_df : pd.DataFrame
            New overview data to merge in.
        key_column : str
            Column used to match experiments between the two DataFrames.

        Returns
        -------
        None
        """
        if self.overview_df.empty:
            self.overview_df = incoming_df.copy()
            self.overview_df["Processed"] = "False"
            return

        existing_df = self.overview_df.copy().set_index(key_column)
        incoming_df = incoming_df.copy().set_index(key_column)

        overlapping_keys = existing_df.index.intersection(incoming_df.index)

        existing_only = existing_df.drop(index=overlapping_keys)
        incoming_only = incoming_df.drop(index=overlapping_keys)

        # Looping over overlapping keys to check for differences and merge accordingly
        merged_rows = []
        for key in overlapping_keys:
            existing_row = existing_df.loc[key]
            incoming_row = incoming_df.loc[key]

            shared_cols = existing_row.index.intersection(incoming_row.index)
            shared_cols = shared_cols.drop("Processed", errors="ignore")

            # If shared columns are identical, merge by taking existing values and filling in any new columns from incoming
            if existing_row[shared_cols].equals(incoming_row[shared_cols]):
                merged_row = existing_row.combine_first(incoming_row)
            # If there are differences, use incoming row 
            else:
                merged_row = incoming_row

            merged_row.name = key
            merged_rows.append(merged_row)

        merged_df = pd.DataFrame(merged_rows)
        merged_df.index.name = key_column

        self.overview_df = pd.concat(
            [existing_only, incoming_only, merged_df],
            axis=0,
            sort=False
        ).reset_index()

    def save_to_hdf5(self, filename: str):
        """
        Save experiments to HDF5 file.
        """
        with h5py.File(filename, 'w') as f:
            if not self.overview_df.empty:
                write_df_to_hdf(f, self.overview_df, key='overview_df')

            # Always stamp the schema version so older / mismatched readers
            # can detect format drift.
            f.attrs['schema_version'] = SCHEMA_VERSION

            # Save dataset-level dictionaries as attributes
            if self.plotting_instruction:
                f.attrs['plotting_instruction'] = json.dumps(self.plotting_instruction)
            if self.group_mapping:
                f.attrs['group_mapping'] = json.dumps(self.group_mapping)
            if self.processing_parameters:
                f.attrs['processing_parameters'] = json.dumps(self.processing_parameters)
            
            for exp_name, experiment in self.experiments.items():

                if exp_name in f:
                    print(f"Experiment {exp_name} already exists. Overwriting...")
                    del f[exp_name]

                # Create a group for each experiment
                exp_grp = f.create_group(exp_name)

                exp_grp.attrs['experiment_name'] = experiment.experiment_name
                exp_grp.attrs['raw_data_file'] = experiment.raw_data_file
                exp_grp.attrs['color'] = experiment.color
                exp_grp.attrs['group'] = experiment.group

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

        with h5py.File(filename, 'r') as f:
            dataset.overview_df = read_df_from_hdf(f, key='overview_df')

            # Load schema version (None for legacy files written before
            # versioning existed).
            schema_version_attr = f.attrs.get('schema_version')
            if isinstance(schema_version_attr, bytes):
                dataset.schema_version = schema_version_attr.decode('utf-8')
            elif schema_version_attr is not None:
                dataset.schema_version = str(schema_version_attr)

            # Load dataset-level dictionaries from attributes
            if 'plotting_instruction' in f.attrs:
                dataset.plotting_instruction = json.loads(f.attrs['plotting_instruction'])
            if 'group_mapping' in f.attrs:
                dataset.group_mapping = json.loads(f.attrs['group_mapping'])
            if 'processing_parameters' in f.attrs:
                dataset.processing_parameters = json.loads(f.attrs['processing_parameters'])
            
            for exp_name in f.keys():
                if exp_name == 'overview_df':  # Skip the overview_df group
                    continue

                exp_group = f[exp_name]
                
                # Load simple attributes from the experiment group
                experiment_name = exp_group.attrs['experiment_name']
                raw_data_file = exp_group.attrs['raw_data_file'] 
                color = exp_group.attrs['color']
                group = exp_group.attrs.get('group', '')  # Default to empty string if not present
                
                # Load nested dictionaries
                raw_data = load_nested_dict_from_hdf5(exp_group['raw_data']) if 'raw_data' in exp_group else {}
                metadata = load_nested_dict_from_hdf5(exp_group['metadata']) if 'metadata' in exp_group else {}
                processed_data = load_nested_dict_from_hdf5(exp_group['processed_data']) if 'processed_data' in exp_group else {}
                
                single_experiment =  Experiment(
                    experiment_name=experiment_name,
                    raw_data_file=raw_data_file,
                    color=color,
                    group=group,
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

    @classmethod
    def merge_hdf5_files(cls, filenames: List[str], output_filename: str = None):
        """
        Merge multiple HDF5 files into a single ExperimentalDataset.
        
        Parameters
        ----------
        filenames : List[str]
            List of HDF5 file paths to merge
        output_filename : str, optional
            Path to save the merged dataset. If None, doesn't save.
            
        Returns
        -------
        ExperimentalDataset
            Merged dataset containing experiments from all files
            
        Raises
        ------
        ValueError
            If duplicate experiment names are found across files
            
        Examples
        --------
        >>> merged = ExperimentalDataset.merge_hdf5_files(
        ...     ['exp1.h5', 'exp2.h5', 'exp3.h5'],
        ...     output_filename='merged_experiments.h5'
        ... )
        """
        merged_dataset = cls()
        overview_dfs = []
        duplicate_experiments = []
        
        for filename in filenames:
            print(f"Loading {filename}...")
            temp_dataset = cls.load_from_hdf5(filename)
            
            # Check for duplicate experiment names
            for exp_name in temp_dataset.experiments.keys():
                if exp_name in merged_dataset.experiments:
                    duplicate_experiments.append((exp_name, filename))
                else:
                    merged_dataset.add_experiment(temp_dataset.experiments[exp_name])
            
            # Collect overview DataFrames
            if not temp_dataset.overview_df.empty:
                overview_dfs.append(temp_dataset.overview_df)

            # Merge plotting_instruction dictionaries
            if temp_dataset.plotting_instruction:
                merged_dataset.plotting_instruction.update(temp_dataset.plotting_instruction)

            # Merge group_mapping dictionaries
            if temp_dataset.group_mapping:
                merged_dataset.group_mapping.update(temp_dataset.group_mapping)

            # Merge processing_parameters dictionaries
            if temp_dataset.processing_parameters:
                merged_dataset.processing_parameters.update(temp_dataset.processing_parameters)
        
        # Report duplicates
        if duplicate_experiments:
            print("\nWarning: Found duplicate experiments (skipped):")
            for exp_name, filename in duplicate_experiments:
                print(f"  - '{exp_name}' in {filename}")
        
        # Merge overview DataFrames
        if overview_dfs:
            merged_dataset.overview_df = pd.concat(overview_dfs, ignore_index=True)
            # Remove duplicate rows if any
            merged_dataset.overview_df = merged_dataset.overview_df.drop_duplicates()
        
        print(f"\nMerged dataset contains {len(merged_dataset.experiments)} experiments")
        
        # Save if output filename provided
        if output_filename:
            print(f"Saving merged dataset to {output_filename}...")
            merged_dataset.save_to_hdf5(output_filename)
        
        return merged_dataset

def usage_example():
    """Demonstrate basic ExperimentalDataset functionality."""
    


    # Create a dataset and add an experiment
    dataset = ExperimentalDataset(overview_df=pd.DataFrame({
        'Experiment': ['Exp1', 'Exp2'],
        'Description': ['First experiment', 'Second experiment']
    }))
    
    # Add dataset-level attributes
    dataset.plotting_instruction = {'xlabel': 'Time (s)', 'ylabel': 'Current (mA)'}
    dataset.group_mapping = {'GroupA': ['Exp1'], 'GroupB': ['Exp2']}

    exp1 = Experiment(
        experiment_name="Exp1",
        raw_data_file="data/exp1.h5",
        color="blue",
        group="GroupA",
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
    dataset.save_to_hdf5("src/tests/experiments.h5")

    # Load from HDF5
    loaded_dataset = ExperimentalDataset.load_from_hdf5("src/tests/experiments.h5")
    loaded_dataset.print_experiments()
    
    print(f"Plotting instructions: {loaded_dataset.plotting_instruction}")
    print(f"Group mapping: {loaded_dataset.group_mapping}")
    print(f"Exp1 group: {loaded_dataset.experiments['Exp1'].group}")

    print(loaded_dataset.experiments['Exp1'].processed_data['baseline_corrected']['fit_parameters'])
    print(loaded_dataset.overview_df)


if __name__ == '__main__':
    usage_example()




