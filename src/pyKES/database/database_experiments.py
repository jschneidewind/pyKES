"""HDF5-backed storage for experimental datasets.

An `ExperimentalDataset` is the single object every other part of pyKES reads
from: the fitting code takes its experiments, the Streamlit pages render them,
and the analysis utilities write their results back into them. It holds a
mapping of `Experiment` objects, an overview DataFrame describing them, and
the dataset-level configuration an embedding app supplies (plotting
instructions, group mapping, processing parameters).

Each `Experiment` keeps three dictionaries side by side: the ``metadata`` read
from the overview sheet, the ``raw_data`` as measured, and the
``processed_data`` a processing function derived from them. Keeping the raw
data in the file is what makes `pyKES.database.data_processing.reprocess_experiments`
possible — an improved algorithm can be applied to a finished dataset without
going back to the original instrument files.

Datasets round-trip through HDF5. Nested dictionaries become nested groups,
NumPy arrays are stored natively, and anything else falls back to JSON and then
to pickle, so a processing function is free to return whatever structure suits
it. Keys containing ``'/'`` — common in metadata columns such as
``'Catalyst loading [wt% Rh/Cr]'`` — are escaped, since HDF5 would otherwise
read them as path separators.

Every file records a `SCHEMA_VERSION` and a provenance dictionary naming the
pyKES version, the embedding app's version and the relevant timestamps; see
`pyKES.utilities.version_information` and ``docs/versioning_and_reprocessing.md``.
"""

import pandas as pd
import numpy as np
import h5py
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Union, Optional
import json
import pickle
from io import StringIO

from pyKES.utilities.version_information import (
    build_version_information,
    describe_version_information,
    stamp_version_information,
)

# Bump when the on-disk layout changes in a way older readers cannot ignore
# (renamed/removed groups, changed required attributes). Purely additive
# changes to processing_parameters or per-experiment dicts do not require
# a bump.
# 1.1 adds the dataset-level 'version' attribute and the per-experiment
# 'version' attribute (both JSON, both optional for readers).
SCHEMA_VERSION = "1.1"


def import_overview_excel(file_name, 
                          sheet_name,
                          dtype = None):
    
    '''
    Read the overview sheet describing a set of experiments.

    Parameters
    ----------
    file_name : str
        Path to the Excel workbook.
    sheet_name : str
        Sheet holding the overview table.
    dtype : dict, optional
        Per-column dtypes handed to `pandas.read_excel`. Worth setting for
        columns pandas would otherwise guess wrongly — flag columns such as
        ``'Processed'`` are compared as the strings ``'True'`` / ``'False'``
        elsewhere and should be read with ``str``.

    Returns
    -------
    pandas.DataFrame
        The overview sheet.
    '''

    df = pd.read_excel(file_name, 
                       sheet_name = sheet_name, 
                       dtype = dtype)

    return df

@dataclass
class Experiment:
    """
    Store data and metadata for a single experiment.

    Parameters
    ----------
    experiment_name : str
        Unique name of the experiment within its dataset.
    raw_data_file : str
        Source the raw data was read from.
    color, group : str
        Display color and group used by the Streamlit pages.
    metadata, raw_data, processed_data : dict
        Experiment metadata, raw measurements, and the output of the
        processing function.
    version : dict, optional
        Provenance of the processing run that produced `processed_data`
        (pyKES version, timestamps, external app version). Written by the
        ingestion and reprocessing pipelines; empty for experiments read from
        files predating schema 1.1.
    """
    experiment_name: str
    raw_data_file: str
    color: str
    group: str
    metadata: Dict[str, any]
    raw_data: Dict[str, any]
    processed_data: Dict[str, any]
    version: Dict[str, Any] = field(default_factory=dict)

def _sanitize_hdf5_key(key: Any) -> str:
    """Convert dict keys to HDF5-safe strings without breaking nested paths."""
    key_str = key if isinstance(key, str) else str(key)
    return key_str.replace('/', '__SLASH__')


def save_nested_dict_to_hdf5(group, data_dict, prefix=""):
    """
    Write a nested dictionary into an HDF5 group.

    Nested dictionaries become nested HDF5 paths. NumPy arrays and scalars are
    stored natively; lists and tuples are converted to arrays where possible
    and serialized as JSON otherwise; anything left is pickled. The fallbacks
    exist because processing functions may return arbitrary structures, and a
    dataset that cannot be written is worse than one written inefficiently. The
    encoding used is recorded in the dataset's ``type`` attribute so
    `load_nested_dict_from_hdf5` can undo it.

    Parameters
    ----------
    group : h5py.Group
        Group written into.
    data_dict : dict
        Dictionary to store. Non-string keys are stringified.
    prefix : str, optional
        Path prefix within the group. Set by the recursion.

    Returns
    -------
    None

    Notes
    -----
    ``'/'`` in a key is replaced by ``'__SLASH__'``, since HDF5 would otherwise
    read it as a path separator and split the key into two groups. Metadata
    columns such as ``'Catalyst loading [wt% Rh/Cr]'`` make this a real case.
    """
    for key, value in data_dict.items():
        # Replace '/' in keys to avoid HDF5 path interpretation issues.
        # Data uploads can include integer keys in nested dicts; stringify them
        # before building an HDF5 path.
        safe_key = _sanitize_hdf5_key(key)
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
    Read a nested dictionary back out of an HDF5 group.

    Inverse of `save_nested_dict_to_hdf5`: the group is walked recursively, the
    encoding recorded on each dataset is undone, and the path of each dataset
    is split back into nested dictionary keys.

    Parameters
    ----------
    group : h5py.Group
        Group read from.
    prefix : str, optional
        Path prefix stripped from the dataset names before they become keys.

    Returns
    -------
    dict
        The reconstructed nested dictionary, with ``'__SLASH__'`` restored to
        ``'/'`` in the keys.
    """
    result = {}
    
    def visit_func(name, obj):
        """Rebuild one dataset into its place in the nested result."""
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
    Collection of experiments plus the dataset-level configuration.

    Parameters
    ----------
    experiments : dict
        Mapping of experiment name to `Experiment`.
    overview_df : pandas.DataFrame
        Overview sheet listing the experiments and their metadata.
    plotting_instruction, group_mapping, processing_parameters : dict
        Configuration supplied by the external app.
    version : dict
        Provenance of the dataset: pyKES version, schema version, creation and
        modification timestamps, and the external app's own version
        information. Filled in on the first save; see
        `pyKES.utilities.version_information`.
    schema_version : str, optional
        On-disk layout version the dataset was loaded from. None for datasets
        that have not been written yet or that come from files predating
        versioning.
    """

    experiments: Dict[str, 'Experiment'] = field(default_factory=dict)
    overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    plotting_instruction: Dict[str, Any] = field(default_factory=dict)
    group_mapping: Dict[str, Any] = field(default_factory=dict)
    processing_parameters: Dict[str, Any] = field(default_factory=dict)
    version: Dict[str, Any] = field(default_factory=dict)
    schema_version: Optional[str] = None

    def add_experiment(self, experimental_data: 'Experiment'):
        """
        Add an experiment to the dataset, keyed by its name.

        An experiment whose name is already present is replaced.

        Parameters
        ----------
        experimental_data : Experiment
            Experiment to add.

        Returns
        -------
        None
        """
        self.experiments[experimental_data.experiment_name] = experimental_data

    # -----------------------------------------------------------------
    # Version / provenance handling
    # -----------------------------------------------------------------

    def stamp_version(self,
                      processed: bool = False,
                      external_version: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Refresh the dataset's version dictionary.

        Called automatically on every save and by the processing pipelines;
        call it directly only to record provenance at some other moment.

        Parameters
        ----------
        processed : bool, default False
            Whether a processing function was just run, which additionally
            sets ``'last_processed'``.
        external_version : dict, optional
            Provenance of the external app (e.g. its own project version),
            merged into the existing ``'external_version'`` entry.

        Returns
        -------
        version : dict
            The updated version dictionary (also stored on the dataset).
        """
        if not self.version:
            self.version = build_version_information(SCHEMA_VERSION)

        self.version = stamp_version_information(self.version,
                                                 SCHEMA_VERSION,
                                                 processed=processed,
                                                 external_version=external_version)

        return self.version

    def set_external_version(self, external_version: Dict[str, Any]) -> None:
        """
        Record the external app's own version information.

        Intended for apps that embed pyKES and want their code version stored
        alongside the data::

            dataset.set_external_version({'app': 'photocat',
                                          'version': get_project_version(__file__)})

        Parameters
        ----------
        external_version : dict
            Arbitrary JSON-serializable provenance, merged into any existing
            entry.

        Returns
        -------
        None : None
            ``self.version['external_version']`` is updated in place.
        """
        self.stamp_version(external_version=external_version)

    def describe_version(self) -> str:
        """
        Render the dataset's version dictionary as a one-line summary.

        Returns
        -------
        description : str
            Human-readable provenance summary.
        """
        return describe_version_information(self.version)

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
        Write the whole dataset to an HDF5 file.

        The file is written from scratch, so it always reflects the dataset as
        it is now. The overview DataFrame, the schema version and the
        dataset-level configuration go into the file root; each experiment
        becomes a group holding its metadata, raw data and processed data. The
        dataset is version-stamped as part of the save, so every file records
        which code wrote it and when.

        Parameters
        ----------
        filename : str
            Path written to. An existing file is overwritten.

        Returns
        -------
        None
        """
        with h5py.File(filename, 'w') as f:
            if not self.overview_df.empty:
                write_df_to_hdf(f, self.overview_df, key='overview_df')

            # Always stamp the schema version so older / mismatched readers
            # can detect format drift.
            f.attrs['schema_version'] = SCHEMA_VERSION
            self.schema_version = SCHEMA_VERSION

            # Provenance of the file: which pyKES (and which external app)
            # wrote it, and when it was created / last touched.
            f.attrs['version'] = json.dumps(self.stamp_version())

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

                if experiment.version:
                    exp_grp.attrs['version'] = json.dumps(experiment.version)

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
        Read a dataset back from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to a file written by `save_to_hdf5`.

        Returns
        -------
        ExperimentalDataset
            The reconstructed dataset.

        Notes
        -----
        Files predating schema 1.1 carry neither a dataset-level nor a
        per-experiment ``version`` attribute; those fields come back empty
        rather than raising, so older files stay readable.
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
            if 'version' in f.attrs:
                dataset.version = json.loads(f.attrs['version'])
            
            for exp_name in f.keys():
                if exp_name == 'overview_df':  # Skip the overview_df group
                    continue

                exp_group = f[exp_name]
                
                # Load simple attributes from the experiment group
                experiment_name = exp_group.attrs['experiment_name']
                raw_data_file = exp_group.attrs['raw_data_file'] 
                color = exp_group.attrs['color']
                group = exp_group.attrs.get('group', '')  # Default to empty string if not present

                # Absent for experiments written before schema 1.1
                version = json.loads(exp_group.attrs['version']) if 'version' in exp_group.attrs else {}
                
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
                    processed_data=processed_data,
                    version=version
                )

                dataset.add_experiment(single_experiment)

        return dataset
    
    def list_experiments(self) -> List[str]:
        """
        Names of all experiments in the dataset.

        Returns
        -------
        list of str
            Experiment names, sorted alphabetically.
        """
        return sorted(self.experiments.keys())

    def print_experiments(self):
        """
        Print a numbered list of the experiments in the dataset.

        Returns
        -------
        None
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

            # Carry over the external provenance of every source file; the
            # merged dataset itself is stamped as newly created on save.
            source_external_version = (temp_dataset.version or {}).get('external_version') or {}
            if source_external_version:
                merged_dataset.stamp_version(external_version=source_external_version)
        
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
        
        merged_dataset.stamp_version()
        merged_dataset.version['merged_from'] = [str(filename) for filename in filenames]

        print(f"\nMerged dataset contains {len(merged_dataset.experiments)} experiments")
        
        # Save if output filename provided
        if output_filename:
            print(f"Saving merged dataset to {output_filename}...")
            merged_dataset.save_to_hdf5(output_filename)
        
        return merged_dataset

def usage_example():
    """
    Build, save and reload a small dataset.

    Covers the round trip the class exists for, including the parts that are
    easy to get wrong: nested dictionaries inside ``processed_data``, NumPy
    arrays in ``raw_data``, and dataset-level configuration stored as file
    attributes.

    Returns
    -------
    None
        Writes ``src/tests/experiments.h5`` and prints what it reads back.
    """
    


    # Create a dataset and add an experiment
    dataset = ExperimentalDataset(overview_df=pd.DataFrame({
        'Experiment': ['Exp1', 'Exp2'],
        'Description': ['First experiment', 'Second experiment']
    }))
    
    # Add dataset-level attributes
    dataset.plotting_instruction = {'xlabel': 'Time (s)', 'ylabel': 'Current (mA)'}
    dataset.group_mapping = {'GroupA': ['Exp1'], 'GroupB': ['Exp2']}
    dataset.set_external_version({'app': 'usage_example', 'version': '0.3.0'})

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
    
    print(f"Version information: {loaded_dataset.describe_version()}")
    print(f"Plotting instructions: {loaded_dataset.plotting_instruction}")
    print(f"Group mapping: {loaded_dataset.group_mapping}")
    print(f"Exp1 group: {loaded_dataset.experiments['Exp1'].group}")

    print(loaded_dataset.experiments['Exp1'].processed_data['baseline_corrected']['fit_parameters'])
    print(loaded_dataset.overview_df)


if __name__ == '__main__':
    usage_example()




