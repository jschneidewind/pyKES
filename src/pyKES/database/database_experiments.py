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
# 1.2 adds the dataset-level 'experiment_column' attribute and the guarantee
# that per-experiment metadata mirrors the overview row (see
# `ExperimentalDataset`). Both are additive: a 1.1 reader ignores the
# attribute, and a 1.2 file loads in a 1.1 reader unchanged.
SCHEMA_VERSION = "1.2"

# Compression is applied only to arrays of at least this many elements. HDF5
# refuses to compress scalar datasets outright, and on very small arrays the
# filter costs more than it saves.
COMPRESSION_MINIMUM_ELEMENTS = 128

# gzip level used when a caller asks for compression without naming a level.
# Measured on a real 44-experiment dataset: level 4 costs about 1 ms per
# experiment and saves a third of the bytes.
DEFAULT_COMPRESSION_LEVEL = 4

# The processed flag lives in overview_df as text, not as a bool: it round-trips
# through Excel and HDF5 and comes back as these strings (see
# `pyKES.database.data_processing.ensure_processed_column`).
PROCESSED_FLAG_COLUMN = 'Processed'
PROCESSED_TRUE = 'True'
PROCESSED_FALSE = 'False'

# Overview column naming the experiments, unless a dataset says otherwise. The
# ingestion entry points and the Streamlit configuration default to the same
# name, so a dataset built either way lines up with the sheet.
DEFAULT_EXPERIMENT_COLUMN = 'Experiment'


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

# Metadata keys routinely contain slashes — 'Catalyst concentration [g/L]',
# 'Irradiance A [mW/cm2]' — while '/' is also the separator of a nested path.
# Escaping the one inside the other keeps a joined path splittable, which is
# what both the HDF5 layout and any index built over it rely on.
KEY_SLASH_PLACEHOLDER = '__SLASH__'


def sanitize_key(key: Any) -> str:
    """
    Escape a dict key so it can be joined into a slash-separated path.

    Parameters
    ----------
    key : Any
        Key to escape; non-strings are stringified, since uploads can carry
        integer keys in nested dictionaries.

    Returns
    -------
    escaped : str
        Key with any slash replaced by `KEY_SLASH_PLACEHOLDER`.
    """
    key_str = key if isinstance(key, str) else str(key)

    return key_str.replace('/', KEY_SLASH_PLACEHOLDER)


def restore_key(key: str) -> str:
    """
    Reverse `sanitize_key` on one component of a split path.

    Applied per component *after* splitting on the separator, never to a whole
    path — the whole point is that the escaped slashes survive the split.

    Parameters
    ----------
    key : str
        One escaped path component.

    Returns
    -------
    restored : str
        Key as it was originally written.
    """
    return key.replace(KEY_SLASH_PLACEHOLDER, '/')


def compression_arguments(value, compression):
    """
    Decide the h5py compression keywords for one value.

    Parameters
    ----------
    value : Any
        Value about to be written as a dataset.
    compression : str or None
        Compression filter requested by the caller, e.g. ``'gzip'``.

    Returns
    -------
    keywords : dict
        Keyword arguments for ``create_dataset``; empty when the value is too
        small to be worth compressing or no compression was requested.
    """
    if compression is None:
        return {}

    if not isinstance(value, np.ndarray) or value.ndim == 0:
        return {}

    if value.size < COMPRESSION_MINIMUM_ELEMENTS:
        return {}

    return {'compression': compression, 'compression_opts': DEFAULT_COMPRESSION_LEVEL}


def save_nested_dict_to_hdf5(group, data_dict, prefix="", compression=None):
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
    compression : str or None, optional
        Compression filter passed to h5py, e.g. ``'gzip'``. Applied only to
        arrays of at least `COMPRESSION_MINIMUM_ELEMENTS` elements; see
        `compression_arguments`.

    Returns
    -------
    None

    Notes
    -----
    ``'/'`` in a key is replaced by `KEY_SLASH_PLACEHOLDER`, since HDF5 would
    otherwise read it as a path separator and split the key into two groups.
    Metadata columns such as ``'Catalyst loading [wt% Rh/Cr]'`` make this a
    real case. `restore_key` reverses it, per path component.
    """
    for key, value in data_dict.items():
        # Replace '/' in keys to avoid HDF5 path interpretation issues.
        # Data uploads can include integer keys in nested dicts; stringify them
        # before building an HDF5 path.
        safe_key = sanitize_key(key)
        full_key = f"{prefix}/{safe_key}" if prefix else safe_key

        if isinstance(value, np.ndarray):
            # Save numpy arrays directly
            group.create_dataset(full_key, data=value,
                                 **compression_arguments(value, compression))

        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            save_nested_dict_to_hdf5(group, value, full_key, compression)
            
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


def processing_relevant_change(existing_row: pd.Series,
                               incoming_row: pd.Series,
                               processing_relevant_columns: List[str]) -> bool:
    """
    Report whether two overview rows disagree on any processing-relevant column.

    Parameters
    ----------
    existing_row, incoming_row : pandas.Series
        Rows of the stored and the incoming overview sheet, indexed by column.
    processing_relevant_columns : list of str
        Columns whose value the processing function depends on.

    Returns
    -------
    bool
        True when at least one declared column differs. A column the incoming
        sheet adds counts as a change, since the processing function would
        read a value where it previously read nothing; a column neither row
        carries is ignored.
    """

    declared_columns = [column for column in processing_relevant_columns
                        if column in existing_row.index or column in incoming_row.index]

    return not existing_row.reindex(declared_columns).equals(
        incoming_row.reindex(declared_columns))


def merge_overview_row(existing_row: pd.Series,
                       incoming_row: pd.Series,
                       processing_relevant_columns: Optional[List[str]] = None) -> pd.Series:
    """
    Merge one overview row that both the stored and the incoming sheet hold.

    Parameters
    ----------
    existing_row, incoming_row : pandas.Series
        The two versions of the row, indexed by column.
    processing_relevant_columns : list of str, optional
        Columns whose change clears the processed flag; see
        `ExperimentalDataset.update_overview_df`.

    Returns
    -------
    merged_row : pandas.Series
        The stored row extended by the sheet's new columns when the two agree,
        otherwise the incoming row carrying an explicit processed flag.
    """

    shared_columns = existing_row.index.intersection(incoming_row.index)
    shared_columns = shared_columns.drop(PROCESSED_FLAG_COLUMN, errors="ignore")

    if existing_row[shared_columns].equals(incoming_row[shared_columns]):
        return existing_row.combine_first(incoming_row)

    # The incoming sheet is authoritative for the values but carries no
    # processing history, so the flag has to be decided here. Left to
    # `combine_first` it would come back as NaN, which
    # `select_unprocessed_experiments` reads as unprocessed only by accident.
    merged_row = incoming_row.copy()

    if (processing_relevant_columns is None
            or processing_relevant_change(existing_row, incoming_row,
                                          processing_relevant_columns)):
        merged_row[PROCESSED_FLAG_COLUMN] = PROCESSED_FALSE
    else:
        merged_row[PROCESSED_FLAG_COLUMN] = existing_row.get(PROCESSED_FLAG_COLUMN,
                                                             PROCESSED_FALSE)

    return merged_row


def decode_hdf5_text(attribute_value) -> str:
    """
    Read an HDF5 string attribute as text.

    Parameters
    ----------
    attribute_value : bytes or str
        Value as h5py hands it over, which depends on how it was written.

    Returns
    -------
    str
        The attribute as text.
    """

    if isinstance(attribute_value, bytes):
        return attribute_value.decode('utf-8')

    return str(attribute_value)


# =============================================================================
# The overview sheet as the single source of experiment metadata
# =============================================================================

# Columns of the overview sheet that are not metadata. The processed flag is a
# derived, pipeline-owned value: it changes when an experiment is processed,
# without anything about the experiment's metadata changing, so policing it as
# metadata would report a divergence after every ingestion run.
NON_METADATA_OVERVIEW_COLUMNS = (PROCESSED_FLAG_COLUMN,)

# Stands in for a column the stored metadata does not carry at all, so a
# divergence report can tell 'absent' from 'present and None'.
METADATA_KEY_ABSENT = '<absent>'


def values_agree(stored_value, overview_value) -> bool:
    """
    Report whether a stored metadata value still matches its overview cell.

    Parameters
    ----------
    stored_value, overview_value : Any
        The value held in ``Experiment.metadata`` and the one in the overview
        row.

    Returns
    -------
    bool
        True when the two describe the same value.

    Notes
    -----
    Compared by value rather than by identity or type, because the same cell
    comes back as a NumPy scalar from a DataFrame, as a plain Python scalar
    from JSON and as either from HDF5. Two missing values agree: a blank cell
    that round-trips as NaN has not diverged from one stored as NaN.
    """

    if pd.isna(stored_value) and pd.isna(overview_value):
        return True

    return bool(stored_value == overview_value)


def overview_metadata_for_experiment(overview_df: pd.DataFrame,
                                     experiment_column: str,
                                     experiment_name: str) -> dict:
    """
    Read the metadata one experiment's overview row defines.

    Parameters
    ----------
    overview_df : pandas.DataFrame
        Overview sheet of the dataset.
    experiment_column : str
        Column naming the experiments.
    experiment_name : str
        Experiment to look up.

    Returns
    -------
    dict
        ``{column: value}`` for every overview column the sheet owns, empty
        when the sheet has no row for this experiment — which is the normal
        state of an experiment merged in from another file, and means the
        sheet makes no claim about its metadata.
    """

    if overview_df.empty or experiment_column not in overview_df.columns:
        return {}

    # Compared as text because an overview sheet may name its experiments with
    # numbers, which pandas then holds as ints while the experiments dict is
    # keyed by the string they were added under.
    matching_rows = overview_df.loc[
        overview_df[experiment_column].astype(str).eq(str(experiment_name))]

    if matching_rows.empty:
        return {}

    return {column: value for column, value in matching_rows.iloc[0].to_dict().items()
            if column not in NON_METADATA_OVERVIEW_COLUMNS}


def experiment_metadata_divergences(experiment: 'Experiment', overview_metadata: dict) -> dict:
    """
    Find where one experiment's stored metadata contradicts its overview row.

    Parameters
    ----------
    experiment : Experiment
        Experiment whose ``metadata`` is checked.
    overview_metadata : dict
        What the overview row says, as returned by
        `overview_metadata_for_experiment`.

    Returns
    -------
    dict
        ``{column: (stored_value, overview_value)}`` for every disagreeing
        column, with `METADATA_KEY_ABSENT` as the stored value where the
        metadata does not carry the column at all.
    """

    divergences = {}

    for column, overview_value in overview_metadata.items():
        if column not in experiment.metadata:
            divergences[column] = (METADATA_KEY_ABSENT, overview_value)
        elif not values_agree(experiment.metadata[column], overview_value):
            divergences[column] = (experiment.metadata[column], overview_value)

    return divergences


def apply_overview_metadata(experiment: 'Experiment', overview_metadata: dict) -> None:
    """
    Overwrite one experiment's overview-owned metadata with the sheet's values.

    Parameters
    ----------
    experiment : Experiment
        Experiment mutated in place.
    overview_metadata : dict
        What the overview row says.

    Returns
    -------
    None : None

    Notes
    -----
    Only the overview columns are written, so keys a
    ``metadata_retrival_function`` adds of its own — the reference one adds
    ``'experiment_name'`` — survive. ``color`` and ``group`` are re-derived
    from the refreshed metadata, but only from a value the sheet actually
    supplies: a blank colour cell arrives as NaN, and assigning that would
    replace a usable colour with one no plotting library accepts.
    """

    experiment.metadata.update(overview_metadata)

    for attribute in ('color', 'group'):
        value = experiment.metadata.get(attribute, None)

        if value is not None and not pd.isna(value):
            setattr(experiment, attribute, value)


def describe_metadata_divergences(divergences: dict, maximum_reported: int = 5) -> str:
    """
    Render a divergence report as one line of prose.

    Parameters
    ----------
    divergences : dict
        ``{experiment_name: {column: (stored_value, overview_value)}}``.
    maximum_reported : int, optional
        How many experiments to name before summarising the rest, so a
        dataset-wide mismatch does not produce a wall of text.

    Returns
    -------
    str
        Human-readable summary, empty when there is nothing to report.
    """

    if not divergences:
        return ''

    reported_names = sorted(divergences)[:maximum_reported]

    described = '; '.join(
        f"{experiment_name}: "
        + ', '.join(f"{column} {stored!r} -> {expected!r}"
                    for column, (stored, expected) in sorted(divergences[experiment_name].items()))
        for experiment_name in reported_names)

    remaining = len(divergences) - len(reported_names)

    return described + (f"; and {remaining} more experiment(s)" if remaining else '')

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
    experiment_column : str, optional
        Column of ``overview_df`` naming the experiments. The dataset needs it
        to know which row belongs to which experiment, which is what makes the
        metadata invariant below enforceable.
    metadata_repair_report : dict
        What the last `synchronize_experiment_metadata` corrected, as
        ``{experiment_name: {column: (stale_value, overview_value)}}``. Filled
        in by `load_from_hdf5` so a caller can tell the user that a file it
        just opened had diverged. Not persisted.

    Notes
    -----
    ``overview_df`` is the single source of every experiment's metadata: for
    every column of the sheet, ``Experiment.metadata`` holds exactly what the
    experiment's row holds. Every mutation path re-establishes that
    (`add_experiment`, `update_overview_df`, the editing and reprocessing
    pipelines), `load_from_hdf5` repairs files that predate the guarantee, and
    `save_to_hdf5` refuses to write a dataset that breaks it. Keys a
    ``metadata_retrival_function`` adds that are not overview columns are the
    app's own and are left alone.
    """

    experiments: Dict[str, 'Experiment'] = field(default_factory=dict)
    overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    plotting_instruction: Dict[str, Any] = field(default_factory=dict)
    group_mapping: Dict[str, Any] = field(default_factory=dict)
    processing_parameters: Dict[str, Any] = field(default_factory=dict)
    version: Dict[str, Any] = field(default_factory=dict)
    schema_version: Optional[str] = None
    experiment_column: str = DEFAULT_EXPERIMENT_COLUMN
    metadata_repair_report: Dict[str, Any] = field(default_factory=dict)

    def add_experiment(self, experimental_data: 'Experiment') -> Dict[str, Any]:
        """
        Add an experiment to the dataset, keyed by its name.

        An experiment whose name is already present is replaced.

        Every experiment enters a dataset through here, so this is where the
        overview sheet is imposed on the metadata the caller brought: a
        processing pipeline that transformed an overview column on its way in
        gets that column put back, rather than the two versions drifting apart
        unnoticed.

        Parameters
        ----------
        experimental_data : Experiment
            Experiment to add; its ``metadata`` is aligned with its overview
            row in place.

        Returns
        -------
        dict
            What had to be corrected to satisfy the invariant, in the shape
            `metadata_divergences` returns. Empty for the usual case of an
            experiment whose metadata already matches its row.
        """
        self.experiments[experimental_data.experiment_name] = experimental_data

        overview_metadata = overview_metadata_for_experiment(
            self.overview_df, self.experiment_column, experimental_data.experiment_name)

        divergences = experiment_metadata_divergences(experimental_data, overview_metadata)

        apply_overview_metadata(experimental_data, overview_metadata)

        return divergences

    # -----------------------------------------------------------------
    # Metadata / overview consistency
    # -----------------------------------------------------------------

    def metadata_divergences(self, experiment_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Find experiments whose stored metadata contradicts the overview sheet.

        Parameters
        ----------
        experiment_names : list of str, optional
            Experiments to check. Defaults to every experiment in the dataset.

        Returns
        -------
        dict
            ``{experiment_name: {column: (stored_value, overview_value)}}``,
            empty when the dataset is consistent.
        """

        if experiment_names is None:
            experiment_names = list(self.experiments)

        divergences = {}

        for experiment_name in experiment_names:
            experiment = self.experiments[experiment_name]
            experiment_divergences = experiment_metadata_divergences(
                experiment,
                overview_metadata_for_experiment(self.overview_df,
                                                 self.experiment_column,
                                                 experiment_name))

            if experiment_divergences:
                divergences[experiment_name] = experiment_divergences

        return divergences

    def synchronize_experiment_metadata(self,
                                        experiment_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Re-derive stored metadata from the overview sheet, and report the fixes.

        Parameters
        ----------
        experiment_names : list of str, optional
            Experiments to synchronize. Defaults to every experiment in the
            dataset. Names without an overview row are left untouched.

        Returns
        -------
        dict
            What was corrected, in the shape `metadata_divergences` returns.
        """

        divergences = self.metadata_divergences(experiment_names)

        for experiment_name in divergences:
            apply_overview_metadata(
                self.experiments[experiment_name],
                overview_metadata_for_experiment(self.overview_df,
                                                 self.experiment_column,
                                                 experiment_name))

        return divergences

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
                        key_column: str,
                        processing_relevant_columns: Optional[List[str]] = None) -> None:
        """
        Merge an incoming overview DataFrame into the existing overview_df.

        The incoming sheet takes precedence: for every row both sides hold,
        the sheet's values win over whatever is stored, including edits made
        in the app. That is what makes re-uploading a corrected workbook the
        way to undo an editing session. The stored metadata of the affected
        experiments is re-derived at the end, so the sheet reaches the
        experiments and not just the overview table.

        Parameters
        ----------
        incoming_df : pd.DataFrame
            New overview data to merge in.
        key_column : str
            Column used to match experiments between the two DataFrames. Also
            adopted as the dataset's ``experiment_column``, since the sheet
            being merged is what defines how rows are addressed.
        processing_relevant_columns : list of str, optional
            Columns whose change invalidates the processed data. When given,
            a row that differs only in other columns keeps its processed
            flag, so re-uploading a sheet with a corrected comment does not
            invalidate a processing run. Defaults to treating every
            difference as processing-relevant.

        Returns
        -------
        None
        """
        self.experiment_column = key_column

        if self.overview_df.empty:
            self.overview_df = incoming_df.copy()
            self.overview_df[PROCESSED_FLAG_COLUMN] = PROCESSED_FALSE
            self.synchronize_experiment_metadata()
            return

        existing_df = self.overview_df.copy().set_index(key_column)
        incoming_df = incoming_df.copy().set_index(key_column)

        overlapping_keys = existing_df.index.intersection(incoming_df.index)

        existing_only = existing_df.drop(index=overlapping_keys)
        incoming_only = incoming_df.drop(index=overlapping_keys)

        merged_rows = []
        for key in overlapping_keys:
            merged_row = merge_overview_row(existing_df.loc[key],
                                            incoming_df.loc[key],
                                            processing_relevant_columns)
            merged_row.name = key
            merged_rows.append(merged_row)

        merged_df = pd.DataFrame(merged_rows)
        merged_df.index.name = key_column

        self.overview_df = pd.concat(
            [existing_only, incoming_only, merged_df],
            axis=0,
            sort=False
        ).reset_index()

        self.synchronize_experiment_metadata()

    def save_to_hdf5(self, filename: str, compression: Optional[str] = None,
                     verbose: bool = True):
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
        compression : str or None, optional
            Compression filter for array datasets, e.g. ``'gzip'``. Compression
            is transparent to readers, so a compressed file loads unchanged and
            needs no `SCHEMA_VERSION` bump.
        verbose : bool, optional
            Print one line per experiment written. Set ``False`` when writing
            one file per experiment in a loop, where the per-experiment print
            is noise rather than progress.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any experiment's metadata contradicts its overview row. Every
            mutation path re-establishes that invariant, so reaching a save
            with it broken means something wrote to ``Experiment.metadata``
            behind the dataset's back — and writing the file would put the
            contradiction on disk, where it has already cost users their
            edits. `synchronize_experiment_metadata` resolves it in favour of
            the sheet.
        """
        divergences = self.metadata_divergences()

        if divergences:
            raise ValueError(
                "Refusing to save: the stored metadata of "
                f"{len(divergences)} experiment(s) contradicts overview_df. "
                f"{describe_metadata_divergences(divergences)} "
                "Call synchronize_experiment_metadata() to adopt the overview "
                "values, or correct overview_df.")

        with h5py.File(filename, 'w') as f:
            if not self.overview_df.empty:
                write_df_to_hdf(f, self.overview_df, key='overview_df')

            # Always stamp the schema version so older / mismatched readers
            # can detect format drift.
            f.attrs['schema_version'] = SCHEMA_VERSION
            self.schema_version = SCHEMA_VERSION

            # Which overview column names the experiments. Stored so a reader
            # can line the sheet up with the experiment groups without being
            # told, which is what the metadata invariant rests on.
            f.attrs['experiment_column'] = self.experiment_column

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
                    save_nested_dict_to_hdf5(raw_data_group, experiment.raw_data,
                                             compression=compression)

                if experiment.metadata:
                    metadata_group = exp_grp.create_group('metadata')
                    save_nested_dict_to_hdf5(metadata_group, experiment.metadata,
                                             compression=compression)

                if experiment.processed_data:
                    processed_data_group = exp_grp.create_group('processed_data')
                    save_nested_dict_to_hdf5(processed_data_group, experiment.processed_data,
                                             compression=compression)

                if verbose:
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

            # Absent before schema 1.2, where every dataset used the default.
            # Read before the experiments, since aligning their metadata with
            # the sheet depends on knowing which column addresses the rows.
            if 'experiment_column' in f.attrs:
                dataset.experiment_column = decode_hdf5_text(f.attrs['experiment_column'])

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

                repaired_columns = dataset.add_experiment(single_experiment)

                # Files written before the invariant existed can hold metadata
                # the overview sheet contradicts; adding the experiment fixes
                # it, and the report lets a caller say so.
                if repaired_columns:
                    dataset.metadata_repair_report[experiment_name] = repaired_columns

        if dataset.metadata_repair_report:
            print("Stored metadata disagreed with overview_df and was re-derived from it. "
                  + describe_metadata_divergences(dataset.metadata_repair_report))

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

            # Every source file addresses its rows the same way in practice;
            # adopting the last one read keeps the merged sheet addressable
            # rather than falling back to the default when it does not apply.
            merged_dataset.experiment_column = temp_dataset.experiment_column

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

        # The experiments were added before their rows arrived, so nothing had
        # a sheet to be aligned with at the time.
        merged_dataset.synchronize_experiment_metadata()

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




