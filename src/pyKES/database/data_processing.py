import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import traceback
from typing import Callable, Optional
from pathlib import Path

from pyKES.database.database_experiments import ExperimentalDataset, Experiment, SCHEMA_VERSION
from pyKES.utilities.version_information import stamp_version_information


def stamp_experiment_version(experiment: Experiment,
                             external_version: Optional[dict] = None) -> None:
    """
    Record on an experiment which code produced its processed data.

    Parameters
    ----------
    experiment : Experiment
        Experiment whose ``version`` dict is updated in place. On a first run
        the dict is created; on a reprocessing run its ``created`` timestamp
        is preserved and ``last_processed`` is refreshed.
    external_version : dict, optional
        Provenance of the external app supplying the processing functions.

    Returns
    -------
    None : None
    """
    stamp_version_information(experiment.version,
                              SCHEMA_VERSION,
                              processed=True,
                              external_version=external_version)


def resolve_external_version(database: ExperimentalDataset,
                             external_version: Optional[dict]) -> Optional[dict]:
    """
    Choose the external provenance to stamp onto processed experiments.

    Parameters
    ----------
    database : ExperimentalDataset
        Dataset whose ``version['external_version']`` acts as the default, so
        an app that called `ExperimentalDataset.set_external_version` once does
        not have to repeat itself on every processing call.
    external_version : dict or None
        Explicitly supplied provenance, which takes precedence. An empty dict
        counts as "not supplied", so a config that simply leaves the field at
        its default does not suppress the dataset's own entry.

    Returns
    -------
    external_version : dict or None
        Provenance to record, or None when neither source provides one.
    """
    if external_version:
        return external_version

    return (database.version or {}).get('external_version') or None

def generate_list_of_files(keywords, directory):

    files = [
        os.path.join(directory, file) 
        for file in os.listdir(directory) 
        if any(keyword in file for keyword in keywords)
        and not file.startswith('~$')
    ]

    return files

def read_in_single_experiment(file_name: str,
                              database: ExperimentalDataset,
                              metadata_retrival_function: callable, 
                              raw_data_reading_function: callable,
                              processing_function: callable,
                              directory: Optional[Path] = None,
                              legacy_mode = True,
                              external_version: Optional[dict] = None):
    """
    Legacy mode is for use with file-based processing and use in multi-processing mode.

    Non-legacy mode is for use with overview_df-based processing in single-threaded mode, 
    where the file name is not necessarily the key to retrieve metadata and raw data.
    In this case, the file name is used as an argument to the metadata retrieval function,
    which then retrieves the necessary metadata and file paths for raw data reading and processing.


    """
    
    try:
        if legacy_mode:
            metadata_dict = metadata_retrival_function(file_name, database.overview_df)
            raw_data_dict = raw_data_reading_function(file_name, metadata_dict)
            processed_data_dict = processing_function(raw_data_dict, metadata_dict)

        else:
            metadata_dict = metadata_retrival_function(file_name, database.overview_df)
            raw_data_dict = raw_data_reading_function(directory, metadata_dict)
            processed_data_dict = processing_function(raw_data_dict, metadata_dict)

        experiment = Experiment(
            experiment_name = metadata_dict['experiment_name'],
            raw_data_file = file_name,
            color = metadata_dict.get('color', 'black'),
            group = metadata_dict.get('group', 'default'),
            metadata = metadata_dict,
            raw_data = raw_data_dict,
            processed_data = processed_data_dict
        )

        stamp_experiment_version(experiment, external_version)

        return {
            'success': True,
            'data': experiment
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f'{file_name} analysis failed, not added to dataset, error: {str(e)}')
        print("Full traceback:")
        print(tb)

        return {
            'success': False,
            'file': file_name,
            'error': f"{str(e)}\n\nFull traceback:\n{tb}"
        }


def ensure_processed_column(database: ExperimentalDataset) -> None:
    """
    Make sure ``overview_df`` carries the ``Processed`` flag as text.

    Idempotent, and called by both entry points that touch the flag, so
    neither depends on the other having run first.

    Parameters
    ----------
    database : ExperimentalDataset
        Dataset whose ``overview_df`` is given the column if it lacks one.

    Returns
    -------
    None : None
    """

    # The flag is text, not a bool: it round-trips through Excel and HDF5, and
    # comes back as the strings this module compares against. Seeding it with a
    # bool gives the column bool dtype, which pandas then refuses to write
    # 'True' into once the first experiment succeeds.
    if "Processed" not in database.overview_df.columns:
        database.overview_df["Processed"] = "False"


def select_unprocessed_experiments(database: ExperimentalDataset,
                                   overview_df_experiment_column: Optional[str] = 'Experiment') -> list:
    """
    List the experiments of the overview sheet that still need processing.

    Parameters
    ----------
    database : ExperimentalDataset
        Dataset whose ``overview_df`` lists the experiments. A missing
        ``Processed`` column is added to it in place.
    overview_df_experiment_column : str, optional
        Column of ``overview_df`` holding the experiment names.

    Returns
    -------
    list of str
        Experiments that are either not flagged as processed or not yet held
        by ``database.experiments``.
    """

    ensure_processed_column(database)

    unprocessed = (
        database.overview_df["Processed"].ne('True')
        | ~database.overview_df[overview_df_experiment_column].isin(database.experiments))

    return database.overview_df.loc[unprocessed,
                    overview_df_experiment_column].astype(str).tolist()


def ingest_experiment(experiment_name: str,
                      database: ExperimentalDataset,
                      metadata_retrival_function: callable,
                      raw_data_reading_function: callable,
                      processing_function: callable,
                      overview_df_experiment_column: Optional[str] = 'Experiment',
                      directory: Optional[Path] = None,
                      external_version: Optional[dict] = None) -> dict:
    """
    Process one experiment of the overview sheet and add it to the dataset.

    One step of `read_in_experiments_single_threaded`, exposed separately so a
    caller can drive the ingestion one experiment at a time — which is what the
    Streamlit page does to keep its progress bar alive in the browser.

    Parameters
    ----------
    experiment_name : str
        Experiment to process, as named in ``overview_df``.
    database : ExperimentalDataset
        Dataset the experiment is added to; mutated in place on success.
    metadata_retrival_function : callable
        Returns the metadata dict for an experiment name.
    raw_data_reading_function : callable
        Reads the raw data for an experiment from ``directory``.
    processing_function : callable
        Turns raw data and metadata into the processed-data dict.
    overview_df_experiment_column : str, optional
        Column of ``overview_df`` holding the experiment names.
    directory : Path, optional
        Directory the raw-data files are read from.
    external_version : dict, optional
        Provenance of the external app, stamped onto the experiment. Already
        resolved — unlike the loop functions, this is not defaulted to the
        dataset's own entry.

    Returns
    -------
    result : dict
        Result of `read_in_single_experiment`.
    """

    ensure_processed_column(database)

    result = read_in_single_experiment(
        file_name = experiment_name,
        database = database,
        metadata_retrival_function = metadata_retrival_function,
        raw_data_reading_function = raw_data_reading_function,
        processing_function = processing_function,
        directory = directory,
        legacy_mode = False,
        external_version = external_version
    )

    if not result['success']:
        print(f"Failed to process {result['file']}: {result['error']}")
        return result

    database.add_experiment(result['data'])

    database.overview_df.loc[
        database.overview_df[overview_df_experiment_column].eq(result['data'].experiment_name),
            "Processed",
        ] = 'True'

    return result


def finalize_processing_run(database: ExperimentalDataset,
                            results: list,
                            external_version: Optional[dict] = None) -> None:
    """
    Stamp the dataset version after a processing or reprocessing run.

    Parameters
    ----------
    database : ExperimentalDataset
        Dataset whose ``version`` dict is updated in place.
    results : list of dict
        Results of the run; the dataset is only stamped when at least one
        experiment succeeded.
    external_version : dict, optional
        Provenance of the external app.

    Returns
    -------
    None : None
    """

    if any(result['success'] for result in results):
        database.stamp_version(processed=True, external_version=external_version)


def read_in_experiments_single_threaded(database: ExperimentalDataset,
                                        metadata_retrival_function: callable,
                                        raw_data_reading_function: callable,
                                        processing_function: callable,
                                        overview_df_experiment_column: Optional[str] = 'Experiment',
                                        directory: Optional[Path] = None,
                                        external_version: Optional[dict] = None,
                                        progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None):
    """
    Process every experiment of the overview dataframe that is not yet in the database.

    Parameters
    ----------
    database : ExperimentalDataset
        Dataset whose ``overview_df`` lists the experiments; successfully
        processed experiments are added to it in place.
    metadata_retrival_function : callable
        Returns the metadata dict for an experiment name.
    raw_data_reading_function : callable
        Reads the raw data for an experiment from ``directory``.
    processing_function : callable
        Turns raw data and metadata into the processed-data dict.
    overview_df_experiment_column : str, optional
        Column of ``overview_df`` holding the experiment names.
    directory : Path, optional
        Directory the raw-data files are read from.
    external_version : dict, optional
        Provenance of the external app (e.g. its git commit), stamped onto
        every processed experiment. Defaults to the dataset's own
        ``version['external_version']``.
    progress_callback : callable, optional
        Called as ``(completed, total, experiment_name)`` once before the loop
        with ``(0, total, None)`` — so callers can display the total before the
        first, potentially slow, experiment — and again after each experiment.

    Returns
    -------
    list of dict
        One result dict per processed experiment, as returned by
        `read_in_single_experiment`.
    """

    experiments = select_unprocessed_experiments(database, overview_df_experiment_column)

    total_experiments = len(experiments)
    external_version = resolve_external_version(database, external_version)

    if progress_callback is not None:
        progress_callback(0, total_experiments, None)

    results = []

    for completed, experiment_name in enumerate(experiments, start=1):

        results.append(ingest_experiment(
            experiment_name = experiment_name,
            database = database,
            metadata_retrival_function = metadata_retrival_function,
            raw_data_reading_function = raw_data_reading_function,
            processing_function = processing_function,
            overview_df_experiment_column = overview_df_experiment_column,
            directory = directory,
            external_version = external_version
        ))

        if progress_callback is not None:
            progress_callback(completed, total_experiments, experiment_name)

    finalize_processing_run(database, results, external_version)

    return results


def read_in_experiments_multiprocessing(database: ExperimentalDataset,
                                        metadata_retrival_function: callable,
                                        raw_data_reading_function: callable,
                                        processing_function: callable,
                                        keywords: Optional[list] = None, 
                                        directory: Optional[str] = None,
                                        overview_df_based_processing: Optional[bool] = False,
                                        overview_df_experiment_column: Optional[str] = 'Experiment',
                                        external_version: Optional[dict] = None): 
    """
    
    """

    if overview_df_based_processing:
        files = database.overview_df[overview_df_experiment_column].tolist()
    else:
        files = generate_list_of_files(keywords, directory)

    read_in_single_experiment_partial = partial(read_in_single_experiment, 
                                          database = database,
                                          metadata_retrival_function = metadata_retrival_function,
                                          raw_data_reading_function = raw_data_reading_function,
                                          processing_function = processing_function,
                                          external_version = resolve_external_version(database, external_version))

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(read_in_single_experiment_partial, files))

    for result in results:
        if result['success']:
            database.add_experiment(result['data'])
        else:
            print(f"Failed to process {result['file']}: {result['error']}")

    finalize_processing_run(database, results, external_version)

    return results


# =============================================================================
# Reprocessing of experiments already stored in a dataset
# =============================================================================

def reprocess_single_experiment(experiment: Experiment,
                                processing_function: callable,
                                metadata_retrival_function: Optional[callable] = None,
                                overview_df=None,
                                external_version: Optional[dict] = None) -> dict:
    """
    Rerun the processing step of one experiment that is already in a dataset.

    The raw data stored in the HDF5 file is reused, so no raw-data files are
    needed. The metadata is either taken from the experiment as stored or
    refreshed from the overview DataFrame, which is what makes an edited
    overview sheet take effect.

    The experiment is only mutated once the processing function has returned,
    so a failing experiment keeps its previous processed data.

    Parameters
    ----------
    experiment : Experiment
        Experiment to reprocess; mutated in place on success.
    processing_function : callable
        ``(raw_data_dict, metadata_dict) -> processed_data_dict``.
    metadata_retrival_function : callable, optional
        ``(experiment_name, overview_df) -> metadata_dict``. When given, the
        metadata (and with it ``color`` and ``group``) is refreshed before
        processing; otherwise the stored metadata is reused unchanged.
    overview_df : pandas.DataFrame, optional
        Overview sheet handed to `metadata_retrival_function`.
    external_version : dict, optional
        Provenance of the external app, stamped onto the experiment.

    Returns
    -------
    result : dict
        ``{'success': True, 'data': experiment}`` or
        ``{'success': False, 'file': experiment_name, 'error': message}``,
        matching the result shape of `read_in_single_experiment`.
    """
    try:
        if metadata_retrival_function is not None:
            metadata_dict = metadata_retrival_function(experiment.experiment_name, overview_df)
        else:
            metadata_dict = experiment.metadata

        processed_data_dict = processing_function(experiment.raw_data, metadata_dict)

        experiment.metadata = metadata_dict
        experiment.processed_data = processed_data_dict
        experiment.color = metadata_dict.get('color', experiment.color)
        experiment.group = metadata_dict.get('group', experiment.group)

        stamp_experiment_version(experiment, external_version)

        return {
            'success': True,
            'data': experiment
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f'{experiment.experiment_name} reprocessing failed, previous results kept, error: {str(e)}')
        print("Full traceback:")
        print(tb)

        return {
            'success': False,
            'file': experiment.experiment_name,
            'error': f"{str(e)}\n\nFull traceback:\n{tb}"
        }


def reprocess_experiment_by_name(experiment_name: str,
                                 database: ExperimentalDataset,
                                 processing_function: callable,
                                 metadata_retrival_function: Optional[callable] = None,
                                 external_version: Optional[dict] = None) -> dict:
    """
    Reprocess one experiment of a dataset, addressed by name.

    Name-addressed counterpart of `reprocess_single_experiment`, so that a
    caller stepping through the experiments one at a time can reach both
    pipelines through the same ``(experiment_name, **context)`` call shape as
    `ingest_experiment`.

    Parameters
    ----------
    experiment_name : str
        Experiment to reprocess; must be present in ``database.experiments``.
    database : ExperimentalDataset
        Dataset holding the experiment and the overview sheet.
    processing_function : callable
        ``(raw_data_dict, metadata_dict) -> processed_data_dict``.
    metadata_retrival_function : callable, optional
        When given, metadata is refreshed from ``database.overview_df``.
    external_version : dict, optional
        Provenance of the external app, already resolved.

    Returns
    -------
    result : dict
        Result of `reprocess_single_experiment`.
    """

    return reprocess_single_experiment(
        experiment = database.experiments[experiment_name],
        processing_function = processing_function,
        metadata_retrival_function = metadata_retrival_function,
        overview_df = database.overview_df,
        external_version = external_version
    )


def select_experiments_to_reprocess(database: ExperimentalDataset,
                                    experiment_names: Optional[list] = None) -> list:
    """
    Resolve which experiments of a dataset a reprocessing run covers.

    Parameters
    ----------
    database : ExperimentalDataset
        Dataset holding the experiments.
    experiment_names : list of str, optional
        Requested experiments. Defaults to every experiment in the dataset.

    Returns
    -------
    list of str
        Experiments to reprocess.

    Raises
    ------
    ValueError
        If any requested experiment is not held by the dataset.
    """

    if experiment_names is None:
        return sorted(database.experiments.keys())

    unknown_experiments = [name for name in experiment_names if name not in database.experiments]
    if unknown_experiments:
        raise ValueError(f"Experiments not present in the dataset: {unknown_experiments}")

    return list(experiment_names)


def reprocess_experiments(database: ExperimentalDataset,
                          processing_function: callable,
                          metadata_retrival_function: Optional[callable] = None,
                          experiment_names: Optional[list] = None,
                          external_version: Optional[dict] = None,
                          progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None):
    """
    Rerun the processing step for experiments already held by a dataset.

    This is the path for updating an existing HDF5 file — for example after a
    change to the max-rate algorithm — without going back to the raw-data
    files: metadata and raw data come from the file, only ``processed_data``
    is rebuilt. The dataset's version dictionary records the run.

    Parameters
    ----------
    database : ExperimentalDataset
        Dataset holding the experiments; mutated in place.
    processing_function : callable
        ``(raw_data_dict, metadata_dict) -> processed_data_dict``.
    metadata_retrival_function : callable, optional
        When given, metadata is refreshed from ``database.overview_df`` before
        processing; otherwise the stored metadata is reused.
    experiment_names : list of str, optional
        Experiments to reprocess. Defaults to every experiment in the dataset.
    external_version : dict, optional
        Provenance of the external app. Defaults to the dataset's own
        ``version['external_version']``.
    progress_callback : callable, optional
        Called as ``(completed, total, experiment_name)``, once before the loop
        with ``(0, total, None)`` and again after each experiment — the same
        convention as `read_in_experiments_single_threaded`.

    Returns
    -------
    list of dict
        One result dict per experiment, as returned by
        `reprocess_single_experiment`.
    """
    experiment_names = select_experiments_to_reprocess(database, experiment_names)

    total_experiments = len(experiment_names)
    external_version = resolve_external_version(database, external_version)

    if progress_callback is not None:
        progress_callback(0, total_experiments, None)

    results = []

    for completed, experiment_name in enumerate(experiment_names, start=1):

        results.append(reprocess_experiment_by_name(
            experiment_name = experiment_name,
            database = database,
            processing_function = processing_function,
            metadata_retrival_function = metadata_retrival_function,
            external_version = external_version
        ))

        if progress_callback is not None:
            progress_callback(completed, total_experiments, experiment_name)

    finalize_processing_run(database, results, external_version)

    return results


def testing():
    
    from tests.data.processing_functions_overview_df import (metadata_retrival_function, 
                                                             raw_data_reading_function, 
                                                             processing_function)
    from tests.data.processing_parameters import PROCESSING_PARAMETERS, GROUP_MAPPING, PLOTTING_INSTRUCTIONS

    import pandas as pd
    import pprint as pp
    
    overview_df = pd.read_excel(
        '/Users/jacob/Documents/Water_Splitting/Projects/pyKES/pyKES/src/tests/data/251204_O2_H2_Experiment_Overview.xlsx',
        sheet_name='Sheet1',
        dtype={'active': str,
               'D2O': str,
               'Processed': str}  # Force 'active' and 'D2O' columns to be read as strings
    )

    dataset = ExperimentalDataset(
                    overview_df = overview_df,
                    group_mapping = GROUP_MAPPING,
                    plotting_instruction = PLOTTING_INSTRUCTIONS,
                    processing_parameters = PROCESSING_PARAMETERS
                    )
    
    read_in_experiments_single_threaded(
        database = dataset,
        metadata_retrival_function = metadata_retrival_function,
        raw_data_reading_function = raw_data_reading_function,
        processing_function = processing_function,
        overview_df_experiment_column = 'Experiment',
        directory = Path('/Users/jacob/Documents/Water_Splitting/Projects/pyKES/pyKES/src/tests/data/data_files')
    )

    pp.pprint(dataset.experiments['NB-316'])












if __name__ == '__main__':
    testing()