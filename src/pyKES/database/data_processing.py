import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import traceback
from typing import Optional
from pathlib import Path

from pyKES.database.database_experiments import ExperimentalDataset, Experiment

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
                              legacy_mode = True):
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


def read_in_experiments_single_threaded(database: ExperimentalDataset,
                                        metadata_retrival_function: callable,
                                        raw_data_reading_function: callable,
                                        processing_function: callable,
                                        overview_df_experiment_column: Optional[str] = 'Experiment',
                                        directory: Optional[Path] = None):
    """
    
    """

    if "Processed" not in database.overview_df.columns:
        database.overview_df["Processed"] = False

    # Returning only the experiments which have not been processed 
    # Do not contain "Processed" column or "Processed" is not True
    # also returns experiments which are not in the database.experiments dict,

    mask = (
    database.overview_df["Processed"].ne('True')
    | ~database.overview_df[overview_df_experiment_column].isin(database.experiments))
    experiments = database.overview_df.loc[mask, 
                    overview_df_experiment_column].astype(str).tolist()
    
    results = []
    
    for experiment_name in experiments:

        result = read_in_single_experiment(
            file_name = experiment_name,
            database = database,
            metadata_retrival_function = metadata_retrival_function,
            raw_data_reading_function = raw_data_reading_function,
            processing_function = processing_function,
            directory = directory,
            legacy_mode = False
        )

        results.append(result)

        if result['success']:
            # Add experiment to database
            database.add_experiment(result['data'])

            # Setting "Processed" to True in dataframe
            database.overview_df.loc[
                database.overview_df[overview_df_experiment_column].eq(result['data'].experiment_name),
                    "Processed",
                ] = 'True'
            
        else:
            print(f"Failed to process {result['file']}: {result['error']}")

    return results


def read_in_experiments_multiprocessing(database: ExperimentalDataset,
                                        metadata_retrival_function: callable,
                                        raw_data_reading_function: callable,
                                        processing_function: callable,
                                        keywords: Optional[list] = None, 
                                        directory: Optional[str] = None,
                                        overview_df_based_processing: Optional[bool] = False,
                                        overview_df_experiment_column: Optional[str] = 'Experiment'): 
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
                                          processing_function = processing_function)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(read_in_single_experiment_partial, files))

    for result in results:
        if result['success']:
            database.add_experiment(result['data'])
        else:
            print(f"Failed to process {result['file']}: {result['error']}")

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