import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import traceback
from typing import Optional

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
                              processing_function: callable):
    """


    """
    
    try:
        overview_df = database.overview_df

        metadata_dict = metadata_retrival_function(file_name, overview_df)
        raw_data_dict = raw_data_reading_function(file_name, metadata_dict)
        processed_data_dict = processing_function(raw_data_dict, metadata_dict)

        experiment = Experiment(
            experiment_name = metadata_dict['experiment_name'],
            raw_data_file = file_name,
            color = metadata_dict.get('color', 'black'),
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

    print(files)

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