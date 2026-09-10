from tests.data.raw_file_reading_functions import reading_H2_file, reading_O2_file

from pathlib import Path


def metadata_retrival_function(experiment_name, overview_df):
    '''
    Given an experiment name and an overview DataFrame, retrieves the metadata for the specified experiment.
    Returns a dictionary containing the metadata.


    '''

    experiment_row = overview_df[overview_df['Experiment'] == experiment_name]
    
    if experiment_row.empty:
        raise ValueError(f"No experiment found with name: {experiment_name}")
    
    if len(experiment_row) > 1:
        raise ValueError(f"Multiple experiments found with name: {experiment_name}")
    
    metadata_dict = experiment_row.iloc[0].to_dict()
    metadata_dict['experiment_name'] = metadata_dict['Experiment']

    return metadata_dict

def raw_data_reading_function(directory: Path, metadata_dict):
    '''
    '''

    file_H2 = directory / metadata_dict['File name H2']
    file_O2 = directory / metadata_dict['File name O2']

    if 'Gas phase' in metadata_dict['group']:
        raw_data_H2 = reading_H2_file(file_H2, mode = 'gas')
        raw_data_O2 = reading_O2_file(file_O2, channel = 4)

    else:
        raw_data_H2 = reading_H2_file(file_H2, mode = 'liquid')
        raw_data_O2 = reading_O2_file(file_O2, channel = 2)

    return raw_data_H2 | raw_data_O2

def processing_function(raw_data_dict, metadata_dict):
    '''
    '''

    return {}


