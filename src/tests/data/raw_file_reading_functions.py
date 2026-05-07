import pandas as pd

def reading_H2_file(file_H2, mode = 'liquid'):
    '''
    Reading data from UniAmp H2 sensor files.
    '''

    raw_data = pd.read_csv(file_H2, sep = ';')
    
    time_s = raw_data['Time since start (s)'].to_numpy()

    try:
        H2_temp = raw_data['Sensor 2 - TEMP-UNIAMP (°C)'].to_numpy()
    except KeyError:
        H2_temp = 0

    raw_data_dict = {
            'H2_time_s': time_s,
            'H2_temperature': H2_temp}
    
    if mode == 'liquid':
        H2_umol_L = raw_data['Sensor 1 - H2 (μmol/L)'].to_numpy()
        raw_data_dict['H2_umol_L'] = H2_umol_L

    elif mode == 'gas':
        H2_Pa = raw_data['Sensor 1 - H2 (Pa)'].to_numpy()
        raw_data_dict['H2_Pa'] = H2_Pa
    
    else:
        raise ValueError("Mode must be either 'liquid' or 'gas'.")

    return raw_data_dict

def reading_O2_file(file_O2,
                    channel):
    '''
	Reading data from FireStingO2 files. 
	channel 1 = gas phase O2 on channel 1
	channel 2 = liquid phase O2 on channel 2
	channel 3 = liquid phase O2 on channel 1
	channel 4 = gas phase O2 on channel 2
	channel 5 = gas phase O2 on channel 1 with updated naming convention
	'''

    channel_mapping = {1: {'O2': 'Oxygen (%O2) [A Ch.1 Main]', 
						   'dt': ' dt (s) [A Ch.1 Main]',
						   'Temp': 'Sample Temp. (°C) [A Ch.1 CompT]'},
					   2: {'O2': 'Oxygen (µmol/L) [A Ch.2 Main]',
					   	   'dt': ' dt (s) [A Ch.2 Main]',
					   	   'Temp': 'Sample Temp. (°C) [A Ch.2 CompT]'},
					   3: {'O2': 'Oxygen (µmol/L) [A Ch.1 Main]', 
						   'dt': ' dt (s) [A Ch.1 Main]',
						   'Temp': 'Sample Temp. (°C) [A Ch.1 CompT]'},
					   4: {'O2': 'Oxygen (%O2) [A Ch.2 Main]',
					   	   'dt': ' dt (s) [A Ch.2 Main]',
					   	   'Temp': 'Sample Temp. (°C) [A Ch.2 CompT]'},
					   5: {'O2': 'Oxygen (%O2) [ Ch.1 Main]',
							'dt': ' dt (s) [ Ch.1 Main]',
							'Temp': 'Optical Temp. (°C) [ Ch.1 CompT]'}}
				
    data = pd.read_csv(
                file_O2, 
				encoding = 'ISO8859', 
                sep = '	', 
				skip_blank_lines = True, 
                comment = '#', 
                parse_dates = [0], 
                dayfirst = True)
	
    data_strings = channel_mapping[channel]

    o2_data = data[data_strings['O2']].to_numpy()
    time = data[data_strings['dt']].to_numpy()

    try:
        temp = data[data_strings['Temp']].to_numpy()
    except KeyError:
        temp = 0

    raw_data_dict = {
        'O2_time_s': time,
        'O2_data': o2_data,
        'O2_temperature': temp
    }

    return raw_data_dict
