from pyKES.utilities.find_nearest import find_nearest

def offset_correction(time, 
                      data, 
                      offset,
                      start, 
                      end):
    '''
    Apply offset correction to time and data arrays based on specified start and end times.
    '''

    start = start + offset
    
    idx = find_nearest(time, (start, end))

    time_reaction = time[idx[0]:idx[1]]
    time_reaction = time_reaction - time_reaction[0]

    data_reaction = data[idx[0]:idx[1]]
    data_reaction = data_reaction - data_reaction[0]

    return time_reaction, data_reaction