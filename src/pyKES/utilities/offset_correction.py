"""Cut a reaction window out of a measurement and zero it."""

from pyKES.utilities.find_nearest import find_nearest

def offset_correction(time, 
                      data, 
                      offset,
                      start, 
                      end):
    '''
    Cut out the reaction window of a measurement and zero its origin.

    A recorded trace usually starts before the reaction does — the sensor is
    logging while the sample is placed, the lamp is switched on, the baseline
    settles. This selects the stretch that belongs to the reaction and shifts
    both axes so it begins at (0, 0), which is what makes traces from different
    runs comparable.

    Parameters
    ----------
    time : numpy.ndarray
        Time points of the measurement, ascending.
    data : numpy.ndarray
        Measured values, the same length as `time`.
    offset : float
        Shift applied to `start`, in the unit of `time`. The delay between the
        nominal start of the experiment and the actual onset of the reaction.
    start, end : float
        Bounds of the reaction window, in the unit of `time`. The nearest
        available sample is used for each.

    Returns
    -------
    time_reaction : numpy.ndarray
        Times within the window, starting at 0.
    data_reaction : numpy.ndarray
        Values within the window, starting at 0.
    '''

    start = start + offset
    
    idx = find_nearest(time, (start, end))

    time_reaction = time[idx[0]:idx[1]]
    time_reaction = time_reaction - time_reaction[0]

    data_reaction = data[idx[0]:idx[1]]
    data_reaction = data_reaction - data_reaction[0]

    return time_reaction, data_reaction