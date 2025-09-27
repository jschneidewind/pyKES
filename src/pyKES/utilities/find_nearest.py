import math
import numpy as np

def find_nearest(array, values):
    """
    Find the indices of the nearest values in the array to a list of target values.

    Parameters:
        array (numpy.ndarray): The input array.
        values (numpy.ndarray or scalar): The target values.

    Returns:
        list: A list of indices of the nearest values in the array to the target values.

    Note:
        If the input array has more than one dimension, the function flattens the first dimension.
        The function uses the `numpy.searchsorted` function to find the indices of the target values in the array.
        The function then checks if the index is not the last index in the array and if the difference between the target value and the previous value in the array is less than the difference between the target value and the current value in the array. If both conditions are true, the index of the previous value is returned, otherwise the index of the current value is returned.
    """

    if array.ndim != 1:
        array_1d = array[:, 0]
    else:
        array_1d = array

    values = np.atleast_1d(values)
    hits = []

    for i in range(len(values)):
        idx = np.searchsorted(array_1d, values[i], side="left")
        if idx > 0 and (
            idx == len(array_1d)
            or math.fabs(values[i] - array_1d[idx - 1])
            < math.fabs(values[i] - array_1d[idx])
        ):
            hits.append(idx - 1)
        else:
            hits.append(idx)

    return hits