"""Locate the array entries closest to given target values."""

import math
import numpy as np

def find_nearest(array, values):
    """
    Find the indices of the array entries closest to a set of target values.

    Used to translate physical positions into array indices: a time in seconds
    into the index of the sample nearest it, for instance.

    Parameters
    ----------
    array : numpy.ndarray
        Values searched, ascending. A two-dimensional array is reduced to its
        first column.
    values : array_like or scalar
        Target values.

    Returns
    -------
    list of int
        Index of the closest entry of `array` for each target value, in the
        order the targets were given.

    Notes
    -----
    Uses `numpy.searchsorted` to bracket each target, then picks whichever of
    the two neighbours is closer — so the result is the nearest entry, not
    merely the insertion point, and the search stays logarithmic in the length
    of the array.
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