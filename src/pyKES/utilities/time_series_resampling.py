"""Resample an unevenly sampled time series onto a regular grid."""

import pandas as pd
from scipy.stats import binned_statistic
import numpy as np

def resample_time_series(time_values, data_values, interval = 5):
    """
    Bin a time series into fixed-width intervals and average each bin.

    Reduces both the sampling rate and the noise of a densely sampled sensor
    trace: averaging within a bin suppresses uncorrelated noise, unlike simply
    taking every n-th point. Bins are of equal width in time, so the result is
    evenly spaced even when the input is not.

    Parameters
    ----------
    time_values : array_like
        Time points, ascending.
    data_values : array_like
        Measured values, the same length as `time_values`.
    interval : float, optional
        Bin width, in the unit of `time_values`.

    Returns
    -------
    new_time : numpy.ndarray
        Bin centers.
    new_data : numpy.ndarray
        Mean of the values in each bin. Empty bins come back as NaN.
    """

    bins = np.arange(time_values[0], time_values[-1] + interval, interval)

    # Average data values in each bin
    new_data, bin_edges, _ = binned_statistic(
        time_values, data_values, statistic='mean', bins=bins)

    # Calculate bin centers for new time values
    new_time = (bin_edges[:-1] + bin_edges[1:]) / 2

    return new_time, new_data

def testing():
    """
    Resample a noisy sine and plot it against the original.

    Returns
    -------
    None
        Shows the comparison plot.
    """

    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 500)
    y = np.sin(x) + 0.1 * np.random.randn(500)

    new_x, new_y = resample_time_series(x, y, interval = 0.1)

    plt.plot(x, y, label='Original Data')
    plt.plot(new_x, new_y, 'o-', label='Resampled Data', linewidth=2)
    plt.show()


    

if __name__ == "__main__":
    testing()   