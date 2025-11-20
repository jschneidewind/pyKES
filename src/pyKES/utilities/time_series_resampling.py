import pandas as pd
from scipy.stats import binned_statistic
import numpy as np

def resample_time_series(time_values, data_values, interval = 5):

    bins = np.arange(time_values[0], time_values[-1] + interval, interval)

    # Average data values in each bin
    new_data, bin_edges, _ = binned_statistic(
        time_values, data_values, statistic='mean', bins=bins)

    # Calculate bin centers for new time values
    new_time = (bin_edges[:-1] + bin_edges[1:]) / 2

    return new_time, new_data

def testing():

    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 500)
    y = np.sin(x) + 0.1 * np.random.randn(500)

    new_x, new_y = resample_time_series(x, y, interval = 0.1)

    plt.plot(x, y, label='Original Data')
    plt.plot(new_x, new_y, 'o-', label='Resampled Data', linewidth=2)
    plt.show()


    

if __name__ == "__main__":
    testing()   