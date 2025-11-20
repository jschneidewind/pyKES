from scipy.interpolate import interp1d
import numpy as np

def harmonize_time_series(time1, data1, time2, data2, method='linear'):
    """
    Harmonize two time series to a common time grid.
    
    Parameters
    ----------
    time1, data1 : np.array
        First time series
    time2, data2 : np.array
        Second time series
    method : str
        Interpolation method ('linear', 'cubic', etc.)
    
    Returns
    -------
    common_time : np.array
        Common time grid
    data1_interp, data2_interp : np.array
        Interpolated data on common grid
    """
    # Find overlapping time range
    t_start = max(time1[0], time2[0])
    t_end = min(time1[-1], time2[-1])
    
    # Create common time grid (use finer timestep)
    dt = min(np.mean(np.diff(time1)), np.mean(np.diff(time2)))
    common_time = np.arange(t_start, t_end, dt)
    
    # Interpolate both datasets
    f1 = interp1d(time1, data1, kind=method, fill_value='extrapolate')
    f2 = interp1d(time2, data2, kind=method, fill_value='extrapolate')
    
    data1_interp = f1(common_time)
    data2_interp = f2(common_time)
    
    return common_time, data1_interp, data2_interp


def testing():
    # write a simple test case with noisy data
    t1 = np.linspace(0, 10, 500)
    d1 = np.sin(t1) + 0.1 * np.random.randn(len(t1))
    t2 = np.linspace(2, 12, 600)
    d2 = np.sin(t2) + 0.1 * np.random.randn(len(t2))
    common_t, d1_interp, d2_interp = harmonize_time_series(t1, d1, t2, d2)
    print("Common time:", common_t)
    print("Data1 interp:", d1_interp)
    print("Data2 interp:", d2_interp)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t1, d1, 'o', label='Data1 original')
    #plt.plot(t2, d2, 'o', label='Data2 original')
    plt.plot(common_t, d1_interp, 'o-', label='Data1 interp')
    #plt.plot(common_t, d2_interp, '-', label='Data2 interp')


    plt.legend()
    plt.show()

if __name__ == "__main__":
    testing()