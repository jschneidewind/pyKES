"""Robust extraction of maximum rates from noisy kinetic time series.

Real sensor traces carry two very different kinds of disturbance. Short
artifacts (bubbles, spikes, level shifts) hit a handful of samples and are
far away from everything around them. Low-frequency artifacts (thermal
drift of the optode, stirring beats, slow baseline waves) are as smooth as
the kinetics themselves and cannot be told apart from them sample by
sample. A smoother that fits one length scale to the whole trace has to
compromise between the two, and on noisy traces it settles on a length
scale short enough to track the slow disturbance -- which then dominates
the derivative and inflates the maximum rate.

The pipeline therefore separates the two disturbances by *structure*
rather than by amplitude:

1. Noise characterization: a robust second-difference variogram measures
   how much of the scatter is uncorrelated (white) and how much is
   correlated, and over what time the correlated part decorrelates. Being
   a median statistic on a trend-free quantity, it is not misled by the
   kinetics and not moved by the occasional artifact.
2. Smoothing: the data are modelled as the sum of a slow kinetic component
   (Matern-5/2), a stationary nuisance component pinned to what stage 1
   measured (Matern-3/2) and white noise, fitted by exact O(n) Kalman
   filtering and RTS smoothing. Recurring low-frequency structure then has
   an explanation that costs the kinetic component nothing, so the kinetic
   length scale stays long and the reported rate is the derivative of the
   kinetic component alone.
3. Excursion rejection: every sample is predicted twice, once from the data
   well before it and once from the data well after it. Where the data
   disagree with both predictions while the two agree with each other, the
   trace has left its own curve and come back -- a bubble. Across a genuine
   transition the two sides disagree instead, and nothing is rejected. The
   model is then refitted without the excursions, which matters because a
   single bubble is enough to pull the kinetic length scale down to its own
   width.
4. Robust reweighting: two IRLS passes inflate the observation variance of
   whatever the fit still does not explain. Nothing is deleted outright, so
   a sharp but genuine feature can never be cut out of the series the way
   hard masking cuts it.
5. Rate extraction: the headline number is the largest rate sustained over
   a time window, read from the kinetic component.
6. Cross-checking: a rolling weighted regression of the raw data provides
   a smoothing-free second opinion, and automatic quality flags mark
   curves that need human review.

The modelling assumption behind stages 2 and 3 is that the kinetics are the
slowest structure in the trace: a one-off sharp transition (onset,
light-off) is kinetic and must survive, but structure that *recurs* on a
time scale shorter than the overall rise and decay is instrumental. The
nuisance component is deliberately generic -- a short-correlation-time
Matern process, not an oscillator -- so it absorbs any low-frequency
disturbance, periodic or not.

See docs/max_rate.md for a detailed explanation of every stage.

Inputs and outputs are `Quantity` objects: `time` carries the dimension
time, `values` the dimension substance, and every physical field of the
returned `MaxRateResult` is a `Quantity` as well (the maximum rate, for
instance, has the dimension substance / time). All internal math runs on
plain floats in seconds and moles.

Typical use::

    from pyKES.utilities.unit_handler import Quantity
    from pyKES.utilities.max_rate import extract_max_rate

    result = extract_max_rate(Quantity(time_seconds, 's'), Quantity(amount_umol, 'umol'))
    print(result.max_rate.unit['umol / h'], result.max_rate_std, result.flags)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, minimize

from pyKES.utilities.unit_handler import Quantity

# --- Calculation units --------------------------------------------------------
# Every stage of the pipeline works on plain floats in these units: Quantity
# inputs are reduced to magnitudes on entry and results are re-wrapped on exit.
TIME_UNIT = 's'
AMOUNT_UNIT = 'mol'
RATE_UNIT = 'mol / s'

# --- Statistical conversion factors ----------------------------------------
MAD_TO_STD = 1.4826                  # median absolute deviation -> standard deviation (Gaussian)
SIGNAL_STATE_DIMENSION = 3           # Matern-5/2 state: (value, first derivative, second derivative)
NUISANCE_STATE_DIMENSION = 2         # Matern-3/2 state: (value, first derivative)
STATE_DIMENSION = SIGNAL_STATE_DIMENSION + NUISANCE_STATE_DIMENSION

# --- Noise characterization (robust variogram) -------------------------------
VARIOGRAM_LAG_COUNT = 24             # log-spaced lags probed by the variogram
VARIOGRAM_MAX_LAG_FRACTION = 0.05    # longest lag, as a fraction of the series duration
VARIOGRAM_MIN_SAMPLES = 40           # a lag is only used if this many differences remain
# The variogram model has four parameters, so fewer lags than this cannot split
# the scatter into a white and a correlated part at all -- whatever the fit
# returns is then an arbitrary point on a ridge, not a measurement.
VARIOGRAM_MIN_RESOLVED_LAGS = 6
CORRELATED_NOISE_SIGNIFICANCE = 0.1  # correlated noise below this variance ratio counts as absent
NUISANCE_MIN_LENGTHSCALE_STEPS = 3.0 # floor on the nuisance correlation time, in median time steps
NUISANCE_ABSENT_FRACTION = 1e-3      # residual nuisance amplitude, as a fraction of the white noise
# A nuisance allowed to decorrelate slowly stops being distinguishable from
# kinetics and starts absorbing the curvature of the reaction curve itself.
# Measured instrument disturbances sit an order of magnitude below this cap.
NUISANCE_LENGTHSCALE_MAX_FRACTION = 0.02

# --- Gaussian-process model --------------------------------------------------
KINETIC_SEPARATION_FACTOR = 2.0      # kinetic length scale floor, in nuisance correlation times
NUISANCE_AMPLITUDE_FRACTION = 0.5    # nuisance std may not exceed this fraction of the trace's robust spread
LENGTHSCALE_MIN_STEPS = 20           # length scale floor in median time steps (guards against noise fitting)
LENGTHSCALE_MIN_SPAN_FRACTION = 0.002
LENGTHSCALE_MAX_SPAN_FRACTION = 0.5

# --- Robust reweighting ------------------------------------------------------
ROBUST_PASSES = 2                    # IRLS refits after the first smoothing pass
ROBUST_WEIGHT_FLOOR = 1e-4           # smallest weight, i.e. largest variance inflation
GROSS_OUTLIER_THRESHOLD = 8.0        # pre-fit rejection of spikes, in robust stds of a median-filter residual
GROSS_OUTLIER_WINDOW_STEPS = 11      # width of that median filter, in samples
OUTLIER_WEIGHT_THRESHOLD = 0.5       # weights below this are reported in `outlier_mask`

# --- Excursion rejection (two-sided predictive test) -------------------------
ARTIFACT_GAP_FRACTION = 0.02         # blind gap of the two-sided predictive test, as a fraction of the duration
ARTIFACT_GAP_MIN_STEPS = 15          # ... and at least this many time steps
ARTIFACT_GAP_NUISANCE_FACTOR = 3.0   # ... and this many nuisance correlation times
ARTIFACT_GAP_MAX_FRACTION = 0.2      # the gap may never exceed this fraction of the samples
ARTIFACT_STIFFNESS_FACTOR = 8.0      # the reference curve must be this much smoother than the gap is wide
ARTIFACT_PASSES = 2                  # repeats of the two-sided test; each one cleans the filter it uses
ARTIFACT_GROWTH_CAP_FACTOR = 5.0     # an excursion may be grown this many gaps before it counts as permanent
ARTIFACT_RETURN_TOLERANCE = 1.0      # growth stops once the deviation is back within this many stds
ARTIFACT_SETTLED_SAMPLES = 5         # ... for this many consecutive samples

# --- Rate windows ------------------------------------------------------------
WINDOW_MEDIAN_STEPS = 25             # default window: at least this many median time steps ...
WINDOW_SPAN_FRACTION = 0.02          # ... and at least this fraction of the series duration ...
WINDOW_MAX_SPAN_FRACTION = 0.1       # ... but never more than this fraction of it
MIN_WINDOW_POINTS = 10               # rolling regression needs at least this many points per window
MIN_WINDOW_WEIGHT_FRACTION = 0.5     # windows with less effective weight than this are not trusted

# --- Quality-flag thresholds -------------------------------------------------
HIGH_UNCERTAINTY_FRACTION = 0.5      # flag if the uncertainty exceeds this fraction of the max rate
OUTLIER_FRACTION_WARNING = 0.05      # flag if more than this fraction of samples was downweighted
DISAGREEMENT_RELATIVE = 0.2          # estimators must differ by more than this fraction ...
DISAGREEMENT_SIGMA = 3.0             # ... and this many combined standard deviations to flag
INSTANTANEOUS_SPIKE_FACTOR = 3.0     # flag if the instantaneous max exceeds this multiple of the windowed max
CORRELATED_NOISE_RATE_FRACTION = 1.0 # flag if the nuisance slope scale reaches this fraction of the max rate
SIGNIFICANCE_SIGMA = 3.0             # flag if the max rate is not this many stds above zero
RESIDUAL_AUTOCORRELATION_WARNING = 0.9
LENGTHSCALE_BOUND_FACTOR = 1.05      # flag if the length scale sits within this factor of its lower bound


def quantity_magnitude(quantity, unit, name):
    """
    Read the numeric magnitude of a `Quantity` in one of the calculation units.

    Parameters
    ----------
    quantity : Quantity
        Value to reduce to a magnitude.
    unit : str
        Target unit; a dimension mismatch raises inside the unit handler.
    name : str
        Argument name, used only in the error message.

    Returns
    -------
    float or ndarray
        The magnitude expressed in `unit`.
    """
    if not isinstance(quantity, Quantity):
        raise TypeError(f"{name} must be a Quantity compatible with '{unit}', "
                        f'got {type(quantity).__name__}')

    return quantity.unit[unit]


def robust_scale(values):
    """
    Median-absolute-deviation estimate of a standard deviation.

    Parameters
    ----------
    values : ndarray
        Sample of a roughly zero-centred quantity.

    Returns
    -------
    float
        Robust standard deviation; zero for a constant input.
    """
    return float(MAD_TO_STD * np.median(np.abs(values - np.median(values))))


def resolve_window(window, median_time_step, duration):
    """
    Resolve the sustained-rate window to a length in seconds.

    The default is a fraction of the duration, held above a minimum number of
    samples and below a maximum fraction of the run. The sample floor and the
    duration cap cross at exactly
    ``WINDOW_MEDIAN_STEPS / WINDOW_MAX_SPAN_FRACTION`` samples, so the rule is
    continuous: on longer series the cap never binds and the window is
    sampling-limited as before, while on shorter ones it is the duration that
    sets it. Without the cap a coarsely sampled short run gets a window a third
    of the experiment wide, which averages the maximum together with whatever
    follows it.

    Parameters
    ----------
    window : Quantity or None
        User-supplied window (dimension time); the default is used when None.
    median_time_step : float
        Median time step of the series, in seconds.
    duration : float
        Total duration of the series, in seconds.

    Returns
    -------
    float
        Window length in seconds.
    """
    if window is None:
        return min(max(WINDOW_MEDIAN_STEPS * median_time_step,
                       WINDOW_SPAN_FRACTION * duration),
                   WINDOW_MAX_SPAN_FRACTION * duration)

    return float(quantity_magnitude(window, TIME_UNIT, 'window'))


def resolve_lengthscale_bounds(lengthscale_bounds, median_time_step, duration,
                               nuisance_lengthscale):
    """
    Resolve the bounds of the kinetic Gaussian-process length scale, in seconds.

    The lower bound is what keeps the kinetic component from tracking
    correlated noise: it is raised to a multiple of the measured nuisance
    correlation time whenever that is the stronger constraint. On traces
    with white noise only, the nuisance correlation time collapses to about
    one time step and the bound reduces to the sampling-based default.

    Parameters
    ----------
    lengthscale_bounds : tuple of Quantity or None
        (lower, upper) bounds with dimension time. When given, they are used
        verbatim and the nuisance-derived floor is not applied.
    median_time_step : float
        Median time step of the series, in seconds.
    duration : float
        Total duration of the series, in seconds.
    nuisance_lengthscale : float
        Correlation time of the correlated noise, in seconds.

    Returns
    -------
    tuple of float
        (lower, upper) bounds in seconds.
    """
    if lengthscale_bounds is not None:
        lower, upper = lengthscale_bounds
        return (float(quantity_magnitude(lower, TIME_UNIT, 'lengthscale_bounds')),
                float(quantity_magnitude(upper, TIME_UNIT, 'lengthscale_bounds')))

    lower = max(LENGTHSCALE_MIN_STEPS * median_time_step,
                LENGTHSCALE_MIN_SPAN_FRACTION * duration,
                KINETIC_SEPARATION_FACTOR * nuisance_lengthscale)
    upper = LENGTHSCALE_MAX_SPAN_FRACTION * duration

    return (min(lower, 0.5 * upper), upper)


def hyperparameter_magnitudes(hyperparameters):
    """
    Reduce a reused hyperparameter dict of Quantities to plain floats.

    Accepts what `MaxRateResult.hyperparameters` holds, so a fit can be
    handed straight back to `extract_max_rate`.

    Parameters
    ----------
    hyperparameters : dict of Quantity
        ``lengthscale`` and ``nuisance_lengthscale`` (time), ``signal_std``,
        ``nuisance_std`` and ``noise_std`` (substance).

    Returns
    -------
    dict of float
        The same entries in seconds and moles.
    """
    units = {'lengthscale': TIME_UNIT, 'nuisance_lengthscale': TIME_UNIT,
             'signal_std': AMOUNT_UNIT, 'nuisance_std': AMOUNT_UNIT, 'noise_std': AMOUNT_UNIT}

    return {name: float(quantity_magnitude(hyperparameters[name], unit, name))
            for name, unit in units.items()}


def validate_time_series(time, values):
    """
    Convert Quantity inputs to clean, time-sorted arrays in the calculation units.

    Non-finite entries and duplicate time stamps are dropped.

    Parameters
    ----------
    time : Quantity
        Sample times (dimension time).
    values : Quantity
        Measured amounts (dimension substance).

    Returns
    -------
    tuple of ndarray
        The cleaned (time, values) arrays, in seconds and moles.
    """
    time = np.asarray(quantity_magnitude(time, TIME_UNIT, 'time'), dtype=float).ravel()
    values = np.asarray(quantity_magnitude(values, AMOUNT_UNIT, 'values'), dtype=float).ravel()

    if time.shape != values.shape:
        raise ValueError(f'time and values differ in length: {time.shape} vs {values.shape}')

    finite = np.isfinite(time) & np.isfinite(values)
    time, values = time[finite], values[finite]

    order = np.argsort(time, kind='stable')
    time, values = time[order], values[order]

    strictly_increasing = np.concatenate([[True], np.diff(time) > 0])
    time, values = time[strictly_increasing], values[strictly_increasing]

    if len(time) < 20:
        raise ValueError(f'need at least 20 valid points, got {len(time)}')
    if np.ptp(values) == 0:
        raise ValueError('values are constant; no rate can be extracted')

    return time, values


def matern32_correlation(lag, lengthscale):
    """
    Correlation function of a Matern-3/2 process.

    Parameters
    ----------
    lag : ndarray
        Time separations.
    lengthscale : float
        Correlation length scale.

    Returns
    -------
    ndarray
        Correlation at each lag.
    """
    scaled_lag = np.sqrt(3.0) * np.abs(lag) / lengthscale

    return (1.0 + scaled_lag) * np.exp(-scaled_lag)


def second_difference_variogram(time, values):
    """
    Measure the scatter of second differences over a range of lags.

    The second difference ``y(t + h) - 2 y(t) + y(t - h)`` annihilates any
    linear trend, so at short lags it sees only noise; a robust (median
    based) scale keeps spikes and one-off steps from dominating. The way
    that scale grows with the lag is the fingerprint that separates white
    from correlated noise: white noise gives a flat variogram, correlated
    noise a curve that rises until the lag exceeds its correlation time.

    Parameters
    ----------
    time, values : ndarray
        Sample times and values.

    Returns
    -------
    tuple of ndarray
        Lags in time units and the robust variance of the second
        differences at each lag.
    """
    number_of_samples = len(values)
    median_time_step = float(np.median(np.diff(time)))

    max_lag = min(int(VARIOGRAM_MAX_LAG_FRACTION * (time[-1] - time[0]) / median_time_step),
                  (number_of_samples - VARIOGRAM_MIN_SAMPLES) // 2)
    if max_lag < 2:
        return np.array([median_time_step]), np.array([robust_scale(np.diff(values)) ** 2 * 3.0])

    lag_steps = np.unique(np.geomspace(1, max_lag, VARIOGRAM_LAG_COUNT).astype(int))

    variances = np.array([robust_scale(values[2 * lag:] - 2.0 * values[lag:-lag]
                                       + values[:-2 * lag]) ** 2
                          for lag in lag_steps])

    return lag_steps * median_time_step, variances


def variogram_residuals(log_parameters, lags, variances):
    """
    Log-space residuals of the noise model against a measured variogram.

    The model is white noise plus a Matern-3/2 correlated component plus the
    curvature of the underlying kinetics, which enters the second difference
    as ``f'' h**2`` and therefore contributes a term growing with the fourth
    power of the lag.

    Parameters
    ----------
    log_parameters : ndarray
        Logarithms of (white variance, correlated variance, correlation time,
        curvature coefficient).
    lags : ndarray
        Lags of the measured variogram.
    variances : ndarray
        Measured second-difference variances.

    Returns
    -------
    ndarray
        Differences of logarithms, one per lag.
    """
    white_variance, correlated_variance, lengthscale, curvature = np.exp(log_parameters)

    model = (6.0 * white_variance
             + correlated_variance * (6.0 - 8.0 * matern32_correlation(lags, lengthscale)
                                      + 2.0 * matern32_correlation(2.0 * lags, lengthscale))
             + curvature * lags ** 4)

    return np.log(model) - np.log(variances)


def variogram_residuals_at_fixed_lengthscale(log_parameters, lags, variances, lengthscale):
    """
    Log-space residuals of the noise model with the correlation time pinned.

    Used for the refit that follows a clamp of the correlation time to its
    resolution floor: the two variances and the curvature term are free to
    re-share the scatter under the constraint, which they must, because a
    process held to a longer correlation time explains less of the short-lag
    variogram than the unconstrained fit assigned to it.

    Parameters
    ----------
    log_parameters : ndarray
        Logarithms of (white variance, correlated variance, curvature
        coefficient); the correlation time is not among them.
    lags : ndarray
        Lags of the measured variogram.
    variances : ndarray
        Measured second-difference variances.
    lengthscale : float
        Correlation time to hold fixed.

    Returns
    -------
    ndarray
        Differences of logarithms, one per lag.
    """
    log_white, log_correlated, log_curvature = log_parameters

    return variogram_residuals(
        np.array([log_white, log_correlated, np.log(lengthscale), log_curvature]), lags, variances)


def estimate_noise_structure(time, values):
    """
    Split the scatter of a trace into a white and a correlated component.

    Fits the variogram model of `variogram_residuals` to the measured
    second-difference variogram. Two things can stop the correlated component
    from being usable: a variogram with too few lags to separate two
    components in the first place, and a component carrying too little
    variance to matter. Either way it is folded into the white noise, which
    switches the nuisance component of the Gaussian process off in all but
    name. A correlation time shorter than the sampling can resolve is neither:
    it is a constraint the fit has to respect, not a reason to discard what the
    variogram plainly shows.

    Parameters
    ----------
    time, values : ndarray
        Sample times and values.

    Returns
    -------
    dict
        ``white_std``, ``correlated_std`` and ``correlated_lengthscale``.
    """
    median_time_step = float(np.median(np.diff(time)))
    lags, variances = second_difference_variogram(time, values)

    white_variance_initial = max(variances[0] / 6.0, np.finfo(float).tiny)
    correlated_variance_initial = max(variances.max() / 6.0 - white_variance_initial,
                                      0.01 * white_variance_initial)
    lengthscale_initial = float(np.sqrt(lags[0] * lags[-1]))
    curvature_initial = variances[-1] / lags[-1] ** 4

    lengthscale_upper = max(NUISANCE_LENGTHSCALE_MAX_FRACTION * (time[-1] - time[0]),
                            2.0 * median_time_step)
    bounds = (np.log([1e-6 * white_variance_initial, 1e-6 * correlated_variance_initial,
                      median_time_step, 1e-12 * curvature_initial]),
              np.log([1e3 * white_variance_initial, 1e4 * correlated_variance_initial,
                      lengthscale_upper, 1e6 * curvature_initial]))
    initial_guess = np.clip(np.log([white_variance_initial, correlated_variance_initial,
                                    lengthscale_initial, curvature_initial]), *bounds)

    fit = least_squares(variogram_residuals, initial_guess, args=(lags, variances), bounds=bounds)
    white_variance, correlated_variance, lengthscale, _ = np.exp(fit.x)

    # A nuisance one or two samples wide would interpolate the measurement noise
    # point by point, leaving residuals -- and with them the scale the robust
    # reweighting calibrates against -- near zero. The floor that prevents it is
    # a constraint on the fit, not a verdict on the component: pinning the
    # correlation time there and letting the two variances re-share the scatter
    # keeps the split continuous, and the refit is what makes the clamp safe,
    # since a process held to a longer correlation time has to hand the white
    # noise back the short-lag variance it can no longer explain. Discarding the
    # component instead put a well whose correlation time missed the floor by
    # 0.4 % (AE-855_B2) 17 % out, because its correlated noise -- 65 times the
    # white variance -- then had nowhere to go but the kinetic component.
    lengthscale_floor = min(NUISANCE_MIN_LENGTHSCALE_STEPS * median_time_step, lengthscale_upper)
    resolvable = len(lags) >= VARIOGRAM_MIN_RESOLVED_LAGS

    if resolvable and lengthscale < lengthscale_floor:
        free_parameters = fit.x[[0, 1, 3]]
        free_bounds = (bounds[0][[0, 1, 3]], bounds[1][[0, 1, 3]])
        refit = least_squares(variogram_residuals_at_fixed_lengthscale, free_parameters,
                              args=(lags, variances, lengthscale_floor), bounds=free_bounds)
        white_variance, correlated_variance = np.exp(refit.x[:2])
        lengthscale = lengthscale_floor

    # What is left unusable is a variogram too short to separate two components
    # -- a four-parameter fit to a handful of lags lands anywhere on a ridge,
    # and short, coarsely sampled series arrive here routinely -- or a component
    # carrying too little variance to be worth a state of its own.
    unresolved = (not resolvable
                  or correlated_variance < CORRELATED_NOISE_SIGNIFICANCE * white_variance)
    if unresolved:
        white_variance += correlated_variance
        # Not zero: a nuisance block with no prior variance is singular and
        # the smoother cannot solve through it.
        correlated_variance = NUISANCE_ABSENT_FRACTION ** 2 * white_variance
        lengthscale = median_time_step

    return {'white_std': float(np.sqrt(white_variance)),
            'correlated_std': float(np.sqrt(correlated_variance)),
            'correlated_lengthscale': float(lengthscale)}


def gross_outlier_weights(values, noise_std):
    """
    Pre-fit weights that reject only spikes far away from their neighbours.

    A short median filter follows any kinetics and any low-frequency
    disturbance, so what remains in its residual is spikes. The threshold is
    deliberately loose: this stage only keeps gross outliers from distorting
    the hyperparameter fit, and the IRLS passes afterwards do the actual
    downweighting from the fitted model.

    Parameters
    ----------
    values : ndarray
        Measured values.
    noise_std : float
        White-noise standard deviation from `estimate_noise_structure`.

    Returns
    -------
    ndarray
        Weight per sample, either one or `ROBUST_WEIGHT_FLOOR`.
    """
    window = min(GROSS_OUTLIER_WINDOW_STEPS, len(values) // 4 | 1)
    residuals = values - median_filter(values, size=window, mode='nearest')

    scale = max(robust_scale(residuals), noise_std, np.finfo(float).tiny)
    weights = np.ones(len(values))
    weights[np.abs(residuals) > GROSS_OUTLIER_THRESHOLD * scale] = ROBUST_WEIGHT_FLOOR

    return weights


def matern52_state_space_matrices(time_steps, lengthscale, signal_variance):
    """
    Build the discrete-time state-space representation of a Matern-5/2 Gaussian process.

    A Matern-5/2 process is exactly equivalent to a three-dimensional linear
    stochastic differential equation whose state holds the function value and
    its first two derivatives. This function discretizes that equation for
    the given time steps, which is what lets the Kalman filter perform exact
    Gaussian process regression in O(n).

    Parameters
    ----------
    time_steps : ndarray
        Time differences between consecutive samples.
    lengthscale : float
        Kernel length scale (how quickly the underlying curve can change).
    signal_variance : float
        Kernel variance (how far the curve can wander from its mean).

    Returns
    -------
    tuple
        - transitions : ndarray, shape (len(time_steps), 3, 3), state
          transition matrices; ``transitions[k]`` propagates sample k to k+1.
        - process_noises : ndarray, shape (len(time_steps), 3, 3), process
          noise covariance matrices.
        - stationary_covariance : ndarray, shape (3, 3), covariance of the
          state at equilibrium (used as the initial covariance).
    """
    decay_rate = np.sqrt(5.0) / lengthscale

    feedback = np.array([[0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [-decay_rate ** 3, -3.0 * decay_rate ** 2, -3.0 * decay_rate]])

    derivative_variance = signal_variance * decay_rate ** 2 / 3.0
    stationary_covariance = np.array([[signal_variance, 0.0, -derivative_variance],
                                      [0.0, derivative_variance, 0.0],
                                      [-derivative_variance, 0.0, signal_variance * decay_rate ** 4]])

    # The feedback matrix has the triple eigenvalue -decay_rate, so
    # (feedback + decay_rate * I) is nilpotent and the matrix exponential
    # truncates exactly after the quadratic term.
    nilpotent_part = feedback + decay_rate * np.eye(SIGNAL_STATE_DIMENSION)
    nilpotent_steps = nilpotent_part[None, :, :] * time_steps[:, None, None]
    transitions = np.exp(-decay_rate * time_steps)[:, None, None] * (
        np.eye(SIGNAL_STATE_DIMENSION)[None] + nilpotent_steps
        + 0.5 * (nilpotent_steps @ nilpotent_steps))

    process_noises = stationary_covariance[None] - \
        transitions @ stationary_covariance[None] @ np.transpose(transitions, (0, 2, 1))

    return transitions, process_noises, stationary_covariance


def matern32_state_space_matrices(time_steps, lengthscale, signal_variance):
    """
    Build the discrete-time state-space representation of a Matern-3/2 Gaussian process.

    The two-dimensional analogue of `matern52_state_space_matrices`; its
    state holds the function value and its first derivative. Matern-3/2 is
    used for the nuisance component because it is rougher than Matern-5/2
    and therefore a better catch-all for whatever low-frequency disturbance
    the instrument adds.

    Parameters
    ----------
    time_steps : ndarray
        Time differences between consecutive samples.
    lengthscale : float
        Kernel length scale.
    signal_variance : float
        Kernel variance.

    Returns
    -------
    tuple
        Transition matrices, process noise covariances and the stationary
        covariance, shaped as in `matern52_state_space_matrices`.
    """
    decay_rate = np.sqrt(3.0) / lengthscale

    stationary_covariance = np.array([[signal_variance, 0.0],
                                      [0.0, signal_variance * decay_rate ** 2]])

    # (feedback + decay_rate * I) squares to zero here, so the exponential
    # truncates after the linear term.
    nilpotent_part = np.array([[decay_rate, 1.0], [-decay_rate ** 2, -decay_rate]])
    transitions = np.exp(-decay_rate * time_steps)[:, None, None] * (
        np.eye(NUISANCE_STATE_DIMENSION)[None] + nilpotent_part[None] * time_steps[:, None, None])

    process_noises = stationary_covariance[None] - \
        transitions @ stationary_covariance[None] @ np.transpose(transitions, (0, 2, 1))

    return transitions, process_noises, stationary_covariance


def combined_state_space_matrices(time_steps, hyperparameters):
    """
    Stack the kinetic and nuisance state spaces into one block-diagonal model.

    The two components are independent a priori, so their transition and
    process-noise matrices simply sit in separate blocks; only the
    observation couples them, because the sensor sees their sum.

    Parameters
    ----------
    time_steps : ndarray
        Time differences between consecutive samples.
    hyperparameters : dict of float
        ``lengthscale``, ``signal_std``, ``nuisance_lengthscale`` and
        ``nuisance_std``.

    Returns
    -------
    tuple
        Transition matrices, process noise covariances and the stationary
        covariance of the combined 5-dimensional state.
    """
    signal_transitions, signal_noises, signal_stationary = matern52_state_space_matrices(
        time_steps, hyperparameters['lengthscale'], hyperparameters['signal_std'] ** 2)
    nuisance_transitions, nuisance_noises, nuisance_stationary = matern32_state_space_matrices(
        time_steps, hyperparameters['nuisance_lengthscale'], hyperparameters['nuisance_std'] ** 2)

    number_of_steps = len(time_steps)
    transitions = np.zeros((number_of_steps, STATE_DIMENSION, STATE_DIMENSION))
    process_noises = np.zeros_like(transitions)
    stationary_covariance = np.zeros((STATE_DIMENSION, STATE_DIMENSION))

    signal_block = slice(0, SIGNAL_STATE_DIMENSION)
    nuisance_block = slice(SIGNAL_STATE_DIMENSION, STATE_DIMENSION)

    transitions[:, signal_block, signal_block] = signal_transitions
    transitions[:, nuisance_block, nuisance_block] = nuisance_transitions
    process_noises[:, signal_block, signal_block] = signal_noises
    process_noises[:, nuisance_block, nuisance_block] = nuisance_noises
    stationary_covariance[signal_block, signal_block] = signal_stationary
    stationary_covariance[nuisance_block, nuisance_block] = nuisance_stationary

    return transitions, process_noises, stationary_covariance


# The sensor observes the sum of the kinetic and the nuisance value, i.e. the
# two value components of the combined state.
KINETIC_VALUE_INDEX = 0
KINETIC_RATE_INDEX = 1
NUISANCE_VALUE_INDEX = SIGNAL_STATE_DIMENSION


def run_kalman_filter(values_centered, transitions, process_noises, stationary_covariance,
                      noise_variances):
    """
    Run the forward Kalman filter over the combined state.

    Every sample is used; suspect samples enter with an inflated observation
    variance rather than being skipped, which keeps the filter well
    conditioned and avoids the unsupported bridges that hard masking leaves
    behind.

    Parameters
    ----------
    values_centered : ndarray
        Mean-subtracted observations.
    transitions, process_noises, stationary_covariance : ndarray
        Output of `combined_state_space_matrices`.
    noise_variances : ndarray
        Observation noise variance per sample.

    Returns
    -------
    tuple
        - log_likelihood : float, marginal log-likelihood of the observations.
        - filtered_means, filtered_covariances : ndarray, state estimates after each update.
        - predicted_means, predicted_covariances : ndarray, one-step-ahead state estimates.
    """
    number_of_samples = len(values_centered)
    filtered_means = np.empty((number_of_samples, STATE_DIMENSION))
    filtered_covariances = np.empty((number_of_samples, STATE_DIMENSION, STATE_DIMENSION))
    predicted_means = np.empty_like(filtered_means)
    predicted_covariances = np.empty_like(filtered_covariances)

    state_mean = np.zeros(STATE_DIMENSION)
    state_covariance = stationary_covariance.copy()
    log_likelihood = 0.0

    for sample in range(number_of_samples):
        # Propagate the state to the current sample time
        if sample > 0:
            state_mean = transitions[sample - 1] @ state_mean
            state_covariance = transitions[sample - 1] @ state_covariance @ transitions[sample - 1].T \
                + process_noises[sample - 1]
            state_covariance = 0.5 * (state_covariance + state_covariance.T)  # keep symmetric

        predicted_means[sample] = state_mean
        predicted_covariances[sample] = state_covariance

        # Update with the observation (the sum of the two value components).
        # The observation vector is a sum of two unit vectors, so the usual
        # matrix-vector products collapse to picking and adding two columns.
        observation_covariance = state_covariance[:, KINETIC_VALUE_INDEX] \
            + state_covariance[:, NUISANCE_VALUE_INDEX]
        innovation_variance = observation_covariance[KINETIC_VALUE_INDEX] \
            + observation_covariance[NUISANCE_VALUE_INDEX] + noise_variances[sample]
        innovation = values_centered[sample] - state_mean[KINETIC_VALUE_INDEX] \
            - state_mean[NUISANCE_VALUE_INDEX]
        gain = observation_covariance / innovation_variance

        state_mean = state_mean + gain * innovation
        state_covariance = state_covariance - gain[:, None] * observation_covariance[None, :]

        filtered_means[sample] = state_mean
        filtered_covariances[sample] = state_covariance
        log_likelihood += -0.5 * (np.log(2.0 * np.pi * innovation_variance)
                                  + innovation ** 2 / innovation_variance)

    return log_likelihood, filtered_means, filtered_covariances, predicted_means, predicted_covariances


def run_rts_smoother(transitions, filtered_means, filtered_covariances,
                     predicted_means, predicted_covariances):
    """
    Run the backward Rauch-Tung-Striebel pass.

    The smoother refines every filtered state estimate with the information
    from all later samples, turning the causal filter output into the full
    Gaussian process posterior.

    Parameters
    ----------
    transitions : ndarray
        State transition matrices (``transitions[k]``: sample k to k+1).
    filtered_means, filtered_covariances, predicted_means, predicted_covariances : ndarray
        Output of `run_kalman_filter`.

    Returns
    -------
    tuple of ndarray
        Smoothed state means and covariances.
    """
    smoothed_means = filtered_means.copy()
    smoothed_covariances = filtered_covariances.copy()

    for sample in range(len(filtered_means) - 2, -1, -1):
        gain = np.linalg.solve(predicted_covariances[sample + 1].T,
                               (filtered_covariances[sample] @ transitions[sample].T).T).T

        smoothed_means[sample] = filtered_means[sample] \
            + gain @ (smoothed_means[sample + 1] - predicted_means[sample + 1])
        smoothed_covariances[sample] = filtered_covariances[sample] \
            + gain @ (smoothed_covariances[sample + 1] - predicted_covariances[sample + 1]) @ gain.T

    return smoothed_means, smoothed_covariances


def unpack_hyperparameters(log_parameters, nuisance):
    """
    Turn the optimizer's parameter vector into a hyperparameter dict.

    Only the kinetic component and the white noise are optimized. The
    nuisance component is pinned to what the variogram measured, because the
    likelihood on its own cannot choose between the two explanations of a
    slow wiggle -- kinetics that bend or noise that drifts -- and lands in
    whichever optimum the optimizer happens to approach first. The variogram
    settles the question outside the likelihood, from robust statistics that
    a trend cannot bias.

    Parameters
    ----------
    log_parameters : ndarray
        Logarithms of (kinetic length scale, kinetic std, white noise std).
    nuisance : tuple of float
        Fixed (correlation time, standard deviation) of the nuisance component.

    Returns
    -------
    dict of float
        Hyperparameters ready for `combined_state_space_matrices`.
    """
    lengthscale, signal_std, noise_std = np.exp(log_parameters)
    nuisance_lengthscale, nuisance_std = nuisance

    return {'lengthscale': float(lengthscale), 'signal_std': float(signal_std),
            'nuisance_lengthscale': float(nuisance_lengthscale),
            'nuisance_std': float(nuisance_std),
            'noise_std': float(noise_std)}


def negative_log_likelihood(log_parameters, time_steps, values_centered, weights, nuisance):
    """
    Negative marginal log-likelihood of the two-component model, for the optimizer.

    Parameters
    ----------
    log_parameters : ndarray
        Parameter vector as described in `unpack_hyperparameters`.
    time_steps : ndarray
        Time differences between consecutive samples.
    values_centered : ndarray
        Mean-subtracted observations.
    weights : ndarray
        Robustness weights; the observation variance of sample i is
        ``noise_std ** 2 / weights[i]``.
    nuisance : tuple of float
        Fixed (correlation time, standard deviation) of the nuisance component.

    Returns
    -------
    float
        The negative log-likelihood.
    """
    hyperparameters = unpack_hyperparameters(log_parameters, nuisance)

    transitions, process_noises, stationary_covariance = \
        combined_state_space_matrices(time_steps, hyperparameters)
    log_likelihood, *_ = run_kalman_filter(values_centered, transitions, process_noises,
                                           stationary_covariance,
                                           hyperparameters['noise_std'] ** 2 / weights)

    return -log_likelihood


def fit_hyperparameters(time, values_centered, weights, lengthscale_bounds,
                        noise_structure, max_fit_points, previous_fit=None):
    """
    Fit the Gaussian-process hyperparameters by maximizing the marginal likelihood.

    Fitting runs on a decimated subset of the samples; the Kalman likelihood
    is exact for any sampling pattern, so decimation only trades statistical
    for computational efficiency. Two starting length scales are tried to
    avoid local optima. The nuisance correlation time is not fitted here --
    it is measured directly from the variogram, which keeps it identifiable
    even on traces where the correlated component is weak.

    Parameters
    ----------
    time : ndarray
        Sample times.
    values_centered : ndarray
        Mean-subtracted values.
    weights : ndarray
        Robustness weights.
    lengthscale_bounds : tuple of float
        (lower, upper) bounds for the kinetic length scale.
    noise_structure : dict
        Output of `estimate_noise_structure`. It supplies the initial guesses
        and fixes the nuisance component.
    max_fit_points : int
        Decimation target for the optimization.
    previous_fit : dict of float, optional
        Hyperparameters of an earlier fit of the same series. When given the
        optimizer restarts from them alone, which is what makes the refit
        after artifact detection cost a fraction of the first fit.

    Returns
    -------
    dict of float
        Fitted ``lengthscale``, ``signal_std``, ``nuisance_lengthscale``,
        ``nuisance_std`` and ``noise_std``.
    """
    stride = max(1, int(np.ceil(len(time) / max_fit_points)))
    fit_times = time[::stride]
    fit_values = values_centered[::stride]
    fit_weights = weights[::stride]
    fit_time_steps = np.diff(fit_times)

    signal_std_initial = max(np.std(fit_values), np.finfo(float).tiny)
    noise_std_initial = max(noise_structure['white_std'], 1e-9 * signal_std_initial)
    duration = fit_times[-1] - fit_times[0]

    # The nuisance may never grow into a rival explanation of the reaction
    # itself, however the variogram was misled.
    nuisance = (noise_structure['correlated_lengthscale'],
                min(noise_structure['correlated_std'],
                    NUISANCE_AMPLITUDE_FRACTION * robust_scale(fit_values)))

    bounds = [np.log(lengthscale_bounds),
              np.log((1e-3 * signal_std_initial, 1e3 * signal_std_initial)),
              np.log((1e-2 * noise_std_initial, 1e3 * noise_std_initial))]

    if previous_fit is None:
        starting_lengthscales = (duration / 100.0, duration / 10.0)
    else:
        starting_lengthscales = (previous_fit['lengthscale'],)

    best_fit = None
    for lengthscale_initial in starting_lengthscales:
        initial_guess = np.log([np.clip(lengthscale_initial, *lengthscale_bounds),
                                signal_std_initial, noise_std_initial])

        fit = minimize(negative_log_likelihood, np.clip(initial_guess, *np.transpose(bounds)),
                       args=(fit_time_steps, fit_values, fit_weights, nuisance),
                       method='Nelder-Mead', bounds=bounds,
                       options={'xatol': 0.02, 'fatol': 0.1, 'maxiter': 400})

        if best_fit is None or fit.fun < best_fit.fun:
            best_fit = fit

    return unpack_hyperparameters(best_fit.x, nuisance)


def smooth_series(time, values_centered, weights, hyperparameters):
    """
    Run one full filter-and-smoother pass and read off the two components.

    Parameters
    ----------
    time : ndarray
        Sample times.
    values_centered : ndarray
        Mean-subtracted values.
    weights : ndarray
        Robustness weights.
    hyperparameters : dict of float
        Gaussian-process hyperparameters.

    Returns
    -------
    dict
        ``signal`` and ``rate`` with their variances, the ``nuisance``
        component, and the marginal ``log_likelihood``.
    """
    transitions, process_noises, stationary_covariance = \
        combined_state_space_matrices(np.diff(time), hyperparameters)

    log_likelihood, filtered_means, filtered_covariances, predicted_means, predicted_covariances = \
        run_kalman_filter(values_centered, transitions, process_noises, stationary_covariance,
                          hyperparameters['noise_std'] ** 2 / weights)
    smoothed_means, smoothed_covariances = run_rts_smoother(
        transitions, filtered_means, filtered_covariances, predicted_means, predicted_covariances)

    return {'signal': smoothed_means[:, KINETIC_VALUE_INDEX],
            'signal_variance': np.maximum(
                smoothed_covariances[:, KINETIC_VALUE_INDEX, KINETIC_VALUE_INDEX], 0.0),
            'rate': smoothed_means[:, KINETIC_RATE_INDEX],
            'rate_variance': np.maximum(
                smoothed_covariances[:, KINETIC_RATE_INDEX, KINETIC_RATE_INDEX], 0.0),
            'nuisance': smoothed_means[:, NUISANCE_VALUE_INDEX],
            'log_likelihood': float(log_likelihood)}


def resolve_artifact_gap(median_time_step, duration, number_of_samples,
                         nuisance_lengthscale):
    """
    Choose the blind gap of the two-sided predictive artifact test, in samples.

    The gap sets the longest artifact that can be detected: an excursion that
    outlasts it is still visible to the prediction reaching over it and is
    read as part of the curve. It also has to be longer than the nuisance
    correlation time, otherwise the prediction from just outside the gap
    still carries the same low-frequency wiggle as the sample under test and
    a harmless wiggle would look like an artifact.

    Parameters
    ----------
    median_time_step : float
        Median time step of the series, in seconds.
    duration : float
        Total duration of the series, in seconds.
    number_of_samples : int
        Length of the series.
    nuisance_lengthscale : float
        Correlation time of the correlated noise, in seconds.

    Returns
    -------
    int
        Gap in samples, at least one.
    """
    gap_time = max(ARTIFACT_GAP_MIN_STEPS * median_time_step,
                   ARTIFACT_GAP_FRACTION * duration,
                   ARTIFACT_GAP_NUISANCE_FACTOR * nuisance_lengthscale)

    return int(np.clip(round(gap_time / median_time_step), 1,
                       ARTIFACT_GAP_MAX_FRACTION * number_of_samples))


def predict_across_gap(time, values_centered, weights, hyperparameters, gap):
    """
    Predict every sample from the data that lies at least `gap` samples before it.

    The Kalman filter is run in the usual way and each filtered state is then
    propagated forward across the gap with a single transition matrix, so the
    prediction for a sample never uses the sample itself nor its immediate
    neighbourhood. Its variance is the model's own extrapolation uncertainty,
    which already contains everything the fit does not claim to know: the
    kinetic curve's freedom to bend, and the full variance of the nuisance
    component once it has decorrelated.

    Parameters
    ----------
    time, values_centered : ndarray
        Sample times and mean-subtracted values.
    weights : ndarray
        Robustness weights for the filter pass.
    hyperparameters : dict of float
        Gaussian-process hyperparameters.
    gap : int
        Number of samples skipped between the last observation used and the
        sample being predicted.

    Returns
    -------
    tuple of ndarray
        Predicted values and their variances; the first `gap` entries are NaN
        because no prediction of that kind exists for them.
    """
    transitions, process_noises, stationary_covariance = \
        combined_state_space_matrices(np.diff(time), hyperparameters)
    _, filtered_means, filtered_covariances, _, _ = run_kalman_filter(
        values_centered, transitions, process_noises, stationary_covariance,
        hyperparameters['noise_std'] ** 2 / weights)

    gap_transitions, gap_noises, _ = combined_state_space_matrices(
        time[gap:] - time[:-gap], hyperparameters)

    means = np.einsum('kij,kj->ki', gap_transitions, filtered_means[:-gap])
    covariances = gap_transitions @ filtered_covariances[:-gap] \
        @ np.transpose(gap_transitions, (0, 2, 1)) + gap_noises

    predictions = np.full(len(time), np.nan)
    variances = np.full(len(time), np.nan)
    predictions[gap:] = means[:, KINETIC_VALUE_INDEX] + means[:, NUISANCE_VALUE_INDEX]
    variances[gap:] = (covariances[:, KINETIC_VALUE_INDEX, KINETIC_VALUE_INDEX]
                       + 2.0 * covariances[:, KINETIC_VALUE_INDEX, NUISANCE_VALUE_INDEX]
                       + covariances[:, NUISANCE_VALUE_INDEX, NUISANCE_VALUE_INDEX])

    return predictions, variances


def grow_excursions(detected, deviation_forward, deviation_backward, tolerance_forward,
                    tolerance_backward, growth_cap):
    """
    Extend each detected excursion over the whole transient it belongs to.

    A bubble does not end where it stops being conspicuous: it jumps, then
    relaxes back over many samples whose individual deviation is too small to
    detect but whose sum is exactly the level offset that inflates a rate.
    Each detected run is therefore extended in both directions until the trace
    comes back to what the *clean* side predicts -- forwards against the
    prediction from after the transient, backwards against the one from
    before it, so the reference never contains the artifact itself.

    Growth that runs past `growth_cap` without ever returning is undone. A
    deviation that persists is not a transient but a genuine change of regime,
    and the samples after it are the new normal, not artifacts.

    Parameters
    ----------
    detected : ndarray of bool
        Samples the deviation test flagged.
    deviation_forward, deviation_backward : ndarray
        Signed deviations from the prediction reaching forwards over the gap
        and from the one reaching backwards.
    tolerance_forward, tolerance_backward : ndarray
        Deviation magnitudes counting as "back on the curve" for each; these
        are tighter than the detection thresholds, so growth stops only once
        the trace is properly back and not merely less conspicuous.
    growth_cap : int
        Largest extension, in samples, in either direction.

    Returns
    -------
    ndarray of bool
        The grown excursion mask.
    """
    excursions = detected.copy()
    edges = np.diff(np.concatenate([[0], detected.astype(np.int8), [0]]))

    for run_start, run_stop in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        sign = np.sign(np.median(deviation_backward[run_start:run_stop]))

        tail = np.arange(run_stop, min(len(detected), run_stop + growth_cap))
        head = np.arange(max(0, run_start - growth_cap), run_start)[::-1]

        returned_tail = first_settled(sign * deviation_backward[tail] <= tolerance_backward[tail])
        returned_head = first_settled(sign * deviation_forward[head] <= tolerance_forward[head])

        # A transient has to end on both sides; an excursion that never comes
        # back is a permanent change of regime and its samples are the new
        # normal, not artifacts.
        if returned_tail is None or (returned_head is None and run_start > 0):
            continue

        excursions[tail[:returned_tail]] = True
        if returned_head is not None:
            excursions[head[:returned_head]] = True

    return excursions


def first_settled(on_trend):
    """
    Position of the first run of `ARTIFACT_SETTLED_SAMPLES` on-trend samples.

    A single sample crossing back inside the tolerance proves nothing when
    the tolerance is of the order of the noise, so a return has to hold for a
    few samples in a row.

    Parameters
    ----------
    on_trend : ndarray of bool
        Whether each sample is back within tolerance.

    Returns
    -------
    int or None
        Index of the start of the first settled run, or None if there is none.
    """
    if len(on_trend) < ARTIFACT_SETTLED_SAMPLES:
        return None

    settled = np.convolve(on_trend.astype(int), np.ones(ARTIFACT_SETTLED_SAMPLES, dtype=int),
                          mode='valid') == ARTIFACT_SETTLED_SAMPLES

    return int(np.argmax(settled)) if settled.any() else None


def artifact_weights(time, values_centered, weights, hyperparameters, reference_lengthscale,
                     gap, threshold):
    """
    Reject excursions the trace comes back from, and nothing else.

    Every sample is predicted twice by `predict_across_gap`: once from the
    data more than `gap` samples before it, once from the data more than
    `gap` samples after it. A sample belongs to an artifact when all three of
    the following hold.

    - The data disagree with the prediction from the past.
    - The data disagree with the prediction from the future.
    - The two predictions nevertheless agree *with each other*.

    The third condition is what protects the kinetics. Across a genuine
    transition -- a light-on step, a steep sigmoidal onset, a kink -- the two
    sides see different curves and say so, and no sample there is touched,
    however badly either side predicts it on its own. Across a bubble both
    sides describe the same undisturbed curve and only the data depart from
    it. `grow_excursions` then extends each detection over the transient's
    full relaxation, and undoes the growth wherever the trace never returns.

    None of this uses the fitted curve at the sample itself, which is what
    lets it see an artifact the fit has already bent to follow. Residual-based
    reweighting alone cannot: a length scale short enough to track a bubble
    leaves no residual to flag. For the same reason the kinetic length scale
    used here is `reference_lengthscale`, which is the fitted one raised to at
    least `ARTIFACT_STIFFNESS_FACTOR` times the gap. A fit that has already
    bent around an artifact predicts almost nothing across the gap -- its
    extrapolation uncertainty is then as large as the whole trace -- and would
    clear that artifact of suspicion. Raising it only to what the gap needs,
    rather than to the smoothest curve allowed, keeps the test from calling
    the steepest stretch of a genuinely fast reaction an artifact.

    Parameters
    ----------
    time, values_centered : ndarray
        Sample times and mean-subtracted values.
    weights : ndarray
        Robustness weights for the filter passes.
    hyperparameters : dict of float
        Gaussian-process hyperparameters; only the kinetic length scale is
        overridden.
    reference_lengthscale : float
        Kinetic length scale to assume while hunting artifacts.
    gap : int
        Blind gap in samples; artifacts up to roughly this long are detected
        directly, longer ones through the growth step.
    threshold : float
        Standardized deviation at which a sample counts as unexplained.

    Returns
    -------
    ndarray
        Weight per sample: one outside excursions, `ROBUST_WEIGHT_FLOOR`
        inside them.
    """
    hyperparameters = dict(hyperparameters, lengthscale=reference_lengthscale)
    noise_variance = hyperparameters['noise_std'] ** 2

    forward, forward_variance = predict_across_gap(time, values_centered, weights,
                                                   hyperparameters, gap)
    backward, backward_variance = predict_across_gap(time[::-1] * -1.0, values_centered[::-1],
                                                     weights[::-1], hyperparameters, gap)
    backward, backward_variance = backward[::-1], backward_variance[::-1]

    deviation_forward = values_centered - forward
    deviation_backward = values_centered - backward
    tolerance_forward = threshold * np.sqrt(forward_variance + noise_variance)
    tolerance_backward = threshold * np.sqrt(backward_variance + noise_variance)
    disagreement = np.abs(forward - backward) / np.sqrt(forward_variance + backward_variance
                                                        + 2.0 * noise_variance)

    # Samples too close to either end have only one prediction and stay untested.
    detected = np.nan_to_num((np.abs(deviation_forward) > tolerance_forward)
                             & (np.abs(deviation_backward) > tolerance_backward)
                             & (disagreement <= threshold), nan=False)

    # Hysteresis: a sample has to be conspicuous to start an excursion but
    # only unremarkable to end one, so the whole relaxation tail is covered.
    settle = ARTIFACT_RETURN_TOLERANCE / threshold
    excursions = grow_excursions(detected, deviation_forward, deviation_backward,
                                 settle * tolerance_forward, settle * tolerance_backward,
                                 int(ARTIFACT_GROWTH_CAP_FACTOR * gap))

    return np.where(excursions, ROBUST_WEIGHT_FLOOR, 1.0)


def robust_weights_from_zscores(zscores, threshold):
    """
    Redescending robustness weights from standardized deviations.

    Samples within `threshold` keep full weight; beyond it the weight falls
    off as the square of the ratio, so a sample's influence -- weight times
    deviation -- *decreases* the further out it lies. This is the IRLS weight
    of a Cauchy loss, and the squaring matters: with the merely bounded
    influence of a Huber weight, the several hundred samples of one bubble
    transient still pull the fit hard enough to shorten the kinetic length
    scale to the bubble's own width. Weights are floored rather than zeroed
    so that no sample is ever fully removed.

    Parameters
    ----------
    zscores : ndarray
        Standardized absolute deviations.
    threshold : float
        Deviation at which downweighting starts.

    Returns
    -------
    ndarray
        Weight per sample, in (`ROBUST_WEIGHT_FLOOR`, 1].
    """
    ratio = threshold / np.maximum(np.nan_to_num(zscores, nan=0.0), threshold)

    return np.clip(ratio ** 2, ROBUST_WEIGHT_FLOOR, 1.0)


def robust_weights(residuals, noise_std, threshold):
    """
    Huber weights from the residuals of a fitted model.

    This is the second half of the robustness stage: it catches whatever the
    fit did *not* follow, chiefly single-sample spikes, and refines the
    weights once the curve is close to right.

    Parameters
    ----------
    residuals : ndarray
        Observation minus fitted value.
    noise_std : float
        White-noise standard deviation of the fit.
    threshold : float
        Standardized residual at which downweighting starts.

    Returns
    -------
    ndarray
        Weight per sample, in (`ROBUST_WEIGHT_FLOOR`, 1].
    """
    scale = max(noise_std, robust_scale(residuals), np.finfo(float).tiny)

    return robust_weights_from_zscores(np.abs(residuals) / scale, threshold)


def calculate_windowed_rates(time, signal, rate_variance, window, weights):
    """
    Calculate the average rate over a centred window from the kinetic component.

    The windowed rate at time t is (f(t + w/2) - f(t - w/2)) / w, which
    equals the mean of the derivative over the window and suppresses any
    residual short-lived artifact contribution by the ratio of artifact
    duration to window length. Windows whose samples are mostly downweighted
    are dropped: there the curve is an unsupported bridge, so its slope is
    not evidence of a rate.

    Its uncertainty is taken as the mean posterior variance of the
    derivative across the window. That is the variance the window average
    would have if the derivative errors were perfectly correlated over it,
    so it is an upper bound -- exact for windows shorter than the kinetic
    length scale, conservative for longer ones. Differencing the endpoint
    values instead would need their posterior covariance, which the marginal
    smoother output does not carry; assuming independence there badly
    overstates the error once a nuisance component makes the absolute level
    of the kinetic curve ambiguous.

    Parameters
    ----------
    time : ndarray
        Sample times.
    signal, rate_variance : ndarray
        Posterior mean of the kinetic component and posterior variance of
        its derivative.
    window : float
        Window length in time units.
    weights : ndarray
        Robustness weights.

    Returns
    -------
    tuple of ndarray
        Window centre times, windowed rates and their standard deviations.
    """
    half_window = 0.5 * window
    inside = (time >= time[0] + half_window) & (time <= time[-1] - half_window)
    centers = time[inside]

    # Mean weight and mean rate variance inside each window, via prefix sums
    cumulative_weights = np.concatenate([[0.0], np.cumsum(weights)])
    cumulative_variances = np.concatenate([[0.0], np.cumsum(rate_variance)])
    window_first = np.searchsorted(time, centers - half_window, side='left')
    window_last = np.searchsorted(time, centers + half_window, side='right')
    window_points = np.maximum(window_last - window_first, 1)

    mean_weight = (cumulative_weights[window_last] - cumulative_weights[window_first]) \
        / window_points
    mean_variance = (cumulative_variances[window_last] - cumulative_variances[window_first]) \
        / window_points

    supported = mean_weight >= MIN_WINDOW_WEIGHT_FRACTION
    if not supported.any():
        raise ValueError('no window with sufficient unmasked data; '
                         'inspect the series or adjust outlier settings')
    centers = centers[supported]

    curve_low = np.interp(centers - half_window, time, signal)
    curve_high = np.interp(centers + half_window, time, signal)

    windowed_rates = (curve_high - curve_low) / window
    windowed_rate_stds = np.sqrt(mean_variance[supported])

    return centers, windowed_rates, windowed_rate_stds


def calculate_rolling_slopes(time, values, weights, window, median_time_step):
    """
    Calculate the weighted least-squares slope of the raw data in a sliding window.

    This is the smoothing-free cross-check for the Gaussian-process result,
    computed in O(n) with prefix sums. Downweighted samples contribute in
    proportion to their weight.

    Parameters
    ----------
    time, values : ndarray
        Sample times and values.
    weights : ndarray
        Robustness weights.
    window : float
        Window length in time units.
    median_time_step : float
        Median time step, used to scale how many points a window must hold.

    Returns
    -------
    ndarray
        Window slope centred on each sample; NaN where the window is
        incomplete or holds too few effective points.
    """
    # Requiring `MIN_WINDOW_POINTS` outright was a guard for densely sampled
    # data. On a coarsely sampled series a correctly sized window holds only a
    # handful of samples, and insisting on ten of them silently turns the whole
    # cross-check into NaN -- exactly where a second opinion is most wanted.
    minimum_points = max(3, min(MIN_WINDOW_POINTS, int(0.5 * window / median_time_step)))

    time_shifted = time - time[0]

    # Prefix sums of the weighted regression terms
    zero = np.zeros(1)
    sum_weights = np.concatenate([zero, np.cumsum(weights)])
    sum_time = np.concatenate([zero, np.cumsum(weights * time_shifted)])
    sum_values = np.concatenate([zero, np.cumsum(weights * values)])
    sum_time_squared = np.concatenate([zero, np.cumsum(weights * time_shifted ** 2)])
    sum_time_values = np.concatenate([zero, np.cumsum(weights * time_shifted * values)])

    window_first = np.searchsorted(time, time - 0.5 * window, side='left')
    window_last = np.searchsorted(time, time + 0.5 * window, side='right')

    points = sum_weights[window_last] - sum_weights[window_first]
    total_time = sum_time[window_last] - sum_time[window_first]
    total_values = sum_values[window_last] - sum_values[window_first]
    total_time_squared = sum_time_squared[window_last] - sum_time_squared[window_first]
    total_time_values = sum_time_values[window_last] - sum_time_values[window_first]

    denominator = points * total_time_squared - total_time ** 2
    complete = (time - 0.5 * window >= time[0]) & (time + 0.5 * window <= time[-1])
    valid = complete & (points >= minimum_points) & (denominator > 0)

    slopes = np.full(len(time), np.nan)
    slopes[valid] = (points * total_time_values - total_time * total_values)[valid] / denominator[valid]

    return slopes


def collect_quality_flags(result, time, lengthscale_bounds):
    """
    Collect human-review flags for a finished extraction.

    Parameters
    ----------
    result : MaxRateResult
        Populated result.
    time : ndarray
        Sample times in seconds; the result does not store them.
    lengthscale_bounds : tuple of float
        Bounds in seconds used during hyperparameter fitting.

    Returns
    -------
    list of str
        Flags; empty when nothing is suspicious.
    """
    max_rate = result.max_rate.unit[RATE_UNIT]
    max_rate_std = result.max_rate_std.unit[RATE_UNIT]
    t_max_rate = result.t_max_rate.unit[TIME_UNIT]
    window = result.window.unit[TIME_UNIT]
    crosscheck = result.max_rate_crosscheck.unit[RATE_UNIT]

    flags = []

    if t_max_rate < time[0] + window or t_max_rate > time[-1] - window:
        flags.append('max_rate_at_boundary')

    # The window hit its duration cap, so it was the length of the run and not
    # the sampling that set it: the series is short enough that the window
    # holds only a handful of points and the headline number is noisier.
    if window >= WINDOW_MAX_SPAN_FRACTION * (time[-1] - time[0]):
        flags.append('window_duration_limited')

    if result.hyperparameters['lengthscale'].unit[TIME_UNIT] \
            < LENGTHSCALE_BOUND_FACTOR * lengthscale_bounds[0]:
        flags.append('lengthscale_at_lower_bound')

    if max_rate < SIGNIFICANCE_SIGMA * max_rate_std:
        flags.append('max_rate_not_significant')

    if max_rate_std > HIGH_UNCERTAINTY_FRACTION * abs(max_rate):
        flags.append('high_uncertainty')

    if result.diagnostics['outlier_fraction'] > OUTLIER_FRACTION_WARNING:
        flags.append('many_outliers_masked')

    # Only meaningful once the variogram actually resolved a correlated
    # component; otherwise its correlation time is a placeholder time step and
    # the slope scale derived from it is arbitrary.
    if result.hyperparameters['nuisance_lengthscale'].unit[TIME_UNIT] \
            > result.diagnostics['median_dt'].unit[TIME_UNIT] \
            and result.diagnostics['nuisance_rate_std'].unit[RATE_UNIT] \
            > CORRELATED_NOISE_RATE_FRACTION * abs(max_rate):
        flags.append('strong_correlated_noise')

    if np.isfinite(crosscheck):
        difference = abs(max_rate - crosscheck)
        combined_std = np.hypot(max_rate_std, result.diagnostics['crosscheck_std'].unit[RATE_UNIT])
        if difference > DISAGREEMENT_RELATIVE * abs(max_rate) \
                and difference > DISAGREEMENT_SIGMA * combined_std:
            flags.append('estimator_disagreement')

    if max_rate > 0 \
            and result.max_rate_instantaneous.unit[RATE_UNIT] > INSTANTANEOUS_SPIKE_FACTOR * max_rate:
        flags.append('instantaneous_rate_spike')

    if result.diagnostics['residual_lag1_autocorr'] > RESIDUAL_AUTOCORRELATION_WARNING:
        flags.append('correlated_residuals')

    return flags


@dataclass
class MaxRateResult:
    """
    Result of `extract_max_rate`.

    Every physical field is a `Quantity`: rates carry the dimension
    substance / time, amounts substance, and times time. Read them in any
    compatible unit through the lazy lookup, e.g.
    ``result.max_rate.unit['umol / h']``.

    ``max_rate`` is the robust headline number: the largest rate sustained
    over ``window``. ``max_rate_instantaneous`` is the peak of the smoothed
    derivative; it is upward-biased for noisy series (maximum of a noisy
    estimate) and more artifact-sensitive, so treat it as secondary.
    ``max_rate_crosscheck`` is the smoothing-free rolling least-squares
    slope of the raw data over the same window as ``max_rate``; large
    disagreement raises a flag.

    ``smooth`` is the kinetic component alone and is what ``rate``
    differentiates; ``nuisance`` is the correlated-noise component the fit
    separated out. Their sum is the model of the measured trace, so a plot
    of ``smooth`` deliberately does *not* follow every wiggle of the data.
    ``outlier_mask`` marks the samples the robust fit downweighted; unlike a
    hard mask it does not mean those samples were discarded.

    The input series is deliberately *not* stored: a result is meant to be
    saved alongside the dataset it was computed from, and duplicating the
    time and value arrays there would double the stored series for nothing.
    `plot_max_rate` therefore takes the inputs again. The per-sample arrays
    that are kept (``smooth``, ``nuisance``, ``rate``, their uncertainties
    and ``outlier_mask``) exist nowhere else.
    """

    max_rate: Quantity
    max_rate_std: Quantity
    t_max_rate: Quantity
    window: Quantity
    max_rate_instantaneous: Quantity
    max_rate_instantaneous_std: Quantity
    t_max_rate_instantaneous: Quantity
    max_rate_crosscheck: Quantity
    outlier_mask: np.ndarray
    smooth: Quantity
    smooth_std: Quantity
    nuisance: Quantity
    rate: Quantity
    rate_std: Quantity
    hyperparameters: Dict[str, Quantity]
    diagnostics: Dict[str, Any]
    flags: List[str] = field(default_factory=list)


def extract_max_rate(time, values, window=None, robust_threshold=4.0,
                     lengthscale_bounds=None, max_fit_points=1200, hyperparameters=None):
    """
    Extract the maximum rate of a kinetic time series with uncertainty.

    All calculations run in moles and seconds; the returned Quantities can
    be read back in any compatible unit.

    Parameters
    ----------
    time : Quantity
        Sample times (dimension time).
    values : Quantity
        Measured amounts, e.g. evolved H2 (dimension substance).
    window : Quantity, optional
        Length of the sustained-rate window (dimension time). Defaults to
        max(25 median time steps, 2 % of the series duration).
    robust_threshold : float
        Standardized residual at which a sample starts to be downweighted.
    lengthscale_bounds : tuple of Quantity, optional
        (lower, upper) bounds for the kinetic length scale (dimension time).
        Defaults to (max(20 median time steps, 0.2 % of duration, twice the
        measured noise correlation time), duration / 2). The lower bound is
        what stops the kinetic component from tracking correlated noise.
    max_fit_points : int
        Points used (after decimation) for hyperparameter optimization.
        The final smoothing passes always use every point.
    hyperparameters : dict of Quantity, optional
        ``lengthscale``, ``signal_std``, ``nuisance_lengthscale``,
        ``nuisance_std`` and ``noise_std`` to reuse from a previous fit,
        skipping the optimization (useful for batches of similar
        experiments). Pass ``MaxRateResult.hyperparameters`` straight back.

    Returns
    -------
    MaxRateResult
        Maximum rate estimates with uncertainties, the kinetic and nuisance
        curves, the rate curve, and quality flags for human review.
    """
    time, values = validate_time_series(time, values)

    median_time_step = float(np.median(np.diff(time)))
    duration = time[-1] - time[0]

    window = resolve_window(window, median_time_step, duration)
    if not 0 < window < duration:
        raise ValueError(f'window {window} s outside series duration {duration} s')

    # Stage 1: how much of the scatter is correlated, and over what time
    noise_structure = estimate_noise_structure(time, values)
    lengthscale_bounds = resolve_lengthscale_bounds(lengthscale_bounds, median_time_step,
                                                    duration,
                                                    noise_structure['correlated_lengthscale'])

    # Stage 2: fit the two-component Gaussian process. Weights start out
    # rejecting gross spikes only, so the fit is not dragged around by them.
    weights = gross_outlier_weights(values, noise_structure['white_std'])
    mean_value = float(np.average(values, weights=weights))
    values_centered = values - mean_value

    reuse_hyperparameters = hyperparameters is not None
    hyperparameters = hyperparameter_magnitudes(hyperparameters) if reuse_hyperparameters \
        else fit_hyperparameters(time, values_centered, weights, lengthscale_bounds,
                                 noise_structure, max_fit_points)

    # Stage 3: find the artifacts the first fit bent itself to follow, then
    # refit without them. A single bubble is enough to pull the kinetic length
    # scale down to its own width, so this refit is what protects every later
    # stage, not just the smoothed curve.
    gap = resolve_artifact_gap(median_time_step, duration, len(time),
                               noise_structure['correlated_lengthscale'])
    reference_lengthscale = min(max(hyperparameters['lengthscale'],
                                   ARTIFACT_STIFFNESS_FACTOR * gap * median_time_step),
                                lengthscale_bounds[1])
    for _ in range(ARTIFACT_PASSES):
        weights = np.minimum(weights, artifact_weights(time, values_centered, weights,
                                                       hyperparameters, reference_lengthscale,
                                                       gap, robust_threshold))
    if not reuse_hyperparameters:
        hyperparameters = fit_hyperparameters(time, values_centered, weights, lengthscale_bounds,
                                              noise_structure, max_fit_points,
                                              previous_fit=hyperparameters)

    for _ in range(ROBUST_PASSES):
        smoothed = smooth_series(time, values_centered, weights, hyperparameters)
        residuals = values_centered - smoothed['signal'] - smoothed['nuisance']
        weights = np.minimum(weights, robust_weights(residuals, hyperparameters['noise_std'],
                                                     robust_threshold))
    smoothed = smooth_series(time, values_centered, weights, hyperparameters)

    smooth = mean_value + smoothed['signal']
    rate = smoothed['rate']
    rate_std = np.sqrt(smoothed['rate_variance'])

    # Stage 4: the headline number is the largest window-averaged rate
    centers, windowed_rates, windowed_rate_stds = calculate_windowed_rates(
        time, smooth, smoothed['rate_variance'], window, weights)
    best_window = int(np.argmax(windowed_rates))

    # Instantaneous max only where the data support it; inside a stretch of
    # downweighted samples the derivative is an unsupported interpolation
    supported = np.nonzero(weights >= OUTLIER_WEIGHT_THRESHOLD)[0]
    best_instantaneous = int(supported[np.argmax(rate[supported])])

    # Stage 5: cross-check at the same window centre. The global maximum of
    # the raw rolling slope is itself upward-biased (winner's curse) on
    # noisy data, so it is reported as a diagnostic rather than flagged on.
    rolling_slopes = calculate_rolling_slopes(time, values, weights, window, median_time_step)
    crosscheck = rolling_slopes[int(np.argmin(np.abs(time - centers[best_window])))]

    # Uncertainty of a least-squares slope over ~n equidistant points spanning
    # the window; correlated noise adds to the white-noise term
    total_noise_std = np.hypot(hyperparameters['noise_std'], hyperparameters['nuisance_std'])
    crosscheck_std = total_noise_std * np.sqrt(12.0 * median_time_step / window ** 3)

    # Standard deviation of the derivative of the nuisance component: the
    # spurious rate the correlated noise could contribute if it were mistaken
    # for kinetics.
    nuisance_rate_std = np.sqrt(3.0) * hyperparameters['nuisance_std'] \
        / hyperparameters['nuisance_lengthscale']

    residuals = values_centered - smoothed['signal'] - smoothed['nuisance']
    residual_autocorrelation = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) \
        if len(residuals) > 2 else np.nan

    max_rolling_slope = float(np.nanmax(rolling_slopes)) \
        if np.any(np.isfinite(rolling_slopes)) else np.nan

    outlier_mask = weights < OUTLIER_WEIGHT_THRESHOLD

    diagnostics = {
        'log_likelihood': smoothed['log_likelihood'],
        'outlier_fraction': float(outlier_mask.mean()),
        'residual_lag1_autocorr': residual_autocorrelation,
        'median_dt': Quantity(median_time_step, TIME_UNIT),
        'max_rolling_slope': Quantity(max_rolling_slope, RATE_UNIT),
        'crosscheck_std': Quantity(float(crosscheck_std), RATE_UNIT),
        'nuisance_rate_std': Quantity(float(nuisance_rate_std), RATE_UNIT),
        'lengthscale_lower_bound': Quantity(float(lengthscale_bounds[0]), TIME_UNIT),
    }

    result = MaxRateResult(
        max_rate = Quantity(float(windowed_rates[best_window]), RATE_UNIT),
        max_rate_std = Quantity(float(windowed_rate_stds[best_window]), RATE_UNIT),
        t_max_rate = Quantity(float(centers[best_window]), TIME_UNIT),
        window = Quantity(float(window), TIME_UNIT),
        max_rate_instantaneous = Quantity(float(rate[best_instantaneous]), RATE_UNIT),
        max_rate_instantaneous_std = Quantity(float(rate_std[best_instantaneous]), RATE_UNIT),
        t_max_rate_instantaneous = Quantity(float(time[best_instantaneous]), TIME_UNIT),
        max_rate_crosscheck = Quantity(float(crosscheck), RATE_UNIT),
        outlier_mask = outlier_mask,
        smooth = Quantity(smooth, AMOUNT_UNIT),
        smooth_std = Quantity(np.sqrt(smoothed['signal_variance']), AMOUNT_UNIT),
        nuisance = Quantity(smoothed['nuisance'], AMOUNT_UNIT),
        rate = Quantity(rate, RATE_UNIT),
        rate_std = Quantity(rate_std, RATE_UNIT),
        hyperparameters = {'lengthscale': Quantity(hyperparameters['lengthscale'], TIME_UNIT),
                           'signal_std': Quantity(hyperparameters['signal_std'], AMOUNT_UNIT),
                           'nuisance_lengthscale': Quantity(hyperparameters['nuisance_lengthscale'],
                                                            TIME_UNIT),
                           'nuisance_std': Quantity(hyperparameters['nuisance_std'], AMOUNT_UNIT),
                           'noise_std': Quantity(hyperparameters['noise_std'], AMOUNT_UNIT)},
        diagnostics = diagnostics,
    )
    result.flags = collect_quality_flags(result, time, lengthscale_bounds)

    return result


def plot_max_rate(result, time, values, axes=None):
    """
    Plot a two-panel diagnostic figure: data with the fit, and rate with confidence band.

    The upper panel shows the kinetic component and, dashed, the full model
    (kinetic plus nuisance). Where the two separate, the fit has assigned
    that structure to correlated noise and kept it out of the rate.

    The input series is not stored on the result, so it is passed in again
    here; it is re-validated so that it lines up with the smoothed curve.

    Parameters
    ----------
    result : MaxRateResult
        Result to visualize.
    time : Quantity
        The sample times the result was computed from (dimension time).
    values : Quantity
        The measured amounts the result was computed from (dimension substance).
    axes : tuple of matplotlib.axes.Axes, optional
        (data axis, rate axis); a new figure is created when omitted.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The axes drawn on.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    data_axis, rate_axis = axes

    time, values = validate_time_series(time, values)
    mask = result.outlier_mask
    smooth = result.smooth.unit[AMOUNT_UNIT]
    nuisance = result.nuisance.unit[AMOUNT_UNIT]
    rate = result.rate.unit[RATE_UNIT]
    rate_std = result.rate_std.unit[RATE_UNIT]
    max_rate = result.max_rate.unit[RATE_UNIT]
    max_rate_std = result.max_rate_std.unit[RATE_UNIT]
    t_max_rate = result.t_max_rate.unit[TIME_UNIT]

    # Upper panel: raw data, downweighted samples, both model components and
    # the max-rate window
    data_axis.plot(time, values, '.', ms=3.5, color='0.6', label='data')
    if mask.any():
        data_axis.plot(time[mask], values[mask], 'x', ms=3, color='crimson',
                       label='downweighted')
    data_axis.plot(time, smooth + nuisance, color='0.3', lw=0.8, ls='--',
                   label='kinetic + nuisance')
    data_axis.plot(time, smooth, color='C0', lw=1.5, label='kinetic component')

    half_window = 0.5 * result.window.unit[TIME_UNIT]
    window_times = np.array([t_max_rate - half_window, t_max_rate + half_window])
    window_values = np.interp(window_times, time, smooth)
    data_axis.plot(window_times, window_values, color='C3', lw=2.5, label='max rate window')
    data_axis.set_ylabel(f'amount / {AMOUNT_UNIT}')
    data_axis.legend(fontsize=8)

    # Lower panel: rate curve with confidence band and the extracted maximum
    rate_axis.plot(time, rate, color='C0', lw=1.2, label='rate (GP derivative)')
    rate_axis.fill_between(time, rate - 2 * rate_std, rate + 2 * rate_std,
                           color='C0', alpha=0.25, lw=0, label='±2σ')
    rate_axis.axhline(max_rate, color='C3', ls='--', lw=1,
                      label=f'max rate = {max_rate:.3g} ± {max_rate_std:.2g} {RATE_UNIT}')
    rate_axis.axvline(t_max_rate, color='C3', ls=':', lw=1)
    rate_axis.set_xlabel(f'time / {TIME_UNIT}')
    rate_axis.set_ylabel(f'rate / {RATE_UNIT}')
    rate_axis.legend(fontsize=8)

    if result.flags:
        rate_axis.set_title('flags: ' + ', '.join(result.flags), fontsize=9, color='crimson')

    return axes


def test_function():

    import matplotlib.pyplot as plt

    # Synthetic experiment: induction period, linear H2 evolution in umol,
    # a slow sinusoidal baseline wave whose slope rivals the true rate, and
    # one bubble artifact whose instantaneous slope is 75x the true rate
    rng = np.random.default_rng(0)
    time_seconds = np.arange(0.0, 8000.0, 1.0)
    signal = 0.02 * np.clip(time_seconds - 1000.0, 0.0, None)

    baseline_wave = 3.0 * np.sin(2.0 * np.pi * time_seconds / 400.0)

    artifact = np.zeros_like(time_seconds)
    artifact[5000:5010] = np.linspace(0.0, 15.0, 10)
    artifact[5010:] = 15.0 * np.exp(-(time_seconds[5010:] - time_seconds[5010]) / 100.0)

    time = Quantity(time_seconds, 's')
    values = Quantity(signal + baseline_wave + artifact
                      + 0.2 * rng.standard_normal(len(time_seconds)), 'umol')

    result = extract_max_rate(time, values)

    print(f"max rate: {result.max_rate.unit['umol / s']:.4f} "
          f"± {result.max_rate_std.unit['umol / s']:.4f} umol/s "
          f"(true 0.0200) at t = {result.t_max_rate.unit['s']:.0f} s")
    print(f"in umol/h: {result.max_rate.unit['umol / h']:.2f}")
    print(f"cross-check: {result.max_rate_crosscheck.unit['umol / s']:.4f}, flags: {result.flags}")
    print(f"noise: white {result.hyperparameters['noise_std'].unit['umol']:.3f} umol, "
          f"correlated {result.hyperparameters['nuisance_std'].unit['umol']:.3f} umol "
          f"over {result.hyperparameters['nuisance_lengthscale'].unit['s']:.0f} s")

    plot_max_rate(result, time, values)
    plt.show()

def test_function_experimental_data():

    from pyKES.database.database_experiments import ExperimentalDataset
    import matplotlib.pyplot as plt
    

    data = np.genfromtxt('/Users/jacob/Downloads/MRG-059-Z-1-3.csv', delimiter=',', skip_header = 1)


    time = Quantity(data[:,0], 's')
    values = Quantity(data[:,1], 'umol')

    result = extract_max_rate(time, values)

    plot_max_rate(result, time, values)
    plt.show()



    



if __name__ == "__main__":
    #test_function()
    test_function_experimental_data()
