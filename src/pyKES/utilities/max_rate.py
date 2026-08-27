"""Robust extraction of maximum rates from noisy kinetic time series.

The pipeline runs in four stages:

1. Artifact masking: samples corrupted by sensor artifacts (bubbles,
   spikes, level shifts) are detected from abnormally large jumps and
   excluded from fitting.
2. Smoothing: a Matern-5/2 Gaussian process is fitted to the data with an
   exact O(n) Kalman filter + RTS smoother, yielding the smoothed curve,
   its time derivative (the rate) and uncertainties for both.
3. Rate extraction: the headline number is the largest rate sustained over
   a time window, read from the smoothed curve. Short artifacts can have
   instantaneous slopes an order of magnitude above the true kinetic rate,
   so the sustained-window definition is what makes the result robust.
4. Cross-checking: a rolling linear regression of the raw data provides a
   smoothing-free second opinion, and automatic quality flags mark curves
   that need human review.

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
from scipy.ndimage import binary_dilation, median_filter
from scipy.optimize import minimize

from pyKES.utilities.unit_handler import Quantity

# --- Calculation units --------------------------------------------------------
# Every stage of the pipeline works on plain floats in these units: Quantity
# inputs are reduced to magnitudes on entry and results are re-wrapped on exit.
TIME_UNIT = 's'
AMOUNT_UNIT = 'mol'
RATE_UNIT = 'mol / s'

# --- Statistical conversion factors ----------------------------------------
MAD_TO_STD = 1.4826                  # median absolute deviation -> standard deviation (Gaussian)
STATE_DIMENSION = 3                  # Matern-5/2 state: (value, first derivative, second derivative)

# --- Artifact detection ------------------------------------------------------
JUMP_DETECTION_LAGS = (1, 5)         # sample lags at which increments are tested for jumps
TREND_WINDOW_MAX_SAMPLES = 101       # rolling-median window for the local kinetic trend
NOISE_WINDOW_MAX_SAMPLES = 601       # rolling-median window for the local noise level
GLOBAL_NOISE_FLOOR_FRACTION = 0.5    # local noise level may not drop below this fraction of the global one
MAX_MASKED_FRACTION = 0.5            # if detection would mask more than this, assume it misfired

# --- Transient (bubble) mask growth -----------------------------------------
TRANSIENT_ZSCORE_FACTOR = 2.0        # only jumps this many times above threshold start mask growth
TRANSIENT_RETURN_TOLERANCE_STD = 4.0 # signal counts as "back on trend" within this many noise stds
TRANSIENT_OFFSET_GATE_STD = 8.0      # growth only starts if the far side is offset by at least this
TRANSIENT_SETTLED_SAMPLES = 5        # consecutive on-trend samples that end a transient
TRANSIENT_GROWTH_CAP_SAMPLES = 300   # minimum growth range; scales up with series length
ANCHOR_FIT_SAMPLES = 200             # samples used to fit the extrapolation line before a transient
ANCHOR_FIT_MIN_SAMPLES = 20          # minimum samples for that fit

# --- Rate windows and hyperparameter defaults -------------------------------
WINDOW_MEDIAN_STEPS = 25             # default window: at least this many median time steps ...
WINDOW_SPAN_FRACTION = 0.02          # ... and at least this fraction of the series duration
LENGTHSCALE_MIN_STEPS = 20           # lengthscale floor in median time steps (guards against noise fitting)
LENGTHSCALE_MIN_SPAN_FRACTION = 0.002
LENGTHSCALE_MAX_SPAN_FRACTION = 0.5
MIN_WINDOW_POINTS = 10               # rolling regression needs at least this many points per window
MIN_UNMASKED_WINDOW_FRACTION = 0.5   # windows with less unmasked data than this are not trusted

# --- Quality-flag thresholds -------------------------------------------------
HIGH_UNCERTAINTY_FRACTION = 0.5      # flag if the uncertainty exceeds this fraction of the max rate
OUTLIER_FRACTION_WARNING = 0.05      # flag if more than this fraction of samples was masked
DISAGREEMENT_RELATIVE = 0.2          # estimators must differ by more than this fraction ...
DISAGREEMENT_SIGMA = 3.0             # ... and this many combined standard deviations to flag
INSTANTANEOUS_SPIKE_FACTOR = 3.0     # flag if the instantaneous max exceeds this multiple of the windowed max
RESIDUAL_AUTOCORRELATION_WARNING = 0.9
LENGTHSCALE_BOUND_FACTOR = 2.0       # flag if the lengthscale sits within this factor of its lower bound


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


def resolve_window(window, median_time_step, duration):
    """
    Resolve the sustained-rate window to a length in seconds.

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
        return max(WINDOW_MEDIAN_STEPS * median_time_step, WINDOW_SPAN_FRACTION * duration)

    return float(quantity_magnitude(window, TIME_UNIT, 'window'))


def resolve_lengthscale_bounds(lengthscale_bounds, median_time_step, duration):
    """
    Resolve the Gaussian-process lengthscale bounds to seconds.

    Parameters
    ----------
    lengthscale_bounds : tuple of Quantity or None
        (lower, upper) bounds with dimension time; the default is used when None.
    median_time_step : float
        Median time step of the series, in seconds.
    duration : float
        Total duration of the series, in seconds.

    Returns
    -------
    tuple of float
        (lower, upper) bounds in seconds.
    """
    if lengthscale_bounds is None:
        return (max(LENGTHSCALE_MIN_STEPS * median_time_step,
                    LENGTHSCALE_MIN_SPAN_FRACTION * duration),
                LENGTHSCALE_MAX_SPAN_FRACTION * duration)

    lower, upper = lengthscale_bounds
    return (float(quantity_magnitude(lower, TIME_UNIT, 'lengthscale_bounds')),
            float(quantity_magnitude(upper, TIME_UNIT, 'lengthscale_bounds')))


def hyperparameter_magnitudes(hyperparameters):
    """
    Reduce a reused hyperparameter dict of Quantities to plain floats.

    Accepts what `MaxRateResult.hyperparameters` holds, so a fit can be
    handed straight back to `extract_max_rate`.

    Parameters
    ----------
    hyperparameters : dict of Quantity
        ``lengthscale`` (time), ``signal_std`` and ``noise_std`` (substance).

    Returns
    -------
    dict of float
        The same entries in seconds and moles.
    """
    return {
        'lengthscale': float(quantity_magnitude(hyperparameters['lengthscale'], TIME_UNIT,
                                                'lengthscale')),
        'signal_std': float(quantity_magnitude(hyperparameters['signal_std'], AMOUNT_UNIT,
                                               'signal_std')),
        'noise_std': float(quantity_magnitude(hyperparameters['noise_std'], AMOUNT_UNIT,
                                              'noise_std')),
    }


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


def detect_artifacts(values, jump_threshold, mask_padding):
    """
    Build a boolean mask of samples corrupted by sensor artifacts.

    At dense sampling, genuine kinetics change the signal by much less than
    the noise level per sample, while bubbles and spikes change it by many
    robust standard deviations within a few samples. Increments are
    therefore tested against the local trend at several lags, so that both
    single-sample spikes and jumps spread over a handful of samples (each
    step individually below threshold) are caught. Gross jumps additionally
    trigger `grow_transient_mask`, which masks the slowly relaxing tail of
    a bubble transient.

    Parameters
    ----------
    values : ndarray
        Measured values.
    jump_threshold : float
        Robust z-score above which an increment is flagged as a jump.
    mask_padding : int
        Extra samples masked on each side of every detected artifact.

    Returns
    -------
    ndarray of bool
        True for samples to exclude from fitting.
    """
    number_of_samples = len(values)
    differences = np.diff(values)

    # Windows scale with series length so that short or coarsely sampled
    # series keep them narrow enough to track genuine curvature.
    trend_window = min(TREND_WINDOW_MAX_SAMPLES, max(11, number_of_samples // 20) | 1, len(differences))
    noise_window = min(NOISE_WINDOW_MAX_SAMPLES, max(31, number_of_samples // 4) | 1)

    trend = median_filter(differences, size=trend_window, mode='nearest')
    cumulative_trend = np.concatenate([[0.0], np.cumsum(trend)])
    noise_std = MAD_TO_STD * np.median(np.abs(differences - trend)) / np.sqrt(2.0)

    mask = np.zeros(number_of_samples, dtype=bool)
    jump_zscores = np.zeros(number_of_samples)

    for lag in JUMP_DETECTION_LAGS:
        if lag >= number_of_samples:
            break

        # Increment over `lag` samples, minus the increment the local kinetic trend predicts
        increments = (values[lag:] - values[:-lag]) - (cumulative_trend[lag:] - cumulative_trend[:-lag])

        local_noise = MAD_TO_STD * median_filter(np.abs(increments), size=min(noise_window, len(increments)),
                                                 mode='nearest')
        global_noise = MAD_TO_STD * np.median(np.abs(increments))
        noise_level = np.maximum(np.maximum(local_noise, GLOBAL_NOISE_FLOOR_FRACTION * global_noise),
                                 np.finfo(float).tiny)

        zscores = np.abs(increments) / noise_level
        np.maximum(jump_zscores[:-lag], zscores, out=jump_zscores[:-lag])

        # Mask both endpoints of every flagged increment, plus lag samples of margin
        jump_starts = np.zeros(number_of_samples, dtype=bool)
        jump_starts[:-lag][zscores > jump_threshold] = True
        mask |= binary_dilation(jump_starts, structure=np.ones(2 * lag + 1, dtype=bool))

    if mask.any():
        # Only gross jumps mark true transients; marginally flagged samples
        # (e.g. at a sharp kinetic onset) stay point-masked without growth.
        mask = grow_transient_mask(values, mask, noise_std, jump_zscores,
                                   TRANSIENT_ZSCORE_FACTOR * jump_threshold)

    if mask_padding > 0 and mask.any():
        mask = binary_dilation(mask, iterations=mask_padding)

    # Never mask most of the series; fall back to no masking if detection misfires
    if mask.sum() > MAX_MASKED_FRACTION * number_of_samples:
        return np.zeros(number_of_samples, dtype=bool)

    return mask


def grow_transient_mask(values, mask, noise_std, jump_zscores, gross_jump_threshold):
    """
    Extend masked jumps over the full extent of each bubble-like transient.

    Bubble artifacts start with an abrupt jump but relax back to the curve
    slowly; the relaxation tail looks locally smooth and survives jump
    detection. For each masked run that starts with a gross jump and whose
    far side is still clearly offset from the trend line extrapolated from
    before the jump, samples stay masked until the signal returns to that
    trend line. Growth that never finds its way back is reverted: a
    deviation that persists is a genuine kinetic regime change (onset,
    light-off), not a transient artifact. Growth runs forward only, because
    the clean anchor is always on the near side of an abrupt jump;
    extrapolating backwards from the far side can cross the true curve and
    fake a "return to trend" at a genuine sharp onset.

    Parameters
    ----------
    values : ndarray
        Measured values.
    mask : ndarray of bool
        Initial jump mask (a grown copy is returned).
    noise_std : float
        Robust per-sample noise standard deviation.
    jump_zscores : ndarray
        Per-sample robust z-score of the detrended increments.
    gross_jump_threshold : float
        Minimum z-score within a masked run for it to be treated as a transient.

    Returns
    -------
    ndarray of bool
        The grown mask.
    """
    number_of_samples = len(values)
    growth_cap = max(TRANSIENT_GROWTH_CAP_SAMPLES, number_of_samples // 20)
    noise_std = max(noise_std, np.finfo(float).tiny)
    return_tolerance = TRANSIENT_RETURN_TOLERANCE_STD * noise_std
    offset_gate = TRANSIENT_OFFSET_GATE_STD * noise_std

    mask = mask.copy()
    run_edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    run_starts = np.nonzero(run_edges == 1)[0]
    run_stops = np.nonzero(run_edges == -1)[0]

    for run_start, run_stop in zip(run_starts, run_stops):
        if np.max(jump_zscores[run_start:run_stop]) < gross_jump_threshold:
            continue

        anchor = run_start - 1
        if anchor < 0 or mask[anchor]:
            continue

        # Extrapolating hundreds of samples within a few noise standard
        # deviations needs a precise slope: fit a line to the unmasked
        # samples preceding the anchor.
        fit_indices = np.arange(max(0, anchor - ANCHOR_FIT_SAMPLES + 1), anchor + 1)
        fit_indices = fit_indices[~mask[fit_indices]]
        if len(fit_indices) < ANCHOR_FIT_MIN_SAMPLES:
            continue
        slope, intercept = np.polyfit(fit_indices - anchor, values[fit_indices], 1)

        # No persistent offset after the run means the transient already
        # recovered (or was a pure spike); nothing to grow
        predicted_at_stop = intercept + slope * (run_stop - anchor)
        if abs(values[run_stop] - predicted_at_stop) < offset_gate:
            continue

        # Scan the samples after the run: the transient ends at the first
        # occurrence of enough consecutive samples back on the trend line
        tail_indices = np.arange(run_stop, min(number_of_samples, run_stop + growth_cap))
        predicted_tail = intercept + slope * (tail_indices - anchor)
        on_trend = np.abs(values[tail_indices] - predicted_tail) <= return_tolerance

        settled_counts = np.convolve(on_trend.astype(int), np.ones(TRANSIENT_SETTLED_SAMPLES, dtype=int),
                                     mode='valid')
        settled_positions = np.nonzero(settled_counts == TRANSIENT_SETTLED_SAMPLES)[0]

        if len(settled_positions) == 0:
            # The signal never returns to the trend line: the gross jump is a
            # genuine regime change (onset, light-off), not a transient. Any
            # marginal jump flags in the scanned range are then collateral of
            # the same sharp feature, so unmask them.
            marginal = tail_indices[jump_zscores[tail_indices] < gross_jump_threshold]
            mask[marginal] = False
            continue

        mask[tail_indices[:settled_positions[0]]] = True

    return mask


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
        Kernel lengthscale (how quickly the underlying curve can change).
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
    nilpotent_part = feedback + decay_rate * np.eye(STATE_DIMENSION)
    nilpotent_steps = nilpotent_part[None, :, :] * time_steps[:, None, None]
    transitions = np.exp(-decay_rate * time_steps)[:, None, None] * (
        np.eye(STATE_DIMENSION)[None] + nilpotent_steps + 0.5 * (nilpotent_steps @ nilpotent_steps))

    process_noises = stationary_covariance[None] - \
        transitions @ stationary_covariance[None] @ np.transpose(transitions, (0, 2, 1))

    return transitions, process_noises, stationary_covariance


def run_kalman_filter(values_centered, transitions, process_noises, stationary_covariance,
                      noise_variance, mask):
    """
    Run the forward Kalman filter, observing the first state component.

    Masked samples are treated as missing: the state is propagated but not
    updated, so the filter simply bridges over them.

    Parameters
    ----------
    values_centered : ndarray
        Mean-subtracted observations.
    transitions, process_noises, stationary_covariance : ndarray
        Output of `matern52_state_space_matrices`.
    noise_variance : float
        Observation noise variance.
    mask : ndarray of bool
        True for observations to skip.

    Returns
    -------
    tuple
        - log_likelihood : float, marginal log-likelihood of the unmasked observations.
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

        # Update with the observation (first state component is the curve value)
        if not mask[sample]:
            innovation_variance = state_covariance[0, 0] + noise_variance
            innovation = values_centered[sample] - state_mean[0]
            gain = state_covariance[:, 0] / innovation_variance

            state_mean = state_mean + gain * innovation
            state_covariance = state_covariance - np.outer(gain, state_covariance[0, :])
            log_likelihood += -0.5 * (np.log(2.0 * np.pi * innovation_variance)
                                      + innovation ** 2 / innovation_variance)

        filtered_means[sample] = state_mean
        filtered_covariances[sample] = state_covariance

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


def negative_log_likelihood(log_parameters, time_steps, values_centered, mask):
    """
    Negative marginal log-likelihood of the Matern-5/2 model, for the optimizer.

    Parameters
    ----------
    log_parameters : ndarray
        Logarithms of (lengthscale, signal_std, noise_std).
    time_steps : ndarray
        Time differences between consecutive samples.
    values_centered : ndarray
        Mean-subtracted observations.
    mask : ndarray of bool
        Observations to skip.

    Returns
    -------
    float
        The negative log-likelihood.
    """
    lengthscale, signal_std, noise_std = np.exp(log_parameters)

    transitions, process_noises, stationary_covariance = \
        matern52_state_space_matrices(time_steps, lengthscale, signal_std ** 2)
    log_likelihood, *_ = run_kalman_filter(values_centered, transitions, process_noises,
                                           stationary_covariance, noise_std ** 2, mask)

    return -log_likelihood


def fit_hyperparameters(time, values_centered, mask, lengthscale_bounds, max_fit_points):
    """
    Fit (lengthscale, signal_std, noise_std) by maximizing the marginal likelihood.

    Fitting runs on a decimated subset of the unmasked points; the Kalman
    likelihood is exact for any sampling pattern, so decimation only trades
    statistical for computational efficiency. Two starting lengthscales are
    tried to avoid local optima.

    Parameters
    ----------
    time : ndarray
        Sample times.
    values_centered : ndarray
        Mean-subtracted values.
    mask : ndarray of bool
        Artifact mask.
    lengthscale_bounds : tuple of float
        (lower, upper) bounds for the lengthscale.
    max_fit_points : int
        Decimation target for the optimization.

    Returns
    -------
    dict
        Fitted ``lengthscale``, ``signal_std`` and ``noise_std``.
    """
    unmasked_indices = np.nonzero(~mask)[0]
    stride = max(1, int(np.ceil(len(unmasked_indices) / max_fit_points)))
    fit_indices = unmasked_indices[::stride]

    fit_times = time[fit_indices]
    fit_values = values_centered[fit_indices]
    fit_time_steps = np.diff(fit_times)
    fit_mask = np.zeros(len(fit_times), dtype=bool)

    # Robust initial guesses: noise from the high-frequency differences,
    # signal amplitude from the overall spread
    noise_std_initial = MAD_TO_STD * np.median(np.abs(np.diff(fit_values))) / np.sqrt(2.0)
    noise_std_initial = max(noise_std_initial, 1e-9 * np.std(fit_values))
    signal_std_initial = np.std(fit_values)
    duration = fit_times[-1] - fit_times[0]

    # Generous bounds keep the optimizer in a numerically safe region
    bounds = [np.log(lengthscale_bounds),
              np.log((1e-3 * signal_std_initial, 1e3 * signal_std_initial)),
              np.log((1e-2 * noise_std_initial, 1e3 * noise_std_initial))]

    best_fit = None
    for lengthscale_initial in (duration / 100.0, duration / 10.0):
        lengthscale_initial = np.clip(lengthscale_initial, *lengthscale_bounds)
        initial_guess = np.log([lengthscale_initial, signal_std_initial, noise_std_initial])

        fit = minimize(negative_log_likelihood, initial_guess,
                       args=(fit_time_steps, fit_values, fit_mask),
                       method='Nelder-Mead', bounds=bounds,
                       options={'xatol': 0.02, 'fatol': 0.1, 'maxiter': 250})

        if best_fit is None or fit.fun < best_fit.fun:
            best_fit = fit

    lengthscale, signal_std, noise_std = np.exp(best_fit.x)

    return {'lengthscale': float(lengthscale), 'signal_std': float(signal_std),
            'noise_std': float(noise_std)}


def calculate_windowed_rates(time, smooth, smooth_variance, window, mask):
    """
    Calculate the average rate over a centred window from the smoothed curve.

    The windowed rate at time t is (f(t + w/2) - f(t - w/2)) / w, which
    equals the mean of the derivative over the window and suppresses any
    residual short-lived artifact contribution by the ratio of artifact
    duration to window length. Windows whose data are mostly masked are
    dropped: there the smoothed curve is an unsupported bridge, so its
    slope is not evidence of a rate.

    Parameters
    ----------
    time : ndarray
        Sample times.
    smooth, smooth_variance : ndarray
        Posterior mean and variance of the smoothed curve.
    window : float
        Window length in time units.
    mask : ndarray of bool
        Artifact mask.

    Returns
    -------
    tuple of ndarray
        Window centre times, windowed rates, and conservative standard
        deviations (independence bound on the two endpoint variances).
    """
    half_window = 0.5 * window
    inside = (time >= time[0] + half_window) & (time <= time[-1] - half_window)
    centers = time[inside]

    # Fraction of unmasked samples inside each window, via prefix sums
    unmasked_counts = np.concatenate([[0.0], np.cumsum(~mask)])
    window_first = np.searchsorted(time, centers - half_window, side='left')
    window_last = np.searchsorted(time, centers + half_window, side='right')
    unmasked_fraction = (unmasked_counts[window_last] - unmasked_counts[window_first]) \
        / np.maximum(window_last - window_first, 1)

    supported = unmasked_fraction >= MIN_UNMASKED_WINDOW_FRACTION
    if not supported.any():
        raise ValueError('no window with sufficient unmasked data; '
                         'inspect the series or adjust outlier settings')
    centers = centers[supported]

    curve_low = np.interp(centers - half_window, time, smooth)
    curve_high = np.interp(centers + half_window, time, smooth)
    variance_low = np.interp(centers - half_window, time, smooth_variance)
    variance_high = np.interp(centers + half_window, time, smooth_variance)

    windowed_rates = (curve_high - curve_low) / window
    windowed_rate_stds = np.sqrt(variance_low + variance_high) / window

    return centers, windowed_rates, windowed_rate_stds


def calculate_rolling_slopes(time, values, mask, window):
    """
    Calculate the least-squares slope of the raw data in a sliding window.

    This is the smoothing-free cross-check for the Gaussian-process result,
    computed in O(n) with prefix sums. Masked samples get zero weight.

    Parameters
    ----------
    time, values : ndarray
        Sample times and values.
    mask : ndarray of bool
        Artifact mask.
    window : float
        Window length in time units.

    Returns
    -------
    ndarray
        Window slope centred on each sample; NaN where the window is
        incomplete or holds fewer than `MIN_WINDOW_POINTS` unmasked points.
    """
    time_shifted = time - time[0]
    weights = (~mask).astype(float)

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
    valid = complete & (points >= MIN_WINDOW_POINTS) & (denominator > 0)

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

    if result.hyperparameters['lengthscale'].unit[TIME_UNIT] \
            < LENGTHSCALE_BOUND_FACTOR * lengthscale_bounds[0]:
        flags.append('lengthscale_at_lower_bound')

    if max_rate_std > HIGH_UNCERTAINTY_FRACTION * abs(max_rate):
        flags.append('high_uncertainty')

    if result.diagnostics['outlier_fraction'] > OUTLIER_FRACTION_WARNING:
        flags.append('many_outliers_masked')

    if np.isfinite(crosscheck):
        difference = abs(max_rate - crosscheck)
        combined_std = np.hypot(max_rate_std, result.diagnostics['crosscheck_std'].unit[RATE_UNIT])
        if difference > DISAGREEMENT_RELATIVE * abs(max_rate) \
                and difference > DISAGREEMENT_SIGMA * combined_std:
            flags.append('estimator_disagreement')

    if result.max_rate_instantaneous.unit[RATE_UNIT] > INSTANTANEOUS_SPIKE_FACTOR * max_rate:
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

    The input series is deliberately *not* stored: a result is meant to be
    saved alongside the dataset it was computed from, and duplicating the
    time and value arrays there would double the stored series for nothing.
    `plot_max_rate` therefore takes the inputs again. The per-sample arrays
    that are kept (``smooth``, ``rate``, their uncertainties and
    ``outlier_mask``) exist nowhere else.
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
    rate: Quantity
    rate_std: Quantity
    hyperparameters: Dict[str, Quantity]
    diagnostics: Dict[str, Any]
    flags: List[str] = field(default_factory=list)


def extract_max_rate(time, values, window=None, outlier_threshold=6.0,
                     outlier_pad=10, lengthscale_bounds=None,
                     max_fit_points=1500, hyperparameters=None):
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
    outlier_threshold : float
        Robust z-score on detrended increments above which samples are
        masked as artifacts.
    outlier_pad : int
        Samples additionally masked on each side of a detected artifact.
    lengthscale_bounds : tuple of Quantity, optional
        (lower, upper) bounds for the Gaussian-process lengthscale
        (dimension time). Defaults to (max(20 median time steps, 0.2 % of
        duration), duration / 2). The lower bound guards against fitting
        correlated sensor noise as signal.
    max_fit_points : int
        Points used (after decimation) for hyperparameter optimization.
        The final smoothing pass always uses every point.
    hyperparameters : dict of Quantity, optional
        ``lengthscale``, ``signal_std``, ``noise_std`` to reuse from a
        previous fit, skipping the optimization (useful for batches of
        similar experiments). Pass ``MaxRateResult.hyperparameters``
        straight back.

    Returns
    -------
    MaxRateResult
        Maximum rate estimates with uncertainties, the smoothed curve and
        rate curve, and quality flags for human review.
    """
    time, values = validate_time_series(time, values)

    median_time_step = float(np.median(np.diff(time)))
    duration = time[-1] - time[0]

    window = resolve_window(window, median_time_step, duration)
    if not 0 < window < duration:
        raise ValueError(f'window {window} s outside series duration {duration} s')

    lengthscale_bounds = resolve_lengthscale_bounds(lengthscale_bounds, median_time_step, duration)

    # Stage 1: mask artifacts, then fit the Gaussian-process hyperparameters
    mask = detect_artifacts(values, outlier_threshold, outlier_pad)
    mean_value = float(np.mean(values[~mask]))
    values_centered = values - mean_value

    hyperparameters = fit_hyperparameters(time, values_centered, mask,
                                          lengthscale_bounds, max_fit_points) \
        if hyperparameters is None else hyperparameter_magnitudes(hyperparameters)

    # Stage 2: smooth the full series; the state directly holds the curve
    # value, the rate, and their uncertainties
    transitions, process_noises, stationary_covariance = matern52_state_space_matrices(
        np.diff(time), hyperparameters['lengthscale'], hyperparameters['signal_std'] ** 2)

    log_likelihood, filtered_means, filtered_covariances, predicted_means, predicted_covariances = \
        run_kalman_filter(values_centered, transitions, process_noises, stationary_covariance,
                          hyperparameters['noise_std'] ** 2, mask)
    smoothed_means, smoothed_covariances = run_rts_smoother(
        transitions, filtered_means, filtered_covariances, predicted_means, predicted_covariances)

    smooth = mean_value + smoothed_means[:, 0]
    smooth_variance = np.maximum(smoothed_covariances[:, 0, 0], 0.0)
    rate = smoothed_means[:, 1]
    rate_std = np.sqrt(np.maximum(smoothed_covariances[:, 1, 1], 0.0))

    # Stage 3: the headline number is the largest window-averaged rate
    centers, windowed_rates, windowed_rate_stds = calculate_windowed_rates(
        time, smooth, smooth_variance, window, mask)
    best_window = int(np.argmax(windowed_rates))

    # Instantaneous max only at observed samples; inside masked gaps the
    # derivative is an unsupported interpolation
    unmasked_indices = np.nonzero(~mask)[0]
    best_instantaneous = int(unmasked_indices[np.argmax(rate[unmasked_indices])])

    # Stage 4: cross-check at the same window centre. The global maximum of
    # the raw rolling slope is itself upward-biased (winner's curse) on
    # noisy data, so it is reported as a diagnostic rather than flagged on.
    rolling_slopes = calculate_rolling_slopes(time, values, mask, window)
    crosscheck = rolling_slopes[int(np.argmin(np.abs(time - centers[best_window])))]

    # Uncertainty of a least-squares slope over ~n equidistant points spanning the window
    crosscheck_std = hyperparameters['noise_std'] * np.sqrt(12.0 * median_time_step / window ** 3)

    residuals = (values - smooth)[~mask]
    residual_autocorrelation = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) \
        if len(residuals) > 2 else np.nan

    max_rolling_slope = float(np.nanmax(rolling_slopes)) \
        if np.any(np.isfinite(rolling_slopes)) else np.nan

    diagnostics = {
        'log_likelihood': float(log_likelihood),
        'outlier_fraction': float(mask.mean()),
        'residual_lag1_autocorr': residual_autocorrelation,
        'median_dt': Quantity(median_time_step, TIME_UNIT),
        'max_rolling_slope': Quantity(max_rolling_slope, RATE_UNIT),
        'crosscheck_std': Quantity(float(crosscheck_std), RATE_UNIT),
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
        outlier_mask = mask,
        smooth = Quantity(smooth, AMOUNT_UNIT),
        smooth_std = Quantity(np.sqrt(smooth_variance), AMOUNT_UNIT),
        rate = Quantity(rate, RATE_UNIT),
        rate_std = Quantity(rate_std, RATE_UNIT),
        hyperparameters = {'lengthscale': Quantity(hyperparameters['lengthscale'], TIME_UNIT),
                         'signal_std': Quantity(hyperparameters['signal_std'], AMOUNT_UNIT),
                         'noise_std': Quantity(hyperparameters['noise_std'], AMOUNT_UNIT)},
        diagnostics = diagnostics,
    )
    result.flags = collect_quality_flags(result, time, lengthscale_bounds)

    return result


def plot_max_rate(result, time, values, axes=None):
    """
    Plot a two-panel diagnostic figure: data with smooth fit, and rate with confidence band.

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
    rate = result.rate.unit[RATE_UNIT]
    rate_std = result.rate_std.unit[RATE_UNIT]
    max_rate = result.max_rate.unit[RATE_UNIT]
    max_rate_std = result.max_rate_std.unit[RATE_UNIT]
    t_max_rate = result.t_max_rate.unit[TIME_UNIT]

    # Upper panel: raw data, masked artifacts, smoothed curve and the max-rate window
    data_axis.plot(time, values, '.', ms=1.5, color='0.6', label='data')
    if mask.any():
        data_axis.plot(time[mask], values[mask], 'x', ms=3, color='crimson',
                       label='masked artifacts')
    data_axis.plot(time, smooth, color='C0', lw=1.5, label='GP smooth')

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
    # one bubble artifact whose instantaneous slope is 75x the true rate
    rng = np.random.default_rng(0)
    time_seconds = np.arange(0.0, 8000.0, 1.0)
    signal = 0.02 * np.clip(time_seconds - 1000.0, 0.0, None)

    artifact = np.zeros_like(time_seconds)
    artifact[5000:5010] = np.linspace(0.0, 15.0, 10)
    artifact[5010:] = 15.0 * np.exp(-(time_seconds[5010:] - time_seconds[5010]) / 100.0)

    time = Quantity(time_seconds, 's')
    values = Quantity(signal + artifact + 0.2 * rng.standard_normal(len(time_seconds)), 'umol')

    result = extract_max_rate(time, values)

    print(f"max rate: {result.max_rate.unit['umol / s']:.4f} "
          f"± {result.max_rate_std.unit['umol / s']:.4f} umol/s "
          f"(true 0.0200) at t = {result.t_max_rate.unit['s']:.0f} s")
    print(f"in umol/h: {result.max_rate.unit['umol / h']:.2f}")
    print(f"cross-check: {result.max_rate_crosscheck.unit['umol / s']:.4f}, flags: {result.flags}")

    plot_max_rate(result, time, values)
    plt.show()


if __name__ == "__main__":
    test_function()
