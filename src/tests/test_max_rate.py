"""Ground-truth tests for pyKES.utilities.max_rate.

Synthetic curves mimic photocatalytic H2/O2 measurements: induction
period, rise, plateau/decline, various noise levels, sensor artifacts
and thousands of points per series.

Series are built as plain arrays in seconds and moles, then wrapped in
`Quantity` objects by `time_quantity` / `amount_quantity`, so the numeric
ground truth is directly comparable to the magnitudes read back from the
result in `'mol / s'`.
"""

import numpy as np
import pytest

from pyKES.utilities.max_rate import (NUISANCE_MIN_LENGTHSCALE_STEPS, WINDOW_MAX_SPAN_FRACTION,
                                      WINDOW_MEDIAN_STEPS, estimate_noise_structure,
                                      extract_max_rate, matern32_state_space_matrices,
                                      resolve_window)
from pyKES.utilities.unit_handler import Quantity

RATE_UNIT = 'mol / s'


def time_quantity(t):
    """Wrap an array of seconds as a Quantity."""
    return Quantity(t, 's')


def amount_quantity(y):
    """Wrap an array of moles as a Quantity."""
    return Quantity(y, 'mol')


def saturating_curve(t, amplitude, time_constant):
    """Exponential approach to a plateau, like the O2 traces of a photocatalysis run."""
    return amplitude * (1.0 - np.exp(-t / time_constant))


def logistic_curve(t, amplitude, rate_constant, t_mid):
    """Logistic curve whose true maximum rate is amplitude * rate_constant / 4."""
    return amplitude / (1.0 + np.exp(-rate_constant * (t - t_mid)))


def induction_ramp(t, t_onset, slope):
    """Zero until t_onset, then a linear rise with the given slope."""
    return slope * np.clip(t - t_onset, 0.0, None)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_logistic_max_rate_recovered(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    true_max = 100.0 * 0.004 / 4.0
    y = logistic_curve(t, 100.0, 0.004, 3000.0) + 0.5 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(true_max, rel=0.05)
    assert result.t_max_rate.unit['s'] == pytest.approx(3000.0, abs=200.0)
    assert result.max_rate_instantaneous.unit[RATE_UNIT] == pytest.approx(true_max, rel=0.10)


def test_induction_period_ramp(rng):
    t = np.arange(0.0, 8000.0, 1.0)
    y = induction_ramp(t, 2000.0, 0.02) + 0.2 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(0.02, rel=0.05)
    assert result.t_max_rate.unit['s'] > 2000.0


def test_artifact_step_does_not_inflate_max_rate(rng):
    t = np.arange(0.0, 8000.0, 1.0)
    y = induction_ramp(t, 1000.0, 0.02) + 0.2 * rng.standard_normal(len(t))
    # Bubble-like transient: fast 15-unit rise over 10 s, then relaxation.
    # Its instantaneous slope (1.5/s) is 75x the true kinetic rate.
    artifact = np.zeros_like(t)
    artifact[5000:5010] = np.linspace(0.0, 15.0, 10)
    artifact[5010:] = 15.0 * np.exp(-(t[5010:] - t[5010]) / 100.0)
    result = extract_max_rate(time_quantity(t), amount_quantity(y + artifact))
    # The last stretch of the relaxation is indistinguishable from the noise,
    # so a little of the transient always survives rejection; without it the
    # windowed rate would come out near 0.1, five times the truth.
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(0.02, rel=0.25)
    assert result.outlier_mask[5010:5200].all()
    assert not result.outlier_mask[:4500].any()


def test_high_noise_sigmoid(rng):
    t = np.arange(0.0, 10000.0, 1.0)
    signal = logistic_curve(t, 0.15, 0.002, 6000.0)
    y = signal + 0.008 * rng.standard_normal(len(t))  # SNR ~ 20 like EA-696
    true_max = 0.15 * 0.002 / 4.0
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(true_max, rel=0.25)


def test_decline_phase_ignored(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    y = np.where(t < 4000.0, 0.01 * t, 40.0 - 0.05 * (t - 4000.0))
    y = y + 0.1 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    # The kink has infinite curvature, which no smoothness prior can represent:
    # the fit rings just before it, overshooting the true rate by ~20 % over one
    # window. Everywhere else the recovered rate is 0.01 to within a percent.
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(0.01, rel=0.25)
    assert result.t_max_rate.unit['s'] < 4000.0


def test_irregular_sampling(rng):
    t = np.sort(rng.uniform(0.0, 6000.0, 3000))
    true_max = 50.0 * 0.003 / 4.0
    y = logistic_curve(t, 50.0, 0.003, 3000.0) + 0.3 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(true_max, rel=0.10)


def test_uncertainty_covers_truth(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    true_max = 100.0 * 0.004 / 4.0
    y = logistic_curve(t, 100.0, 0.004, 3000.0) + 1.0 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert abs(result.max_rate.unit[RATE_UNIT] - true_max) \
        < 4.0 * result.max_rate_std.unit[RATE_UNIT]


def test_crosscheck_agrees_on_clean_data(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.004, 3000.0) + 0.5 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.max_rate_crosscheck.unit[RATE_UNIT] \
        == pytest.approx(result.max_rate.unit[RATE_UNIT], rel=0.1)
    assert 'estimator_disagreement' not in result.flags


def test_hyperparameter_reuse(rng):
    t = np.arange(0.0, 4000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.004, 2000.0) + 0.5 * rng.standard_normal(len(t))
    first = extract_max_rate(time_quantity(t), amount_quantity(y))
    second = extract_max_rate(time_quantity(t), amount_quantity(y),
                              hyperparameters=first.hyperparameters)
    assert second.max_rate.unit[RATE_UNIT] \
        == pytest.approx(first.max_rate.unit[RATE_UNIT], rel=1e-6)


def test_window_parameter_controls_averaging(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.01, 3000.0) + 0.5 * rng.standard_normal(len(t))
    narrow = extract_max_rate(time_quantity(t), amount_quantity(y), window=Quantity(60.0, 's'))
    wide = extract_max_rate(time_quantity(t), amount_quantity(y), window=Quantity(20.0, 'minute'))
    # Averaging over a window much wider than the steep phase must lower the rate.
    assert wide.max_rate.unit[RATE_UNIT] < narrow.max_rate.unit[RATE_UNIT]
    assert wide.window.unit['s'] == pytest.approx(1200.0)
    assert narrow.max_rate.unit[RATE_UNIT] == pytest.approx(100.0 * 0.01 / 4.0, rel=0.05)


def test_result_is_unit_independent(rng):
    """The same physical series in umol and minutes must give the same rate."""
    t = np.arange(0.0, 6000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.004, 3000.0) + 0.5 * rng.standard_normal(len(t))
    in_base = extract_max_rate(time_quantity(t), amount_quantity(y))
    rescaled = extract_max_rate(Quantity(t / 60.0, 'minute'), Quantity(y * 1e6, 'umol'))
    assert rescaled.max_rate.unit[RATE_UNIT] \
        == pytest.approx(in_base.max_rate.unit[RATE_UNIT], rel=1e-6)
    assert rescaled.max_rate.unit['umol / minute'] \
        == pytest.approx(in_base.max_rate.unit[RATE_UNIT] * 6e7, rel=1e-6)


def test_input_series_is_not_stored(rng):
    """The result must not duplicate the input arrays (dataset storage cost)."""
    t = np.arange(0.0, 4000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.004, 2000.0) + 0.5 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert not hasattr(result, 'time')
    assert not hasattr(result, 'values')


def test_invalid_inputs_raise():
    t = np.arange(100.0)
    with pytest.raises(ValueError):
        extract_max_rate(time_quantity(t), amount_quantity(np.ones_like(t)))  # constant values
    with pytest.raises(ValueError):
        extract_max_rate(time_quantity(t[:10]), amount_quantity(np.arange(10.0)))  # too short
    with pytest.raises(ValueError):
        extract_max_rate(time_quantity(t), amount_quantity(np.arange(100.0)),
                         window=Quantity(1000.0, 's'))  # window > span


def test_non_quantity_inputs_raise():
    t = np.arange(0.0, 4000.0, 1.0)
    y = induction_ramp(t, 1000.0, 0.02)
    with pytest.raises(TypeError):
        extract_max_rate(t, amount_quantity(y))
    with pytest.raises(TypeError):
        extract_max_rate(time_quantity(t), y)


def test_wrong_dimension_inputs_raise():
    t = np.arange(0.0, 4000.0, 1.0)
    y = induction_ramp(t, 1000.0, 0.02)
    with pytest.raises(ValueError):
        extract_max_rate(Quantity(t, 'm'), amount_quantity(y))  # length, not time
    with pytest.raises(ValueError):
        extract_max_rate(time_quantity(t), Quantity(y, 'g'))  # mass, not substance


def test_nan_values_are_dropped(rng):
    t = np.arange(0.0, 4000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.004, 2000.0) + 0.5 * rng.standard_normal(len(t))
    y[::50] = np.nan
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(100.0 * 0.004 / 4.0, rel=0.05)
    assert np.all(np.isfinite(result.smooth.unit['mol']))


def test_low_frequency_noise_does_not_inflate_rate(rng):
    """A baseline wave steeper than the kinetics must not be read as rate.

    This is the failure that motivated the two-component model: a single
    length scale short enough to track the wave makes its slope, not the
    reaction, the largest rate in the series.
    """
    t = np.arange(0.0, 8000.0, 1.0)
    true_max = 0.02
    y = induction_ramp(t, 1000.0, true_max)
    # Peak slope of the wave is 2*pi*3/400 = 0.047, more than twice the truth.
    y = y + 3.0 * np.sin(2.0 * np.pi * t / 400.0) + 0.2 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(true_max, rel=0.15)
    assert 'strong_correlated_noise' in result.flags


def test_low_frequency_noise_is_separated_from_the_curve(rng):
    """The fitted nuisance component must carry the wave, not the kinetic one."""
    t = np.arange(0.0, 8000.0, 1.0)
    wave = 3.0 * np.sin(2.0 * np.pi * t / 400.0)
    y = induction_ramp(t, 1000.0, 0.02) + wave + 0.2 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    nuisance = result.nuisance.unit['mol']
    assert np.std(nuisance) == pytest.approx(np.std(wave), rel=0.35)
    # What is left of the wave in the kinetic component is a small fraction of it.
    assert np.std(result.smooth.unit['mol'] - induction_ramp(t, 1000.0, 0.02)) < 0.5 * np.std(wave)


def ornstein_uhlenbeck(rng, length, correlation_samples):
    """Stationary, entirely aperiodic drift with a set correlation time, unit variance."""
    decay = np.exp(-1.0 / correlation_samples)
    innovations = np.sqrt(1.0 - decay ** 2) * rng.standard_normal(length)
    drift = np.empty(length)
    drift[0] = rng.standard_normal()
    for step in range(1, length):
        drift[step] = decay * drift[step - 1] + innovations[step]
    return drift


def test_aperiodic_low_frequency_noise(rng):
    """Robustness must not depend on the disturbance being periodic.

    The nuisance component is a plain short-correlation-time Matern process,
    so a drift with no periodicity at all must be handled just like the sine.
    """
    t = np.arange(0.0, 8000.0, 1.0)
    true_max = 0.02
    drift = 3.0 * ornstein_uhlenbeck(rng, len(t), 100.0)
    y = induction_ramp(t, 1000.0, true_max) + drift + 0.2 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    # An OU drift is nowhere differentiable, so unlike a sine it keeps some
    # slope at every scale and can never be separated out completely; what
    # matters is that the estimate stays close to the truth instead of being
    # set by the drift, as the raw rolling slope is.
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(true_max, rel=0.45)
    assert result.max_rate.unit[RATE_UNIT] \
        < 0.5 * result.diagnostics['max_rolling_slope'].unit[RATE_UNIT]


def test_sharp_onset_survives_artifact_rejection(rng):
    """A one-off step is kinetics, not an artifact, and must not be downweighted."""
    t = np.arange(0.0, 9000.0, 1.0)
    y = induction_ramp(t, 1000.0, 0.05) + 0.3 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    onset = (t > 900.0) & (t < 1400.0)
    assert not result.outlier_mask[onset].any()
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(0.05, rel=0.10)


def test_blank_series_reports_no_significant_rate(rng):
    """A control well that only loses signal must not report a noise-driven rate."""
    t = np.arange(0.0, 3000.0, 3.4)
    drift = -3e-4 * (1.0 - np.exp(-t / 1500.0))
    wave = 1e-5 * np.sin(2.0 * np.pi * t / 150.0)
    y = drift + wave + 3e-6 * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.max_rate.unit[RATE_UNIT] < 1e-7
    assert 'max_rate_not_significant' in result.flags


def test_saturating_curve_initial_rate(rng):
    """The maximum of a saturating curve sits at its start, where masking used to bite."""
    t = np.arange(0.0, 3000.0, 3.4)
    amplitude, time_constant = 0.01, 400.0
    y = saturating_curve(t, amplitude, time_constant) + 1e-4 * rng.standard_normal(len(t))
    window = 60.0
    # Mean of the true derivative over the first computable window.
    true_windowed = amplitude * (1.0 - np.exp(-window / time_constant)) / window
    result = extract_max_rate(time_quantity(t), amount_quantity(y),
                              window=Quantity(window, 's'))
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(true_windowed, rel=0.10)
    assert result.t_max_rate.unit['s'] < 200.0


def test_components_add_up_to_the_data(rng):
    """smooth + nuisance is the model of the trace; the residual is white noise."""
    t = np.arange(0.0, 4000.0, 1.0)
    noise_std = 0.5
    y = logistic_curve(t, 100.0, 0.004, 2000.0) + noise_std * rng.standard_normal(len(t))
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    residuals = y - result.smooth.unit['mol'] - result.nuisance.unit['mol']
    assert np.std(residuals) == pytest.approx(noise_std, rel=0.2)


# --- Correlated noise at the edge of what the sampling resolves ---------------
# A nuisance decorrelating in two or three samples is still correlated noise and
# still has to be carried by the nuisance component. Discarding it instead --
# which the resolution guard used to do the moment the measured correlation time
# slipped below three sampling intervals -- leaves the kinetic component as the
# only thing left to explain the wiggles with, and it obliges.

FLOOR_AMPLITUDE = 100.0
FLOOR_TIME_CONSTANT = 400.0
FLOOR_NUISANCE = 3.0
FLOOR_WHITE = 0.4
FLOOR_TIME_STEP = 3.4


def matern32_draw(rng, t, lengthscale, std):
    """Exact draw from the Matern-3/2 process the nuisance component models."""
    transitions, process_noises, stationary = matern32_state_space_matrices(
        np.diff(t), lengthscale, std ** 2)
    state = np.linalg.cholesky(stationary) @ rng.standard_normal(2)

    path = np.empty(len(t))
    path[0] = state[0]
    for step in range(len(t) - 1):
        state = transitions[step] @ state \
            + np.linalg.cholesky(process_noises[step]) @ rng.standard_normal(2)
        path[step + 1] = state[0]

    return path


def nuisance_floor_series(rng, correlation_steps):
    """A densely sampled saturating curve plus a nuisance of the given correlation time."""
    t = np.arange(0.0, 3500.0, FLOOR_TIME_STEP)
    y = saturating_curve(t, FLOOR_AMPLITUDE, FLOOR_TIME_CONSTANT) \
        + matern32_draw(rng, t, correlation_steps * FLOOR_TIME_STEP, FLOOR_NUISANCE) \
        + FLOOR_WHITE * rng.standard_normal(len(t))
    return t, y


def test_fast_nuisance_is_clamped_not_discarded(rng):
    """A resolvable nuisance below the correlation-time floor is pinned, not deleted."""
    t, y = nuisance_floor_series(rng, correlation_steps=2.0)
    structure = estimate_noise_structure(t, y)

    assert structure['correlated_lengthscale'] == pytest.approx(
        NUISANCE_MIN_LENGTHSCALE_STEPS * FLOOR_TIME_STEP)
    # The component keeps its own amplitude instead of being folded into the
    # white noise, which would leave white_std of the order of FLOOR_NUISANCE.
    assert structure['correlated_std'] == pytest.approx(FLOOR_NUISANCE, rel=0.5)
    assert structure['white_std'] < 0.5 * FLOOR_NUISANCE


def test_max_rate_is_continuous_across_the_nuisance_floor():
    """Crossing the correlation-time floor may not move the answer.

    The same series either side of the floor has the same kinetics, so the two
    maxima must agree. Discarding the sub-floor nuisance instead broke this by
    60 %, because the kinetic length scale then collapsed onto the wiggles.
    """
    rates = []
    for correlation_steps in (2.5, 4.0):
        t, y = nuisance_floor_series(np.random.default_rng(11), correlation_steps)
        result = extract_max_rate(time_quantity(t), amount_quantity(y))
        rates.append(result.max_rate.unit[RATE_UNIT])

    assert rates[0] == pytest.approx(rates[1], rel=0.10)


# --- Short, coarsely sampled series ------------------------------------------
# Roughly 70 points over four minutes, the regime of a hand-logged run: far too
# few samples for the window and the noise model to be set by the sampling.

SHORT_AMPLITUDE = 100.0
SHORT_TIME_CONSTANT = 60.0
SHORT_NOISE = 0.25


def short_series(rng):
    """A ~75-point saturating curve over 250 s, like a short hand-logged run."""
    t = np.arange(0.0, 250.0, 3.3)
    y = saturating_curve(t, SHORT_AMPLITUDE, SHORT_TIME_CONSTANT) \
        + SHORT_NOISE * rng.standard_normal(len(t))
    return t, y


def test_short_series_window_is_duration_limited(rng):
    """The window may never grow to a third of the experiment."""
    t, y = short_series(rng)
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert result.window.unit['s'] <= 0.1 * (t[-1] - t[0]) + 1e-9
    assert 'window_duration_limited' in result.flags


def test_short_series_recovers_initial_rate(rng):
    """The maximum of a fast saturating curve must not be averaged away.

    The sampling-based window floor alone would be 25 * 3.3 = 82 s here, a third
    of the run, which averages the initial maximum together with the plateau and
    costs about a quarter of the rate.
    """
    t, y = short_series(rng)
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    window = result.window.unit['s']
    # Mean of the true derivative over the first computable window.
    true_windowed = SHORT_AMPLITUDE * (1.0 - np.exp(-window / SHORT_TIME_CONSTANT)) / window
    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(true_windowed, rel=0.10)


def test_short_series_has_no_spurious_outliers(rng):
    """A clean short series must not have part of itself rejected as artifacts.

    An unresolvable nuisance would otherwise interpolate the measurement noise,
    collapse the residuals and take a sixth of the series with it.
    """
    t, y = short_series(rng)
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert not result.outlier_mask.any()
    assert 'many_outliers_masked' not in result.flags


def test_short_series_crosscheck_is_finite(rng):
    """The smoothing-free second opinion must survive a duration-limited window."""
    t, y = short_series(rng)
    result = extract_max_rate(time_quantity(t), amount_quantity(y))
    assert np.isfinite(result.max_rate_crosscheck.unit[RATE_UNIT])
    assert result.max_rate_crosscheck.unit[RATE_UNIT] \
        == pytest.approx(result.max_rate.unit[RATE_UNIT], rel=0.15)


def test_unresolvable_nuisance_is_treated_as_white_noise(rng):
    """Too few variogram lags must not turn white noise into a correlated component."""
    t, y = short_series(rng)
    structure = estimate_noise_structure(t, y)
    assert structure['correlated_lengthscale'] == pytest.approx(np.median(np.diff(t)))
    assert structure['white_std'] == pytest.approx(SHORT_NOISE, rel=0.6)
    assert structure['correlated_std'] < 0.01 * structure['white_std']


def test_window_rule_is_continuous_and_unchanged_when_dense(rng):
    """The sample floor and the duration cap must cross without a step."""
    # Densely sampled: the cap does not bind and the old rule is untouched.
    assert resolve_window(None, 1.0, 9000.0) == pytest.approx(180.0)
    # Coarsely sampled: the cap sets the window.
    assert resolve_window(None, 3.3, 250.0) == pytest.approx(25.0)
    # They agree exactly at the crossover, so nothing jumps.
    crossover_steps = WINDOW_MEDIAN_STEPS / WINDOW_MAX_SPAN_FRACTION
    duration = crossover_steps * 2.0
    assert resolve_window(None, 2.0, duration) \
        == pytest.approx(WINDOW_MEDIAN_STEPS * 2.0) \
        == pytest.approx(WINDOW_MAX_SPAN_FRACTION * duration)
