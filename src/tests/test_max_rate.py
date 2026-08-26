"""Ground-truth tests for pyKES.utilities.max_rate.

Synthetic curves mimic photocatalytic H2/O2 measurements: induction
period, rise, plateau/decline, various noise levels, sensor artifacts
and thousands of points per series.
"""

import numpy as np
import pytest

from pyKES.utilities.max_rate import extract_max_rate


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
    result = extract_max_rate(t, y)
    assert result.max_rate == pytest.approx(true_max, rel=0.05)
    assert result.t_max_rate == pytest.approx(3000.0, abs=200.0)
    assert result.max_rate_instantaneous == pytest.approx(true_max, rel=0.10)


def test_induction_period_ramp(rng):
    t = np.arange(0.0, 8000.0, 1.0)
    y = induction_ramp(t, 2000.0, 0.02) + 0.2 * rng.standard_normal(len(t))
    result = extract_max_rate(t, y)
    assert result.max_rate == pytest.approx(0.02, rel=0.05)
    assert result.t_max_rate > 2000.0


def test_artifact_step_does_not_inflate_max_rate(rng):
    t = np.arange(0.0, 8000.0, 1.0)
    y = induction_ramp(t, 1000.0, 0.02) + 0.2 * rng.standard_normal(len(t))
    # Bubble-like transient: fast 15-unit rise over 10 s, then relaxation.
    # Its instantaneous slope (1.5/s) is 75x the true kinetic rate.
    artifact = np.zeros_like(t)
    artifact[5000:5010] = np.linspace(0.0, 15.0, 10)
    artifact[5010:] = 15.0 * np.exp(-(t[5010:] - t[5010]) / 100.0)
    result = extract_max_rate(t, y + artifact)
    assert result.max_rate == pytest.approx(0.02, rel=0.15)
    assert result.outlier_mask.any()


def test_high_noise_sigmoid(rng):
    t = np.arange(0.0, 10000.0, 1.0)
    signal = logistic_curve(t, 0.15, 0.002, 6000.0)
    y = signal + 0.008 * rng.standard_normal(len(t))  # SNR ~ 20 like EA-696
    true_max = 0.15 * 0.002 / 4.0
    result = extract_max_rate(t, y)
    assert result.max_rate == pytest.approx(true_max, rel=0.25)


def test_decline_phase_ignored(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    y = np.where(t < 4000.0, 0.01 * t, 40.0 - 0.05 * (t - 4000.0))
    y = y + 0.1 * rng.standard_normal(len(t))
    result = extract_max_rate(t, y)
    # Mild GP overshoot at the artificial infinite-curvature kink is tolerated.
    assert result.max_rate == pytest.approx(0.01, rel=0.10)
    assert result.t_max_rate < 4000.0


def test_irregular_sampling(rng):
    t = np.sort(rng.uniform(0.0, 6000.0, 3000))
    true_max = 50.0 * 0.003 / 4.0
    y = logistic_curve(t, 50.0, 0.003, 3000.0) + 0.3 * rng.standard_normal(len(t))
    result = extract_max_rate(t, y)
    assert result.max_rate == pytest.approx(true_max, rel=0.10)


def test_uncertainty_covers_truth(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    true_max = 100.0 * 0.004 / 4.0
    y = logistic_curve(t, 100.0, 0.004, 3000.0) + 1.0 * rng.standard_normal(len(t))
    result = extract_max_rate(t, y)
    assert abs(result.max_rate - true_max) < 4.0 * result.max_rate_std


def test_crosscheck_agrees_on_clean_data(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.004, 3000.0) + 0.5 * rng.standard_normal(len(t))
    result = extract_max_rate(t, y)
    assert result.max_rate_crosscheck == pytest.approx(result.max_rate, rel=0.1)
    assert 'estimator_disagreement' not in result.flags


def test_hyperparameter_reuse(rng):
    t = np.arange(0.0, 4000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.004, 2000.0) + 0.5 * rng.standard_normal(len(t))
    first = extract_max_rate(t, y)
    second = extract_max_rate(t, y, hyperparameters=first.hyperparameters)
    assert second.max_rate == pytest.approx(first.max_rate, rel=1e-6)


def test_window_parameter_controls_averaging(rng):
    t = np.arange(0.0, 6000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.01, 3000.0) + 0.5 * rng.standard_normal(len(t))
    narrow = extract_max_rate(t, y, window=60.0)
    wide = extract_max_rate(t, y, window=1200.0)
    # Averaging over a window much wider than the steep phase must lower the rate.
    assert wide.max_rate < narrow.max_rate
    assert narrow.max_rate == pytest.approx(100.0 * 0.01 / 4.0, rel=0.05)


def test_invalid_inputs_raise():
    t = np.arange(100.0)
    with pytest.raises(ValueError):
        extract_max_rate(t, np.ones_like(t))  # constant values
    with pytest.raises(ValueError):
        extract_max_rate(t[:10], np.arange(10.0))  # too short
    with pytest.raises(ValueError):
        extract_max_rate(t, np.arange(100.0), window=1000.0)  # window > span


def test_nan_values_are_dropped(rng):
    t = np.arange(0.0, 4000.0, 1.0)
    y = logistic_curve(t, 100.0, 0.004, 2000.0) + 0.5 * rng.standard_normal(len(t))
    y[::50] = np.nan
    result = extract_max_rate(t, y)
    assert result.max_rate == pytest.approx(100.0 * 0.004 / 4.0, rel=0.05)
    assert np.all(np.isfinite(result.values))
