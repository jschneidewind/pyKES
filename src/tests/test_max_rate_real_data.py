"""Tests for `pyKES.utilities.max_rate` against real measured traces.

The synthetic tests in `test_max_rate.py` check the module against curves whose
answer is known by construction. These check it against traces whose answer is
*not* known, but whose behaviour is: every acquisition format the group records
in, plus the two wells of a 110-well plate that sit either side of the nuisance
resolution floor.

The fixtures in `data/max_rate_real/` are plain two-column CSVs (`time_s`,
`value`), parsed once from the original instrument files:

- five PyroScience Workbench oxygen logs (~9 000 points at 1 Hz)
- two UniAmp hydrogen logger exports (~10 000-12 000 points at 1 Hz)
- two short hand-logged runs (~70 points at 3.3 s)
- two wells of a pyKES well plate (1 031 points at 3.4 s), taken from the
  ``processed_data`` group of ``260901_AE_851_to_AE-855.h5``

The logger and hand-logged traces measure a concentration and the well-plate
traces an amount. `extract_max_rate` requires the dimension substance either
way, so everything is wrapped as ``umol`` and a concentration result is read
per litre — which is what the upstream analysis script does too, and what makes
the reference rates below directly comparable to it.

Together the eleven fixtures exercise all three branches of the noise
characterization on real data: a correlated component resolved above the floor,
one clamped to it, and one folded into the white noise.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyKES.utilities.max_rate import (NUISANCE_MIN_LENGTHSCALE_STEPS, extract_max_rate)
from pyKES.utilities.unit_handler import Quantity

REAL_DATA_DIRECTORY = Path(__file__).parent / 'data' / 'max_rate_real'
AMOUNT_UNIT = 'umol'
RATE_UNIT = 'umol / s'

# The reference rates are what the current pipeline produces on these files.
# They are regression pins, not ground truth: they exist so that a change to the
# module has to be looked at rather than absorbed silently. The tolerance is
# loose enough for a different BLAS to land the Nelder-Mead fit somewhere
# slightly different, and tight enough that any real change in behaviour trips.
REFERENCE_TOLERANCE = 0.05


@dataclass(frozen=True)
class RealTrace:
    """One committed fixture and the maximum rate the pipeline extracts from it."""

    name: str
    source: str
    max_rate: float


REAL_TRACES = [
    RealTrace('2026-08-06_211209_EA-693-TROXROB-Ch2-2', 'txt', 0.0935131),
    RealTrace('2026-08-07_153007_MZ-442-Ch2-2', 'txt', 0.00729027),
    RealTrace('2026-08-07_153007_MZ-443-Ch2-2', 'txt', 0.0207478),
    RealTrace('2026-08-19_112822_VSA-122-Ch2-2', 'txt', 0.0569028),
    RealTrace('2026-08-19_144524_VSA-124-Ch2-2', 'txt', 0.0489906),
    RealTrace('EA-696-Logger-4', 'xlsx', 4.66284e-05),
    RealTrace('EA-698-Logger-2', 'xlsx', 0.00573741),
    RealTrace('MRG-059-V-4-1', 'csv', 0.274135),
    RealTrace('MRG-059-Z-1-3', 'csv', 1.05046),
    RealTrace('AE-855_B2', 'h5', 1.01165e-05),
    RealTrace('AE-855_C2', 'h5', 1.25934e-05),
]

TRACES_BY_NAME = {trace.name: trace for trace in REAL_TRACES}

# The two wells either side of the nuisance resolution floor. B2's correlated
# noise decorrelates just below it and C2's just above, on the same plate and
# the same instrument, which is what makes the pair a test of the boundary
# rather than of either well.
CLAMPED_WELL = 'AE-855_B2'
RESOLVED_WELL = 'AE-855_C2'

# The hand-logged runs are short enough that their variogram cannot support the
# four-parameter noise model at all, so their correlated component is folded.
FOLDED_TRACES = ['MRG-059-V-4-1', 'MRG-059-Z-1-3']


def load_real_trace(name):
    """
    Read one committed fixture as the Quantity pair `extract_max_rate` takes.

    Parameters
    ----------
    name : str
        Fixture name without the .csv suffix.

    Returns
    -------
    tuple of Quantity
        Sample times in seconds and values in umol.
    """
    table = pd.read_csv(REAL_DATA_DIRECTORY / f'{name}.csv')

    return (Quantity(table['time_s'].to_numpy(float), 's'),
            Quantity(table['value'].to_numpy(float), AMOUNT_UNIT))


@lru_cache(maxsize=None)
def analyse_real_trace(name):
    """
    Extract the maximum rate of one fixture, once per test session.

    Several tests assert on the same trace and the longest fixtures take a few
    seconds each, so the result is cached rather than recomputed.

    Parameters
    ----------
    name : str
        Fixture name without the .csv suffix.

    Returns
    -------
    MaxRateResult
        The finished extraction.
    """
    return extract_max_rate(*load_real_trace(name))


@pytest.mark.parametrize('name', sorted(TRACES_BY_NAME), ids=str)
def test_real_trace_is_modelled_by_the_two_components(name):
    """On every real format, kinetic + nuisance must reproduce the trace.

    This is the same property as `test_components_add_up_to_the_data`, asked of
    measured data: what is left over after both components have been subtracted
    has to be the white noise the fit claims, neither an unexplained remainder
    nor a curve that has interpolated the scatter away.
    """
    time, values = load_real_trace(name)
    result = analyse_real_trace(name)

    residuals = values.unit[AMOUNT_UNIT] - result.smooth.unit[AMOUNT_UNIT] \
        - result.nuisance.unit[AMOUNT_UNIT]
    assert np.std(residuals) == pytest.approx(
        result.hyperparameters['noise_std'].unit[AMOUNT_UNIT], rel=0.6)

    assert np.isfinite(result.max_rate.unit[RATE_UNIT])
    assert len(result.smooth.unit[AMOUNT_UNIT]) == len(time.unit['s'])
    assert result.diagnostics['outlier_fraction'] < 0.05


@pytest.mark.parametrize('name', sorted(TRACES_BY_NAME), ids=str)
def test_real_trace_max_rate_matches_reference(name):
    """Regression pin: a change in behaviour on real data has to be deliberate."""
    result = analyse_real_trace(name)

    assert result.max_rate.unit[RATE_UNIT] == pytest.approx(
        TRACES_BY_NAME[name].max_rate, rel=REFERENCE_TOLERANCE)


@pytest.mark.parametrize('name', sorted(TRACES_BY_NAME), ids=str)
def test_real_trace_agrees_with_the_smoothing_free_crosscheck(name):
    """The rolling regression of the raw data is an independent second opinion.

    None of these traces may set the two estimators far enough apart for the
    module to flag it, which is a check on the smoothing that no reference
    value can give: the cross-check never sees the Gaussian process at all.
    """
    assert 'estimator_disagreement' not in analyse_real_trace(name).flags


def test_well_below_the_nuisance_floor_keeps_its_correlated_noise():
    """The well the resolution cliff used to break (AE-855_B2).

    Its correlated noise decorrelates in 2.99 sampling intervals, a shade under
    the three-interval floor. Discarding the component there re-labelled its
    whole amplitude as white noise and left the kinetic component as the only
    thing able to explain wiggles that are in fact correlated -- so the kinetic
    length scale collapsed onto them, the rate curve oscillated instead of
    decaying, and a noise crest became the reported maximum.
    """
    time, _ = load_real_trace(CLAMPED_WELL)
    result = analyse_real_trace(CLAMPED_WELL)
    median_time_step = float(np.median(np.diff(time.unit['s'])))

    # The component is pinned to the floor, not deleted ...
    assert result.hyperparameters['nuisance_lengthscale'].unit['s'] == pytest.approx(
        NUISANCE_MIN_LENGTHSCALE_STEPS * median_time_step)
    # ... and it carries a real amplitude rather than the placeholder left
    # behind when a nuisance is switched off.
    assert result.hyperparameters['nuisance_std'].unit[AMOUNT_UNIT] \
        > result.hyperparameters['noise_std'].unit[AMOUNT_UNIT]

    # The kinetics stay slow: the length scale sits far above the floor that
    # would let them track the noise, instead of pinned just above it.
    assert result.hyperparameters['lengthscale'].unit['s'] \
        > 10.0 * result.diagnostics['lengthscale_lower_bound'].unit['s']

    # A saturating curve's rate decays; it does not turn back on itself dozens
    # of times. The discarded-nuisance fit had 94 turning points, this has ~7.
    rate = result.rate.unit[RATE_UNIT]
    assert int((np.diff(np.sign(np.diff(rate))) != 0).sum()) < 20

    # The failure used to be silent, because switching the nuisance off also
    # set the correlation time to the one value that suppresses this flag.
    assert 'strong_correlated_noise' in result.flags


def test_well_above_the_nuisance_floor_is_untouched():
    """The neighbouring well (AE-855_C2) must not be perturbed by the clamp.

    C2 comes off the same plate with the same instrument noise, but its
    correlation time lands just above the floor. Whatever happens below the
    floor has to leave the wells above it exactly where they were -- that is
    the half of the boundary a fix is most likely to break.
    """
    time, _ = load_real_trace(RESOLVED_WELL)
    result = analyse_real_trace(RESOLVED_WELL)
    median_time_step = float(np.median(np.diff(time.unit['s'])))

    assert result.hyperparameters['nuisance_lengthscale'].unit['s'] \
        > NUISANCE_MIN_LENGTHSCALE_STEPS * median_time_step
    assert result.hyperparameters['nuisance_std'].unit[AMOUNT_UNIT] \
        > result.hyperparameters['noise_std'].unit[AMOUNT_UNIT]
    assert result.hyperparameters['lengthscale'].unit['s'] \
        > 10.0 * result.diagnostics['lengthscale_lower_bound'].unit['s']

    rate = result.rate.unit[RATE_UNIT]
    assert int((np.diff(np.sign(np.diff(rate))) != 0).sum()) < 20


@pytest.mark.parametrize('name', FOLDED_TRACES, ids=str)
def test_short_hand_logged_trace_folds_its_nuisance(name):
    """A run too short for the noise model must still switch the nuisance off.

    These are the real counterpart of `test_unresolvable_nuisance_is_treated_as
    _white_noise`: ~70 points give three variogram lags for a four-parameter
    model, so there is nothing to separate and the correlated component is
    folded into the white noise rather than clamped to the floor.
    """
    time, _ = load_real_trace(name)
    result = analyse_real_trace(name)

    assert result.hyperparameters['nuisance_lengthscale'].unit['s'] == pytest.approx(
        float(np.median(np.diff(time.unit['s']))))
    assert result.hyperparameters['nuisance_std'].unit[AMOUNT_UNIT] \
        < 0.01 * result.hyperparameters['noise_std'].unit[AMOUNT_UNIT]
