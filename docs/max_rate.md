# Maximum-rate extraction (`pyKES.utilities.max_rate`)

`extract_max_rate` takes a measured time series — for example dissolved H2 or O2
concentration recorded every second during a photocatalytic water-splitting
experiment — and answers one question: **what was the highest reaction rate, and
how certain are we about it?**

```python
from pyKES.utilities.max_rate import extract_max_rate, plot_max_rate
from pyKES.utilities.unit_handler import Quantity

time = Quantity(time_seconds, 's')          # any unit of dimension time
amount = Quantity(evolved_h2_umol, 'umol')  # any unit of dimension substance

result = extract_max_rate(time, amount)

print(result.max_rate)                      # Quantity, dimension substance / time
print(result.max_rate.unit['umol / h'])     # read it in whatever unit you want
print(result.max_rate_std)                  # its standard deviation
print(result.flags)                         # empty list = nothing suspicious

plot_max_rate(result, time, amount)         # two-panel diagnostic figure
```

## 0. Units

Inputs and outputs are [`Quantity`](../src/pyKES/utilities/unit_handler/quantity.py)
objects, so a result carries its own units and nothing depends on what the
caller happened to record in:

- **`time`** must have the dimension *time*, **`values`** the dimension
  *substance*. Anything else (a bare NumPy array, a mass, a length) raises
  immediately — there is no "assume seconds" fallback.
- Optional time-valued arguments (`window`, `lengthscale_bounds`,
  the reused `hyperparameters`) are Quantities too.
- Internally everything is reduced to plain floats in **moles and seconds**
  (`AMOUNT_UNIT`, `TIME_UNIT`, `RATE_UNIT` at the top of the module), which is
  what all the formulas below operate on.
- Every physical field of the result is wrapped back up as a `Quantity`, rates
  in `mol / s`. Use `.unit['<unit>']` to convert; the lookup is lazy and cached.

The result deliberately does **not** store the input `time` and `values`
arrays. Results are meant to be saved next to the dataset they were computed
from, and keeping a second copy of every series there would double the file for
no information gain — so `plot_max_rate` takes the inputs again. The per-sample
arrays it does keep (`smooth`, `rate`, their uncertainties, `outlier_mask`)
exist nowhere else.

It is designed for real sensor data: thousands of points per curve, different
noise levels, induction periods, drifting baselines, and sensor artifacts such
as gas bubbles. It works on any monotone-ish kinetic trace, not just water
splitting.

---

## 1. Why this is harder than it looks

The rate is the slope (the derivative) of the measured curve. The naive
approach — subtracting neighbouring points and dividing by the time step — fails
badly on real data, for four reasons:

**Noise is amplified by differentiation.** If each point has ±0.1 µmol/L of
noise and points are 1 s apart, point-to-point slopes are uncertain by about
±0.14 µmol/L/s — often larger than the entire true rate. Any derivative
estimate must therefore smooth the data first, and the amount of smoothing must
be chosen carefully: too little and the derivative is noise, too much and sharp
features (like the end of an induction period) are flattened.

**Taking a maximum of something noisy is biased.** Even after smoothing, the
estimated rate curve wiggles randomly around the true one. Picking its highest
point preferentially picks a spot where the wiggle happened to point upward, so
the naive "maximum of the derivative" systematically *overestimates* the true
maximum (statisticians call this the winner's curse). More smoothing and
averaging over a window both shrink this bias.

**Sensor artifacts look like extreme rates.** When a gas bubble passes a
sensor, the reading can jump by several µmol/L within seconds and then relax
back over minutes. In the example data used to develop this module, one such
transient had an instantaneous slope **10 times larger** than the true maximum
rate. An extractor that does not recognize artifacts will happily report the
bubble as the maximum rate.

**Curve shapes vary.** Some experiments show a long induction period followed
by a burst, some rise almost linearly, some level off or decline after the lamp
is switched off. A method that fits one fixed equation (a straight line, a
logistic curve, …) breaks whenever the data have a different shape. This module
is deliberately *model-free*: it never assumes a functional form for the
kinetics.

## 2. The pipeline at a glance

```
raw series
    │
    ▼
1. validation          drop NaNs, sort by time, remove duplicate time stamps
    │
    ▼
2. artifact masking    find bubble jumps/spikes, mask the full transient
    │
    ▼
3. GP smoothing        fit a Matern-5/2 Gaussian process (exact, O(n) Kalman
    │                  smoother) -> smooth curve + rate + uncertainties
    ▼
4. rate extraction     max_rate = largest window-averaged rate (robust headline)
    │                  max_rate_instantaneous = peak of the rate curve (secondary)
    ▼
5. cross-check         rolling linear regression of the raw data + quality flags
```

Each stage is described below.

## 3. Artifact masking

### Detecting jumps

At one sample per second, chemistry moves the signal by far less than the noise
level between two neighbouring samples — but a bubble moves it by many noise
standard deviations. The detector exploits exactly this:

1. Compute the sample-to-sample differences of the signal.
2. Estimate the *local kinetic trend* of those differences with a rolling
   median (medians ignore outliers, so the trend is not corrupted by the very
   artifacts we are looking for).
3. Compute a robust z-score for every increment: how many (robust) standard
   deviations it deviates from what the local trend predicts.
4. Flag increments whose z-score exceeds `outlier_threshold` (default 6).

This test is run at two lags — over 1 sample and over 5 samples — so that a
jump spread over a few samples, where each individual step stays below
threshold, is still caught by the 5-sample increment.

Robust statistics are used throughout: the noise level is estimated from the
median absolute deviation (scaled by the constant `MAD_TO_STD` = 1.4826, which
converts it to a standard deviation for Gaussian noise), so a handful of
extreme values cannot inflate the noise estimate and hide the artifacts.

### Growing the mask over whole transients

A bubble does not end when the jump ends: the reading typically relaxes back to
the true curve over minutes. That relaxation tail is locally smooth and passes
any jump test, but leaving it in the data would create a phantom rate.

For every masked run that begins with a *gross* jump (z-score at least twice
the threshold — genuine bubbles score 25–75, so this cleanly separates them
from marginal statistical flags), the mask is grown forward:

1. Fit a straight line to the ~200 clean samples before the jump ("where would
   the curve be if the bubble had not happened?").
2. Keep masking samples while the signal stays more than 4 noise standard
   deviations away from that line.
3. Stop when 5 consecutive samples are back on the line — the transient is over.

Crucially, if the signal **never** returns to the line, the growth is reverted:
a deviation that persists is not a bubble but a genuine change in the kinetics
(the end of an induction period, the lamp switching off). This "must return to
trend" rule is what lets the module mask a 300-second bubble transient without
ever masking a real kinetic onset, and marginal flags near such an onset are
unmasked along with it, since they are collateral of the same sharp feature.

Masked samples are not deleted — they are simply ignored during fitting, and
the smoother bridges the gap.

## 4. Smoothing with a Gaussian process

### The idea, without mathematics

The module assumes the truth is *some* smooth curve and asks: given the noisy
data, what is the most plausible smooth curve through them, and how uncertain
is it? This is Gaussian-process (GP) regression. Unlike fitting a formula, a GP
does not prescribe a shape — only a degree of smoothness — so it adapts to
induction periods, bursts, plateaus and declines alike. And because the
derivative of a smooth random curve is itself well defined, the GP directly
yields the **rate and an error bar for the rate at every time point**.

Its behaviour is controlled by three numbers (hyperparameters), all learned
from the data automatically by maximizing the statistical likelihood:

| Hyperparameter | Meaning |
|---|---|
| `lengthscale` | Over what time span the curve can bend. Short = wiggly, long = stiff. |
| `signal_std` | How far the curve wanders overall. |
| `noise_std` | How noisy each individual measurement is. |

The kernel is Matern-5/2, the standard choice for derivative estimation: it is
smooth enough to have a well-defined rate, but flexible enough to follow the
sharp acceleration after an induction period (an infinitely smooth kernel would
round such corners off).

A lower bound on the lengthscale (default: 20 median time steps) prevents a
known failure mode in which the optimizer chases slow, correlated sensor drift
by making the curve absurdly wiggly, mistaking noise for signal.

### Why it is fast: the Kalman trick

Textbook GP regression requires building and inverting an n-by-n matrix —
for n = 10 000 points that is 10^8 numbers and far too slow to do repeatedly.
This module instead uses the state-space formulation: a Matern-5/2 process is
mathematically *identical* to a small physical system whose state is the trio
(curve value, its first derivative, its second derivative) evolving through
time. That system can be estimated with a Kalman filter — a forward sweep that
updates the state estimate one sample at a time — followed by a
Rauch–Tung–Striebel smoother, a backward sweep that feeds information from
later samples back to earlier ones.

The result is **exact** GP regression (verified in development against the
textbook matrix computation to machine precision) in time proportional to n:
about a second for 10 000 points. Because the rate is literally a component of
the state, the derivative and its uncertainty come out of the smoother for
free. Hyperparameters are fitted on a decimated subset (default ≤ 1500 points)
for speed; the final smoothing pass always uses every point.

## 5. What "maximum rate" means here

The module reports two numbers with different characters:

**`max_rate` (the headline number)** is the largest rate *sustained over a time
window* — precisely, the largest value of
(smooth(t + w/2) − smooth(t − w/2)) / w over all window positions t. The
default window is 2 % of the experiment duration (at least 25 time steps).
Averaging over a window is what buys robustness: an artifact that survives
masking can only perturb this number by its amplitude divided by the window
length, and the winner's-curse bias shrinks in the same way. Windows that
overlap masked regions by more than half are excluded — inside a masked gap the
smooth curve is an interpolated bridge, and its slope is not evidence.

**`max_rate_instantaneous` (secondary)** is the peak of the smoothed rate
curve. It answers "how fast was it at the very fastest instant?", but it is
upward-biased on noisy data and more artifact-sensitive. Use it only alongside
its uncertainty, and prefer `max_rate` for reporting and comparing experiments.

Both come with standard deviations (`max_rate_std`,
`max_rate_instantaneous_std`) derived from the GP posterior; the windowed one
uses a deliberately conservative bound.

If your experiments have a short, genuine burst that a 2 %-of-duration window
would average away, pass a smaller `window=` explicitly — the choice of window
*is* the definition of the quantity you are reporting, so state it with your
results.

## 6. Cross-check and quality flags

A second, completely independent estimator runs on the raw (unsmoothed) data: a
rolling least-squares straight-line fit over the same window, ignoring masked
samples. Its value at the same window position as `max_rate` is reported as
`max_rate_crosscheck`. If the two estimators disagree by more than 20 % *and*
more than 3 combined standard deviations, the curve is flagged — the primary
guard against silent over- or under-smoothing.

All flags (empty list = clean curve):

| Flag | Meaning | Typical cause |
|---|---|---|
| `max_rate_at_boundary` | Maximum sits within one window of the start/end | Truncated experiment; rate still rising at cutoff |
| `lengthscale_at_lower_bound` | Fitted lengthscale pinned at its floor | Correlated sensor noise being fitted as signal |
| `high_uncertainty` | Error bar > 50 % of the value | Very noisy data or poorly resolved maximum |
| `many_outliers_masked` | > 5 % of samples masked | Artifact-rich series — inspect the mask in the plot |
| `estimator_disagreement` | GP and rolling regression disagree | Over-/under-smoothing, unusual curve shape |
| `instantaneous_rate_spike` | Instantaneous max > 3× windowed max | Residual artifact or genuinely burst-like kinetics |
| `correlated_residuals` | Strong autocorrelation in fit residuals | Structured sensor noise; uncertainties may be optimistic |

A flag does not mean the number is wrong — it means a human should look at the
`plot_max_rate` figure before trusting it.

## 7. The result object

`extract_max_rate` returns a `MaxRateResult` dataclass:

| Field | Content |
|---|---|
| Field | Content | Dimension |
|---|---|---|
| `max_rate`, `max_rate_std`, `t_max_rate` | Sustained maximum rate, its std, and the window-centre time | substance / time, time |
| `window` | Window length actually used | time |
| `max_rate_instantaneous`, `..._std`, `t_max_rate_instantaneous` | Peak of the rate curve | substance / time, time |
| `max_rate_crosscheck` | Rolling-regression slope at the same window | substance / time |
| `outlier_mask` | Boolean array, True where samples were masked | — |
| `smooth`, `smooth_std` | Smoothed curve with uncertainty | substance |
| `rate`, `rate_std` | Rate curve with uncertainty | substance / time |
| `hyperparameters` | Fitted `lengthscale` (time), `signal_std`, `noise_std` (substance) | Quantities |
| `diagnostics` | Log-likelihood, outlier fraction, residual autocorrelation (plain floats); `median_dt`, `max_rolling_slope`, `crosscheck_std` (Quantities) | mixed |
| `flags` | Quality flags (section 6) | — |

The input series is not among the fields — see section 0. Every `Quantity`
field converts on demand, e.g. `result.rate.unit['umol / minute']` for the whole
rate curve at once.

## 8. Parameters and batch use

```python
extract_max_rate(time, values,           # Quantities: dimension time, substance
                 window=None,             # Quantity (time); None = auto
                 outlier_threshold=6.0,   # jump z-score for masking
                 outlier_pad=10,          # extra masked samples around artifacts
                 lengthscale_bounds=None, # (min, max) Quantities for the GP lengthscale
                 max_fit_points=1500,     # decimation for hyperparameter fitting
                 hyperparameters=None)    # reuse a previous fit (dict of Quantities)
```

- **`window`** is the one parameter worth thinking about (section 5).
- **`outlier_threshold`**: lower it (e.g. 5) if visible artifacts survive,
  raise it if genuine sharp kinetics are being masked.
- **`hyperparameters`**: for a batch of similar experiments, fit once and pass
  `result.hyperparameters` to the remaining curves — this skips the
  optimization (the slowest step) and makes results maximally comparable:

```python
first = extract_max_rate(time_0, values_0)
rest = [extract_max_rate(t, v, hyperparameters=first.hyperparameters)
        for t, v in remaining_experiments]
```

All tuning constants (statistical thresholds, window sizes, growth limits) are
named module-level constants at the top of `max_rate.py` with explanatory
comments.

## 9. Performance and validation

- **Speed**: about 1–2 s per 10 000-point series including hyperparameter
  fitting; ~0.3 s when hyperparameters are reused. Memory is O(n).
- **Synthetic ground truth** (logistic and induction-ramp curves,
  `src/tests/test_max_rate.py`): bias ≈ −0.5 % across noise levels up to
  noise ≈ signal/5; an injected bubble transient with 75× the true slope
  changes the result by < 15 %; irregular and coarse sampling are handled.
- **Real data**: developed against seven H2/O2 sensor series (PyroScience and
  UniAmp loggers) covering induction periods, bursts, near-linear growth,
  plateaus, lamp-off declines, heavy noise and multiple bubble transients; in
  every case the extracted window sits on the physically correct steepest
  phase.

## 10. Background reading

- Swain et al., *Nature Communications* 7:13766 (2016) — Gaussian-process
  inference of time derivatives for growth curves (the same idea, applied to
  microbiology).
- Särkkä & Solin, *Applied Stochastic Differential Equations* (Cambridge,
  2019) — the state-space/Kalman formulation of Gaussian processes used here.
- Van Breugel, Kutz & Brunton, *IEEE Access* 8:196865 (2020) — overview of
  numerical differentiation of noisy data.
