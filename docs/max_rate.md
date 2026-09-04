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

Inputs and outputs are [`Quantity`](https://github.com/jschneidewind/pyKES/blob/main/src/pyKES/utilities/unit_handler/quantity.py)
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
arrays it does keep (`smooth`, `nuisance`, `rate`, their uncertainties,
`outlier_mask`) exist nowhere else.

It is designed for real sensor data: thousands of points per curve, different
noise levels, induction periods, drifting baselines, and sensor artifacts such
as gas bubbles. It works on any monotone-ish kinetic trace, not just water
splitting.

---

## 1. Why this is harder than it looks

The rate is the slope (the derivative) of the measured curve. The naive
approach — subtracting neighbouring points and dividing by the time step — fails
badly on real data, for five reasons:

**Noise is amplified by differentiation.** If each point has ±0.1 µmol/L of
noise and points are 1 s apart, point-to-point slopes are uncertain by about
±0.14 µmol/L/s — often larger than the entire true rate. Any derivative
estimate must therefore smooth the data first, and the amount of smoothing must
be chosen carefully: too little and the derivative is noise, too much and sharp
features (like the end of an induction period) are flattened.

**Not all noise is white, and the coloured part is the dangerous part.**
Optodes drift with temperature, stirring beats against the vessel, the light
source ripples. What these add is not scatter but a slow wave: as smooth as the
kinetics, and therefore invisible to any test that looks at one sample at a
time. It is also the only kind of noise with a *slope*. In the well plates this
module was developed on, a baseline wave of 0.4 µmol amplitude and roughly
150 s period contributes about 1.6·10⁻⁵ µmol/s of apparent rate — larger than
the true maximum rate of the reaction it sits on. A smoother that fits one
length scale to the whole trace has no way to decline this: shortening the
length scale to track the wave raises the likelihood, so that is what the fit
does, and the wave then dominates the derivative.

**Taking a maximum of something noisy is biased.** Even after smoothing, the
estimated rate curve wiggles randomly around the true one. Picking its highest
point preferentially picks a spot where the wiggle happened to point upward, so
the naive "maximum of the derivative" systematically *overestimates* the true
maximum (statisticians call this the winner's curse). More smoothing and
averaging over a window both shrink this bias. Combined with the previous
point, this is why a noisy blank well can report a healthy positive rate.

**Sensor artifacts look like extreme rates.** When a gas bubble passes a
sensor, the reading can jump by several µmol/L within seconds and then relax
back over minutes. In the example data used to develop this module, one such
transient had an instantaneous slope **10 times larger** than the true maximum
rate. An extractor that does not recognize artifacts will happily report the
bubble as the maximum rate. Worse, one bubble is enough to drag the fitted
length scale down to its own width, which then degrades the whole trace.

**Curve shapes vary.** Some experiments show a long induction period followed
by a burst, some rise almost linearly, some level off or decline after the lamp
is switched off. A method that fits one fixed equation (a straight line, a
logistic curve, …) breaks whenever the data have a different shape. This module
is deliberately *model-free*: it never assumes a functional form for the
kinetics.

### The one assumption that makes it possible

A slow wave in the data could be drifting instrument or it could be chemistry.
Nothing in the numbers alone settles that, so the module takes a position and
states it:

> The kinetics are the slowest structure in the trace. A **one-off** sharp
> transition — an onset, a light-off step, a kink — is chemistry and must
> survive untouched. Structure that **recurs** on a time scale shorter than the
> overall rise and decay is the instrument.

Everything below is an implementation of that sentence. Note what it does *not*
say: nothing anywhere assumes the disturbance is periodic. The nuisance model
is a generic short-correlation-time random process, so a sinusoidal ripple, an
aperiodic thermal drift and a random baseline wander are all handled by the
same mechanism.

## 2. The pipeline at a glance

```
raw series
    │
    ▼
1. validation           drop NaNs, sort by time, remove duplicate time stamps
    │
    ▼
2. noise structure      robust second-difference variogram -> how much white
    │                   noise, how much correlated noise, how fast it decorrelates
    ▼
3. GP fit               kinetic Matern-5/2 + nuisance Matern-3/2 + white noise,
    │                   exact O(n) Kalman smoother, hyperparameters by likelihood
    ▼
4. excursion rejection  predict every sample from well before and from well
    │                   after it; reject where the data disagree with both while
    │                   the two agree with each other -> refit
    ▼
5. robust reweighting   two IRLS passes on the remaining residuals
    │
    ▼
6. rate extraction      max_rate = largest window-averaged rate of the KINETIC
    │                   component (robust headline)
    │                   max_rate_instantaneous = peak of the rate curve (secondary)
    ▼
7. cross-check          rolling weighted regression of the raw data + quality flags
```

Each stage is described below.

## 3. Separating the noise from the reaction

### Measuring the noise before fitting anything

Stage 2 answers one question with statistics alone, before any curve is fitted:
**how much of the scatter in this trace is uncorrelated, how much of it is
correlated, and over what time does the correlated part lose memory?**

The tool is a *robust second-difference variogram*. For a range of lags `h`,
take

```
D(h) = y(t + h) − 2·y(t) + y(t − h)
```

and measure its spread with a median-based (MAD) estimator. Two properties make
this the right quantity:

- The second difference **annihilates any straight line**, so a trend
  contributes nothing; only its *curvature* leaks in, and it does so as `h⁴`,
  which the fit models explicitly and subtracts.
- The **median** ignores the occasional bubble. A handful of artifacts, however
  violent, cannot move it.

How the spread grows with the lag is the fingerprint that separates the two
kinds of noise. White noise gives a flat variogram — `Var D(h) = 6σ²` at every
lag. Correlated noise gives a curve that climbs until the lag exceeds its
correlation time and then flattens. Fitting

```
Var D(h) = 6·σ_white² + σ_corr²·(6 − 8·k(h) + 2·k(2h)) + c·h⁴
```

(with `k` the Matern-3/2 correlation function and the last term the kinetic
curvature) gives `σ_white`, `σ_corr` and the correlation time in one cheap
least-squares fit over ~24 lags.

This measurement is not a nicety — it is what keeps the next stage honest. The
likelihood of the smoothing model on its own **cannot** choose between "the
kinetics bent" and "the baseline drifted": both optima exist, they are of
similar height, and the optimizer lands in whichever it approaches first. The
variogram settles the question outside the likelihood, from a statistic that a
trend cannot bias and an artifact cannot move.

Two guards decide whether there is a correlated component at all. It is
discarded if the variogram has **fewer than six lags**, because the model has
four parameters and a fit to a handful of points lands anywhere along a ridge
rather than measuring anything; and it is discarded if it carries **less than
10 % of the white variance**, because a state of its own is not worth adding
for that. In either case its variance is folded into the white noise and its
correlation time collapsed to one time step, which switches the nuisance
component off in all but name. Short, coarsely sampled runs land in the first
of these routinely — a 76-point run gets three lags.

That matters more than it looks, because the nuisance is a state in the model
rather than observation noise. A component invented out of an underdetermined
fit interpolates the measurement scatter point by point: the residuals collapse
towards zero, and with them the scale that the robust reweighting of section 4
calibrates against, so ordinary noise starts scoring as hundreds of standard
deviations and a slice of the series is rejected as artifacts. Folding the
amplitude in, not just collapsing the length scale, is what fixes it.

A third guard is a **floor** rather than a verdict. A Matern process with a
correlation time of one or two samples is indistinguishable from white noise at
that sampling and would interpolate it in the same way, so the correlation time
may not fall below **three sampling intervals**. When the fit lands under it,
the correlation time is pinned there and the two variances are *refitted* under
that constraint — the component is kept, not deleted. The refit is what makes
the clamp honest: a process held to a longer correlation time explains less of
the short-lag variogram, so it has to hand the white noise back the variance it
can no longer account for.

The distinction is not academic. Discarding a sub-floor component instead — as
this stage used to — re-labels its entire amplitude as white noise, and the
smoother is then left with the kinetic component as the only thing that can
explain wiggles which are, in fact, correlated. It obliges: the kinetic length
scale collapses onto them and their crests become the reported rate. One well of
a 110-well plate (AE-855_B2) missed the floor by 0.4 % with correlated noise 65
times the white variance, and came out 17 % high on a rate curve that
oscillated instead of decaying — with no flag raised, because the fold also set
the nuisance correlation time to the value that suppresses
`strong_correlated_noise`.

Fourth, the correlation time may not exceed 2 % of the run duration: a nuisance
allowed to decorrelate slowly stops being distinguishable from kinetics and
starts absorbing the curvature of the reaction curve itself. Measured
instrument disturbances sit an order of magnitude below that cap. They do *not*
keep a comparable distance from the three-sample floor at the other end: on the
110-well plate above, the fitted correlation times run from 3.0 to 17.6 sampling
intervals in an unbroken spread, which is precisely why that end has to be a
constraint the fit is held to rather than a line it can fall off.

### Two components, one observation

Stage 3 models the trace as a sum:

```
measurement  =  kinetic component  +  nuisance component  +  white noise
                (Matern-5/2,          (Matern-3/2,
                 slow, free)           pinned by stage 2)
```

Only the **kinetic component is differentiated** to give the reported rate. The
nuisance component is fitted, reported (`result.nuisance`) and then ignored.

This is what makes the module robust to low-frequency noise, and the mechanism
is worth being explicit about: it is not that the wave is filtered out, it is
that the wave is *given somewhere else to live*. Once the model contains a
component that can account for a 150 s ripple at no cost to the kinetic curve,
the likelihood no longer has any reason to shorten the kinetic length scale to
chase it — so it does not. On the well-plate data that failed before, the
fitted kinetic length scale moves from ~70 s (tracking the ripple, derivative
oscillating between ±2·10⁻⁵ µmol/s) to ~1500 s (tracking the reaction,
derivative decaying smoothly from its initial maximum).

Matern-3/2 is chosen for the nuisance precisely because it is *rougher* than
Matern-5/2: it is a better catch-all for whatever the instrument is doing, and
it cannot masquerade as a smooth reaction curve.

Because the split is uncertain, the uncertainty on the rate now correctly
includes "how much of this slope might have been drift". Traces with strong
correlated noise get visibly wider error bands, which is the honest answer.

### Why the reported curve does not follow every wiggle

`result.smooth` is the **kinetic component alone**. Plotted over the data it
deliberately runs straight through low-frequency wiggles rather than tracing
them — that is the fit working, not failing. `plot_max_rate` also draws
`smooth + nuisance` as a thin dashed line; that is the model of the *measured*
trace, and it is the one that should hug the data. Where the two separate, the
fit has decided that structure was instrumental.

## 4. Rejecting artifacts without cutting out the chemistry

### The problem with the obvious approaches

Two natural strategies both fail, in opposite directions.

*Reject whatever the fit does not explain.* This is standard robust regression,
and against a bubble it is useless: a fit whose length scale has already been
dragged down to the bubble's width passes straight through it, leaving no
residual to flag. The artifact hides behind the damage it caused.

*Reject whatever a deliberately stiff reference curve does not explain.* This
sees the bubble — and also sees every genuine sharp onset, which is exactly the
feature that must survive. The previous version of this module masked, in the
worst well-plate cases, the entire first 600 s of the reaction: the steep
initial rise scored as an artifact, the smoother was left bridging a gap over
the most important part of the trace, and the fit there reverted to the prior
mean and ran in the wrong direction.

### The test that works: predict from both sides

Every sample is predicted twice, by running the Kalman filter forwards and
backwards and propagating each filtered state across a blind **gap** of a few
percent of the run. So each sample gets a prediction from the data well
*before* it and one from the data well *after* it, neither of which has seen
the sample or its immediate neighbourhood. A sample belongs to an artifact when
all three of these hold:

1. the data disagree with the prediction from the past,
2. the data disagree with the prediction from the future,
3. **the two predictions nevertheless agree with each other.**

Condition 3 is what protects the chemistry. Across a genuine transition — an
onset, a steep sigmoid, a light-off kink — the past sees one curve and the
future sees another, and they say so loudly; the sample is left alone no matter
how badly either side predicts it on its own. Across a bubble both sides
describe the same undisturbed curve, and only the data depart from it.

The reference curve used for these predictions is the fitted kinetic length
scale raised, if necessary, to eight times the gap. A fit that has bent around
an artifact predicts essentially nothing across the gap — its extrapolation
uncertainty becomes as large as the whole trace — and would clear that artifact
of suspicion. Raising the length scale only to what the gap needs, rather than
to the stiffest curve available, keeps the test from calling the steepest
stretch of a genuinely fast reaction an artifact.

### Growing over the relaxation tail

A bubble does not end where it stops being conspicuous. It jumps, then relaxes
back over hundreds of samples whose individual deviations are too small to
detect but whose sum is exactly the level offset that inflates a rate. Each
detection is therefore extended in both directions, forwards against the
prediction from *after* the transient and backwards against the one from
*before* it, so the reference is never the artifact itself.

Growth uses **hysteresis**: a sample has to be conspicuous (4 standard
deviations) to start an excursion but only unremarkable (1 standard deviation,
held for 5 consecutive samples) to end one. And growth that runs past five gaps
without ever returning is undone completely — a deviation that persists is not
a transient but a genuine change of regime, and the samples after it are the
new normal, not artifacts.

### Then refit

The model is refitted with the excursions removed, warm-started from the first
fit so it costs a fraction of it. This refit is the point of the whole stage:
it is not mainly about drawing a nicer curve through the bubble, it is about
recovering the kinetic length scale that the bubble corrupted, on which every
later stage depends.

### Finally, ordinary robust reweighting

Two IRLS passes then downweight whatever the fit still does not explain,
chiefly single-sample spikes. The weight function is **redescending** — beyond
the threshold the weight falls as the *square* of the ratio, the IRLS weight of
a Cauchy loss, so a sample's influence decreases the further out it lies. The
merely bounded influence of a Huber weight is not enough here: the several
hundred samples of one transient, each downweighted but not rejected, still
collectively pull the fit.

Nothing is ever deleted. Suspect samples get their observation variance
inflated, up to a floor weight inside a detected transient. `outlier_mask`
reports samples whose weight fell below 0.5; unlike a hard mask it does not
mean those samples were discarded.

## 5. Smoothing with a Gaussian process

### The idea, without mathematics

The module assumes the truth is *some* smooth curve and asks: given the noisy
data, what is the most plausible smooth curve through them, and how uncertain
is it? This is Gaussian-process (GP) regression. Unlike fitting a formula, a GP
does not prescribe a shape — only a degree of smoothness — so it adapts to
induction periods, bursts, plateaus and declines alike. And because the
derivative of a smooth random curve is itself well defined, the GP directly
yields the **rate and an error bar for the rate at every time point**.

Its behaviour is controlled by five numbers (hyperparameters). Three are
learned from the data by maximizing the statistical likelihood; the two
nuisance ones come from the variogram of section 3 and are held fixed, for the
identifiability reason given there.

| Hyperparameter | Meaning | Source |
|---|---|---|
| `lengthscale` | Over what time span the kinetic curve can bend. Short = wiggly, long = stiff. | likelihood |
| `signal_std` | How far the kinetic curve wanders overall. | likelihood |
| `noise_std` | How noisy each individual measurement is (white part). | likelihood |
| `nuisance_lengthscale` | Over what time the correlated noise loses memory. | variogram |
| `nuisance_std` | How large the correlated noise is. | variogram |

The kinetic kernel is Matern-5/2, the standard choice for derivative
estimation: it is smooth enough to have a well-defined rate, but flexible
enough to follow the sharp acceleration after an induction period (an
infinitely smooth kernel would round such corners off).

A lower bound on the kinetic length scale (the largest of 20 median time steps,
0.2 % of the duration, and twice the measured noise correlation time) is the
last line of defence against fitting correlated drift as signal. On traces with
white noise only, the correlation time collapses to about one time step and the
bound reduces to the old sampling-based default, so nothing changes there.

### Why it is fast: the Kalman trick

Textbook GP regression requires building and inverting an n-by-n matrix —
for n = 10 000 points that is 10⁸ numbers and far too slow to do repeatedly.
This module instead uses the state-space formulation: a Matern-5/2 process is
mathematically *identical* to a small physical system whose state is the trio
(curve value, its first derivative, its second derivative) evolving through
time, and a Matern-3/2 process to a two-component system (value, derivative).
Stacking them gives a five-dimensional state; the sensor observes the sum of
the two value components. That system can be estimated with a Kalman filter — a
forward sweep that updates the state estimate one sample at a time — followed
by a Rauch–Tung–Striebel smoother, a backward sweep that feeds information from
later samples back to earlier ones.

The result is **exact** GP regression (verified in development against the
textbook matrix computation to machine precision) in time proportional to n:
about 2–3 s for 10 000 points including all fitting passes. Because the rate is
literally a component of the state, the derivative and its uncertainty come out
of the smoother for free — and because the kinetic and nuisance blocks are
separate states, so does their decomposition. Hyperparameters are fitted on a
decimated subset (default ≤ 1200 points) for speed; the smoothing passes always
use every point.

## 6. What "maximum rate" means here

The module reports two numbers with different characters:

**`max_rate` (the headline number)** is the largest rate *sustained over a time
window* — precisely, the largest value of
(smooth(t + w/2) − smooth(t − w/2)) / w over all window positions t, where
`smooth` is the kinetic component. The default window is 2 % of the experiment
duration, held to at least 25 time steps and at most 10 % of the duration.
Averaging over a window is what buys
robustness: an artifact that survives rejection can only perturb this number by
its amplitude divided by the window length, and the winner's-curse bias shrinks
in the same way. Windows in which the mean robustness weight has fallen below
0.5 are excluded — there the smooth curve is an unsupported bridge, and its
slope is not evidence.

Its uncertainty is the mean posterior variance of the derivative across the
window. That is the variance the window average would have if the derivative
errors were perfectly correlated over it: exact for windows shorter than the
kinetic length scale, conservative for longer ones. (Differencing the endpoint
values instead would need their posterior covariance, which the marginal
smoother output does not carry — and assuming independence there badly
overstates the error once a nuisance component makes the absolute level of the
kinetic curve ambiguous.)

**`max_rate_instantaneous` (secondary)** is the peak of the smoothed rate
curve. It answers "how fast was it at the very fastest instant?", but it is
upward-biased on noisy data and more artifact-sensitive. Use it only alongside
its uncertainty, and prefer `max_rate` for reporting and comparing experiments.

If your experiments have a short, genuine burst that a 2 %-of-duration window
would average away, pass a smaller `window=` explicitly — the choice of window
*is* the definition of the quantity you are reporting, so state it with your
results.

### Short, coarsely sampled runs

The 25-time-step floor and the 10 %-of-duration cap cross at exactly 250
samples, so the rule is continuous and there is no jump between the two
regimes. Above 250 points the cap never binds and the window is set by the
sampling, as it always was. Below it, the cap takes over and the window is
about a tenth of the run (≈ n/10 points).

The cap exists because the floor alone runs away on a short, coarsely sampled
trace. Seventy points over four minutes — a hand-logged run rather than a
continuous log — gives 25 time steps ≈ 88 s, a **third of the whole
experiment**. A window that wide does not measure a maximum; it averages the
maximum together with everything that follows it. On a fast saturating curve
that costs about a quarter of the rate. Whenever the cap is what set the
window, `window_duration_limited` is flagged, because the headline number then
rests on a handful of points and is correspondingly noisier.

### Blanks and control wells

On a well that produces nothing, the honest answer is "no significant rate",
and that is now what comes back: the reported maximum is at or below zero
rather than a positive number manufactured from a noise crest, and
`max_rate_not_significant` is flagged whenever the maximum is less than three
standard deviations above zero. Filtering on that flag is a reliable way to
separate active wells from blanks downstream.

## 7. Cross-check and quality flags

A second, completely independent estimator runs on the raw (unsmoothed) data: a
rolling weighted least-squares straight-line fit over the same window, with
downweighted samples contributing in proportion to their weight. Its value at
the same window position as `max_rate` is reported as `max_rate_crosscheck`. If
the two estimators disagree by more than 20 % *and* more than 3 combined
standard deviations, the curve is flagged — the primary guard against silent
over- or under-smoothing.

All flags (empty list = clean curve):

| Flag | Meaning | Typical cause |
|---|---|---|
| `max_rate_at_boundary` | Maximum sits within one window of the start/end | Truncated experiment; rate still rising at cutoff, or a reaction whose peak is at t = 0 |
| `max_rate_not_significant` | Maximum less than 3 σ above zero | Blank or control well; no measurable reaction |
| `window_duration_limited` | The window hit its 10 %-of-duration cap | Series too short for the sampling-based window rule; the window holds few points |
| `lengthscale_at_lower_bound` | Fitted kinetic length scale pinned at its floor | The fit still wants to track something faster than the model allows |
| `high_uncertainty` | Error bar > 50 % of the value | Very noisy data or poorly resolved maximum |
| `many_outliers_masked` | > 5 % of samples downweighted | Artifact-rich series — inspect the mask in the plot |
| `strong_correlated_noise` | Nuisance slope scale reaches the reported rate | Baseline drift as steep as the reaction; the separation carried the result |
| `estimator_disagreement` | GP and rolling regression disagree | Over-/under-smoothing, unusual curve shape |
| `instantaneous_rate_spike` | Instantaneous max > 3× windowed max | Residual artifact or genuinely burst-like kinetics |
| `correlated_residuals` | Strong autocorrelation in fit residuals | Structure the two-component model did not capture; uncertainties may be optimistic |

A flag does not mean the number is wrong — it means a human should look at the
`plot_max_rate` figure before trusting it. `strong_correlated_noise` in
particular is informational: it says the two-component separation did real work
on this trace, and the result depends on it having been right.

## 8. The result object

`extract_max_rate` returns a `MaxRateResult` dataclass:

| Field | Content | Dimension |
|---|---|---|
| `max_rate`, `max_rate_std`, `t_max_rate` | Sustained maximum rate, its std, and the window-centre time | substance / time, time |
| `window` | Window length actually used | time |
| `max_rate_instantaneous`, `..._std`, `t_max_rate_instantaneous` | Peak of the rate curve | substance / time, time |
| `max_rate_crosscheck` | Rolling-regression slope at the same window | substance / time |
| `outlier_mask` | Boolean array, True where the robust weight fell below 0.5 | — |
| `smooth`, `smooth_std` | Kinetic component with uncertainty — the reaction curve | substance |
| `nuisance` | Correlated-noise component; `smooth + nuisance` models the measured trace | substance |
| `rate`, `rate_std` | Rate curve with uncertainty (derivative of `smooth`) | substance / time |
| `hyperparameters` | `lengthscale`, `nuisance_lengthscale` (time); `signal_std`, `nuisance_std`, `noise_std` (substance) | Quantities |
| `diagnostics` | Log-likelihood, outlier fraction, residual autocorrelation (plain floats); `median_dt`, `max_rolling_slope`, `crosscheck_std`, `nuisance_rate_std`, `lengthscale_lower_bound` (Quantities) | mixed |
| `flags` | Quality flags (section 7) | — |

The input series is not among the fields — see section 0. Every `Quantity`
field converts on demand, e.g. `result.rate.unit['umol / minute']` for the whole
rate curve at once.

## 9. Parameters and batch use

```python
extract_max_rate(time, values,           # Quantities: dimension time, substance
                 window=None,             # Quantity (time); None = auto
                 robust_threshold=4.0,    # standardized deviation at which rejection starts
                 lengthscale_bounds=None, # (min, max) Quantities for the kinetic length scale
                 max_fit_points=1200,     # decimation for hyperparameter fitting
                 hyperparameters=None)    # reuse a previous fit (dict of Quantities)
```

- **`window`** is the one parameter worth thinking about (section 6).
- **`robust_threshold`**: lower it (e.g. 3) if visible artifacts survive, raise
  it if genuine sharp kinetics are being rejected. It sets both the excursion
  test and the IRLS reweighting.
- **`lengthscale_bounds`**: when given, it is used verbatim and the
  noise-derived floor of section 5 is *not* applied. Use it to override the
  automatic choice, not to nudge it.
- **`hyperparameters`**: for a batch of similar experiments, fit once and pass
  `result.hyperparameters` to the remaining curves — this skips both
  optimizations (the slowest step) and makes results maximally comparable.
  Excursion rejection and reweighting still run:

```python
first = extract_max_rate(time_0, values_0)
rest = [extract_max_rate(t, v, hyperparameters=first.hyperparameters)
        for t, v in remaining_experiments]
```

All tuning constants (statistical thresholds, window sizes, growth limits) are
named module-level constants at the top of `max_rate.py` with explanatory
comments.

## 10. Performance and validation

- **Speed**: about 2–3 s per 10 000-point series including both hyperparameter
  fits; ~1 s when hyperparameters are reused. Memory is O(n).
- **Synthetic ground truth** (`src/tests/test_max_rate.py`): logistic and
  induction-ramp curves are recovered to within a few percent across noise
  levels up to noise ≈ signal/5. A sinusoidal baseline whose own slope is more
  than **twice** the true rate changes the result by < 15 %, and the fitted
  nuisance component reproduces the wave's amplitude; an aperiodic
  Ornstein–Uhlenbeck drift is handled by the same mechanism. An injected bubble
  transient with 75× the true instantaneous slope changes the result by < 25 %
  (it would be 5× without rejection). A sharp onset is never downweighted, a
  blank series reports no significant rate, and irregular and coarse sampling
  are handled. A ~75-point run over 250 s is covered separately: its window
  stays duration-limited, its initial rate is recovered to within 10 %, nothing
  is spuriously rejected, and the cross-check stays finite.
- **Series length**: exercised from 67 to 12 272 points, i.e. over two orders
  of magnitude, and from continuous 1 Hz logs to hand-logged runs of a few
  dozen points. The two regimes meet at 250 points without a discontinuity.
- **Real data**: developed against seven H2/O2 sensor series (PyroScience and
  UniAmp loggers), a 66-well plate dataset and two short hand-logged runs,
  covering induction periods, bursts, near-linear growth, plateaus, lamp-off
  declines, heavy noise, blanks and multiple bubble transients. On the 60
  well-behaved wells the reworked pipeline reproduces the previous results to
  within a few percent; on the six artifact-rich wells that motivated the
  rework it removes inflations of up to a factor of two, and on the six blank
  wells it replaces spurious positive rates with correctly flagged near-zero
  ones.

## 11. Background reading

- Swain et al., *Nature Communications* 7:13766 (2016) — Gaussian-process
  inference of time derivatives for growth curves (the same idea, applied to
  microbiology).
- Särkkä & Solin, *Applied Stochastic Differential Equations* (Cambridge,
  2019) — the state-space/Kalman formulation of Gaussian processes used here.
- Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter*
  (Cambridge, 1989) — decomposing a series into unobserved trend and
  disturbance components, which is what section 3 does.
- Van Breugel, Kutz & Brunton, *IEEE Access* 8:196865 (2020) — overview of
  numerical differentiation of noisy data.
