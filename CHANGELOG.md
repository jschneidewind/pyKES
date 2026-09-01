# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project aims to follow Semantic Versioning.

## [0.1.7]

### Added
- Dataset version information: `ExperimentalDataset.version` is a new top-level dictionary (alongside `plotting_instruction`, `group_mapping` and `processing_parameters`) recording the pyKES version, the schema version, the creation, last-modification and last-processing timestamps, and a free-form `external_version` entry for the embedding app. It is written into the HDF5 root as a JSON attribute and shown on the Home and Data Upload pages. Each `Experiment` carries the same kind of dictionary in `Experiment.version`, stamped whenever that experiment is processed, so partial reprocessing runs stay traceable. New module `pyKES.utilities.version_information` provides the helpers, including `get_git_commit` for external apps that want to record the commit of their own processing code; `ExperimentalDataset.set_external_version` and the new `DataUploadConfig.external_version` field feed it in. See [docs/versioning_and_reprocessing.md](docs/versioning_and_reprocessing.md).
- Reprocessing of existing HDF5 files: `pyKES.database.data_processing.reprocess_experiments` (and `reprocess_single_experiment`) rerun the processing step against the metadata and raw data already stored in a dataset, rebuilding `processed_data` without the original raw-data files — the path for applying an improved algorithm to finished files. The metadata is either reused as stored or refreshed from `overview_df` via the optional `metadata_retrival_function`, which also updates `color` and `group`. Experiments whose processing raises keep their previous results. Exposed on the Streamlit Data Upload page as "3. ♻️ Reprocess Existing Experiments", with pipeline and experiment selection, a metadata-refresh toggle and a progress bar.
- Units on the Time-Series page: `time_series_instructions` entries accept optional `'unit_x'` / `'unit_y'` keys, each either a path into the experiment holding the unit string or a literal unit. Units become the axis titles and appear in the hover box. Curves whose y unit differs are drawn against their own y axis, added on the right-hand side, so amounts (`umol`) and rates (`umol / s`) can be shown in one figure. Instructions without units keep the previous single-axis behavior with generic axis titles. See [docs/plotting_instructions.md](docs/plotting_instructions.md).
- Progress reporting for raw-data ingestion: `read_in_experiments_single_threaded` accepts an optional `progress_callback`, called as `(completed, total, experiment_name)` once before the loop (with `total` and `None`) and after each experiment. The Streamlit data-upload page uses it to render a live progress bar naming the experiment being processed.

### Changed
- `streamlit` is now required at `>=1.49`. The components pass `width="stretch"` to `st.dataframe` / `st.plotly_chart`, which older versions reject with `proto.width = width` raising a type error when a page renders a table or a figure.
- **Breaking (display only)**: the analysis results table is transposed — experiments are now rows and the configured analysis results are columns, which is also the layout of the CSV export. `build_results_table` returns a DataFrame indexed by experiment name; the constant `RESULT_NAME_COLUMN` is replaced by `EXPERIMENT_NAME_COLUMN`. The `results_table_instructions` configuration is unchanged.
- HDF5 `SCHEMA_VERSION` bumped to `1.1` for the two new optional `version` attributes (dataset root and per experiment). Older files load unchanged and are stamped on their next save; readers that ignore the attributes are unaffected.
- The time-series component was restructured into documented module-level functions (`build_trace_specifications`, `assign_y_axes`, `build_axis_layout`, `build_figure`, …), so the figure construction is testable without a Streamlit run context. Rendering behavior other than the new units and axes is unchanged.
- `pyKES.utilities.max_rate` reworked to be robust to low-frequency (correlated) sensor noise, which the previous single-length-scale smoother tracked as signal. The trace is now modelled as a slow kinetic component (Matern-5/2) plus a stationary nuisance component (Matern-3/2) plus white noise, and only the kinetic component is differentiated. The nuisance component's correlation time and amplitude are measured beforehand by a robust second-difference variogram rather than fitted by likelihood, because the likelihood alone cannot choose between "the kinetics bent" and "the baseline drifted".
- Artifact handling replaced. Instead of masking samples whose increments look like jumps — which cut out sharp kinetic onsets and, on artifact-rich traces, up to 47 % of the series including the entire initial rise — samples are now rejected only where the data disagree with a prediction from well before *and* a prediction from well after them while those two predictions agree with each other. A genuine transition makes the two sides disagree and is left untouched; a bubble makes them agree and is rejected, then grown over its relaxation tail with hysteresis and undone wherever the trace never returns. Whatever survives is downweighted by redescending (Cauchy) IRLS weights rather than deleted.
- `MaxRateResult` gained a `nuisance` field (the fitted correlated-noise component; `smooth + nuisance` models the measured trace). `smooth` and `rate` now refer to the kinetic component alone, so a plot of `smooth` deliberately does not follow low-frequency wiggles. `hyperparameters` gained `nuisance_lengthscale` and `nuisance_std`; `diagnostics` gained `nuisance_rate_std` and `lengthscale_lower_bound`. `outlier_mask` now marks downweighted rather than deleted samples.
- The uncertainty of `max_rate` is now the mean posterior variance of the derivative across the window. The previous independence bound on the two endpoint values becomes far too conservative once a nuisance component makes the absolute level of the kinetic curve ambiguous.
- New quality flags `max_rate_not_significant` (maximum less than 3 σ above zero, e.g. blank wells), `strong_correlated_noise` (the nuisance component's slope scale reaches the reported rate) and `window_duration_limited` (the series is too short for the sampling-based window rule, so the window holds only a handful of points).
- The default max-rate window is now capped at 10 % of the series duration, on top of the existing floor of 25 median time steps and 2 % of the duration. The floor and the cap cross at exactly 250 samples, so the rule is continuous and series above that length are unaffected.
- **Breaking**: `extract_max_rate` parameters `outlier_threshold` and `outlier_pad` are replaced by a single `robust_threshold` (default 4.0), and `max_fit_points` now defaults to 1200. Reused `hyperparameters` dicts from before this release lack the two nuisance entries and will raise.

### Fixed
- Maximum rates on traces with strong low-frequency noise were overestimated by up to a factor of two, and blank wells reported spurious positive rates driven by noise crests. On a 66-well validation plate the six artifact-rich wells drop by 11–54 %, the six blanks now return near-zero rates flagged `max_rate_not_significant`, and the 60 well-behaved wells reproduce their previous values to within a few percent.
- The Gaussian-process fit could be completely wrong over the first few hundred seconds of a reaction, because the initial rise was masked as an artifact and the smoother reverted to its prior mean across the gap.
- Maximum rates on short, coarsely sampled series were underestimated by up to a quarter, because the sampling-based window floor had no counterpart cap: on a 76-point, 250 s run it produced a window a third of the experiment wide, averaging the maximum together with everything after it. The new duration cap raises such a series from 0.80 to 1.05 µmol/L/s, matching a raw finite-difference reference at the same window to within 1 %.
- On the same short series the noise model collapsed: with only three variogram lags for a four-parameter fit, essentially all of the scatter was assigned to a "correlated" component with a correlation time of 1.5 samples, leaving `white_std` a thousand times too small. That nuisance state then interpolated the measurement noise point by point, driving the residuals — and with them the IRLS scale — to zero, so ordinary noise scored as hundreds of standard deviations and 16 % of the series was rejected as artifacts. A correlated component that decorrelates within three sampling intervals is now folded into the white noise, as are the amplitudes of components already declared insignificant.
- `max_rate_crosscheck` silently became NaN whenever a window held fewer than ten samples, which is the normal case on a short series. The rolling-regression minimum is now scaled to the window.

## [0.1.6]

### Added
- `pyKES.streamlit_app.components.render_results_table`: page showing the numerical analysis results of selected experiments as a table (rows = analysis results, one column per experiment). Rows are configured by the upstream app via the `'results_table_instructions'` entry of `ExperimentalDataset.plotting_instruction`, with optional `'unit'` (converts `Quantity` values), `'format'` and `'error'` keys per row.

### Changed
- `pyKES.utilities.max_rate.extract_max_rate` now works with `Quantity` objects: `time` (dimension time) and `values` (dimension substance) are required as Quantities, as are the optional `window`, `lengthscale_bounds` and reused `hyperparameters`. All calculations run internally in `mol` and `s`, and every physical field of `MaxRateResult` is returned as a `Quantity` (rates in `mol / s`), convertible on demand via `.unit['<unit>']`.
- `MaxRateResult` no longer stores the input `time` and `values` arrays, so saving a result alongside its dataset no longer duplicates the series. `plot_max_rate(result, time, values)` takes the input series as arguments instead.

## [0.1.3]

### Added
- `pyKES.utilities.max_rate`: robust maximum-rate extraction from noisy kinetic time series (`extract_max_rate`, `MaxRateResult`, `plot_max_rate`). Combines multi-scale artifact masking, an exact O(n) Matern-5/2 state-space Gaussian-process smoother (Kalman filter + RTS smoother) with maximum-likelihood hyperparameters, a sustained-window maximum-rate definition with uncertainty, a rolling-regression cross-check and automatic quality flags.

## [0.1.0] - 2026-05-11

### Added
- Initial public package structure for pyKES.
- ODE-based reaction network simulation utilities.
- Parameter fitting framework.
- HDF5-backed experimental data handling utilities.
- Streamlit app components for data upload and analysis visualization.

[Unreleased]: https://github.com/jschneidewind/pyKES/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jschneidewind/pyKES/releases/tag/v0.1.0
