# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project aims to follow Semantic Versioning.

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
