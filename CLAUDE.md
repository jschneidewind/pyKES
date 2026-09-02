# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**pyKES** is a Python package for kinetic modeling of chemical reaction networks. It bundles:

- An HDF5-backed experimental data layer (`ExperimentalDataset` / `Experiment`) and a parallel ingestion pipeline.
- ODE-based reaction simulation (`reaction_ODE`, `reaction_model`) and parameter fitting (`fitting_ODE`).
- Pathway propagation and transformation utilities.
- A reusable Streamlit UI (`streamlit_app/`) that **external repos embed and configure** via dataclasses; they do not fork it.

Install for development with:

```bash
pip install -e <path-to-pyKES>
```

## Repository layout

- [src/pyKES/database/](src/pyKES/database/) — `ExperimentalDataset`, `Experiment`, HDF5 save/load, and `read_in_experiments_multiprocessing` (parallel ingestion).
- [src/pyKES/reaction_ODE.py](src/pyKES/reaction_ODE.py), [src/pyKES/reaction_model.py](src/pyKES/reaction_model.py) — ODE integration and the unified reaction-model interface.
- [src/pyKES/fitting_ODE.py](src/pyKES/fitting_ODE.py) — parameter fitting against experimental data.
- [src/pyKES/pathways/](src/pyKES/pathways/) — pathway propagation and transformation.
- [src/pyKES/utilities/](src/pyKES/utilities/) — small focused helpers (absorption, resampling, attribute resolution, offset correction, JSON serialization, …).
- [src/pyKES/plotting/](src/pyKES/plotting/) — plotting helpers (matplotlib + plotly via Streamlit).
- [src/pyKES/streamlit_app/](src/pyKES/streamlit_app/) — reusable Streamlit pages:
  - [config_interface.py](src/pyKES/streamlit_app/config_interface.py) — `FileUploadHandler`, `DataUploadConfig`, `HomeConfig`, `PyKESStreamlitConfig`.
  - [chunked_processing.py](src/pyKES/streamlit_app/chunked_processing.py) — advances a long processing run one experiment per rerun.
  - [components/](src/pyKES/streamlit_app/components/) — `render_home`, `render_data_upload`, `render_analysis_results`, `render_time_series`.
  - [pages/](src/pyKES/streamlit_app/pages/) — Streamlit page entry points; each delegates to a component.
- [examples/external_repo/](examples/external_repo/) — sample wiring for an external app (Home, config, processing functions).
- [src/tests/](src/tests/) — pytest suite.

## Architecture conventions

- **Single source of truth**: `st.session_state.experimental_dataset` (an `ExperimentalDataset`). Pages mutate it in place.
- **`read_in_experiments_multiprocessing` runs in a `ProcessPoolExecutor`**, so user-supplied processing callables must be importable at module top level (no closures, no lambdas). The Streamlit upload page does *not* use it — it steps through `ingest_experiment` one experiment at a time (see below), which is also what makes it work in the browser, where there are no processes to fork.
- **The pages must survive stlite's single-threaded browser runtime.** External repos deploy them as a static Pyodide page, where Python, Streamlit and the UI share one event loop: a loop that processes everything inside one script run delivers nothing to the screen until it has finished. Long-running page work is therefore chunked across reruns via [streamlit_app/chunked_processing.py](src/pyKES/streamlit_app/chunked_processing.py), never looped inline. See [docs/browser_deployment.md](docs/browser_deployment.md).
- **Two ingestion modes** in `read_in_experiments_multiprocessing`:
  - `keywords` + `directory` — substring match on filenames inside `directory`.
  - `overview_df_based_processing=True` — filenames come from `overview_df[overview_df_experiment_column]`. When `directory` is also provided, non-absolute names are resolved against it (used by the Streamlit uploader to stage files in a temp dir).
- **External-repo configuration is the extension point.** Adding new behavior should generally mean adding a field to a config dataclass, not modifying the components.

## Coding principles

When editing or adding code in this repo, follow these rules. [src/pyKES/reaction_ODE.py](src/pyKES/reaction_ODE.py) is the style benchmark.

1. **Reduce nesting.** Break logic into small, self-contained functions instead of deep `if` / `with` / `try` ladders. The function's name should make its job obvious. But avoid excessive fragmentation: a helper must have one clear, nameable job.
2. **NumPy-style docstrings on every function.** Brief `Parameters` / `Returns` blocks. Skip `Examples` unless they materially clarify usage. Don't repeat type hints in prose.
3. **Meaningful comments only.** Explain the *why* — hidden constraints, invariants, surprising decisions, references to bugs/tickets. Don't restate what well-named code already shows. Separate logical blocks within a function with blank lines.
4. **Fail-fast.** Avoid broad `try/except` and silent fallbacks. Let exceptions propagate; in Streamlit they surface to the user as a traceback. Validate inputs at construction boundaries (`__post_init__`) and trust internal invariants thereafter. `try/finally` for resource cleanup is fine; `try/except: pass` is not.
5. **Be short.** Prefer concise, self-explanatory code over defensive scaffolding. Three clear lines beat ten lines of speculative robustness.
6. **Full-word names.** No single-letter or abbreviated variable names, even for mathematical quantities: `filtered_covariances`, not `P_f`; `lengthscale`, not `ell`. Function names are verbs describing the job (`parse_reactions`, `detect_artifacts`).
7. **No magic numbers.** Statistical factors, thresholds, and window sizes become named module-level constants with a short explanatory comment, grouped into commented sections at the top of the module (see [src/pyKES/utilities/max_rate.py](src/pyKES/utilities/max_rate.py)).
8. **No nested function definitions.** Keep every function at module level; pass extra data through parameters (e.g. `scipy.optimize.minimize(..., args=...)`) instead of closures. Exception: existing ODE-builder closures required by solver APIs.
9. **Avoid `while` loops.** Prefer vectorized NumPy scans (boolean arrays, `np.convolve`, `np.searchsorted`, prefix sums) or bounded `for` loops; sequential recursions (e.g. Kalman filters) use `for`.

## Documentation, tests, changelog

- Non-trivial modules get a companion document in [docs/](docs/) (e.g. [docs/max_rate.md](docs/max_rate.md)): explain how the code works in detail, readable also for non-specialists — motivation, pipeline stages, parameter guidance, validation.
- Tests live in [src/tests/](src/tests/). For numerical/analysis code, test against synthetic data with known ground truth (never against files that only exist locally), and validate new numerical algorithms against a reference implementation during development.
- New features get an entry under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
- Include a small `test_function()` demo with `if __name__ == "__main__":` at the bottom of runnable analysis modules, mirroring `reaction_ODE.py`.

## Running

- Streamlit app: `streamlit run src/pyKES/streamlit_app/Home.py`
- Tests: `pytest`
