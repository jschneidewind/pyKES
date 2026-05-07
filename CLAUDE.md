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
  - [components/](src/pyKES/streamlit_app/components/) — `render_home`, `render_data_upload`, `render_analysis_results`, `render_time_series`.
  - [pages/](src/pyKES/streamlit_app/pages/) — Streamlit page entry points; each delegates to a component.
- [examples/external_repo/](examples/external_repo/) — sample wiring for an external app (Home, config, processing functions).
- [tests/](tests/) — pytest suite (currently focused on `pathways`).

## Architecture conventions

- **Single source of truth**: `st.session_state.experimental_dataset` (an `ExperimentalDataset`). Pages mutate it in place.
- **Ingestion runs in `ProcessPoolExecutor`**, so user-supplied processing callables must be importable at module top level (no closures, no lambdas).
- **Two ingestion modes** in `read_in_experiments_multiprocessing`:
  - `keywords` + `directory` — substring match on filenames inside `directory`.
  - `overview_df_based_processing=True` — filenames come from `overview_df[overview_df_experiment_column]`. When `directory` is also provided, non-absolute names are resolved against it (used by the Streamlit uploader to stage files in a temp dir).
- **External-repo configuration is the extension point.** Adding new behavior should generally mean adding a field to a config dataclass, not modifying the components.

## Coding principles

When editing or adding code in this repo, follow these rules:

1. **Reduce nesting.** Break logic into small, self-contained functions instead of deep `if` / `with` / `try` ladders. The function's name should make its job obvious.
2. **NumPy-style docstrings on every function.** Brief `Parameters` / `Returns` blocks. Skip `Examples` unless they materially clarify usage. Don't repeat type hints in prose.
3. **Meaningful comments only.** Explain the *why* — hidden constraints, invariants, surprising decisions, references to bugs/tickets. Don't restate what well-named code already shows.
4. **Fail-fast.** Avoid broad `try/except` and silent fallbacks. Let exceptions propagate; in Streamlit they surface to the user as a traceback. Validate inputs at construction boundaries (`__post_init__`) and trust internal invariants thereafter. `try/finally` for resource cleanup is fine; `try/except: pass` is not.
5. **Be short.** Prefer concise, self-explanatory code over defensive scaffolding. Three clear lines beat ten lines of speculative robustness.

## Running

- Streamlit app: `streamlit run src/pyKES/streamlit_app/Home.py`
- Tests: `pytest`
