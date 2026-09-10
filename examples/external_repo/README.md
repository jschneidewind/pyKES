# The example app: how an external repository configures pyKES

pyKES ships **components, not an application**. This directory is what an
embedding repository looks like: an entry script, one config object, and the
domain-specific callables that turn *this lab's* files into results. Nothing
under `src/pyKES/streamlit_app/` is forked or edited.

For a walkthrough that drives the app end to end with the data in
`examples/example_data`, see [../README.md](../README.md).

```bash
streamlit run examples/external_repo/Home.py
```

## The four layers

| File | Job |
| --- | --- |
| `Home.py`, `pages/*.py` | one-line delegations to `render_home`, `render_data_upload`, `render_analysis_results`, `render_time_series`, `render_results_table` |
| `config.py` | builds the single `PyKESStreamlitConfig` the pages are given |
| `parameters.py` | what this app declares about its own science: which metadata columns the pipeline reads, the analysis settings, and the plotting instructions |
| `metadata_functions.py`, `raw_data_functions.py`, `processing_functions.py` | the three callables `FileUploadHandler` expects |

## The three callables

A `FileUploadHandler` is "processing-enabled" only when it has all three:

```python
metadata_retrival_function(experiment_name, overview_df) -> metadata_dict
raw_data_reading_function(directory, metadata_dict)      -> raw_data_dict
processing_function(raw_data_dict, metadata_dict)        -> processed_data_dict
```

`directory` is where the upload page staged the files the user just submitted,
so the reader resolves the filenames in the metadata row against it. All three
must be importable at module top level — no closures, no lambdas — because the
non-Streamlit ingestion path runs them in a `ProcessPoolExecutor`.

Only `'experiment_name'` is required of the metadata dictionary. `'color'` and
`'group'` are used by the plotting pages when present.

## Two handlers, not two layers

`config.py` declares a liquid-phase and a gas-phase pipeline because an app
usually owns more than one instrument setup. They are **alternatives**: an
experiment is ingested by whichever pipeline matches it, and the upload page
skips experiments already flagged `Processed`, so a second handler does not add
to an experiment the first one produced. Both appear in the reprocessing
pipeline selector.

## The metadata contract

`overview_df` is the single source of every experiment's metadata. For every
column of the sheet, `Experiment.metadata` holds exactly what that
experiment's row holds — the dataset re-establishes it whenever either side
changes, and refuses to save a file where the two disagree. Two consequences
for an app:

- **A key that is not an overview column is yours.** `metadata_retrival_function`
  here adds `'experiment_name'`, and such keys are never touched.
- **A transformed overview column will be put back.** If the pipeline needs a
  column in another form, derive it under a new key rather than overwriting
  the column's own.

## Declaring what the editor may change

Two lists in `processing_parameters` — see `parameters.py` — decide what the
metadata editor on the Data Upload page allows:

| Declaration | In the grid | Effect of an edit |
| --- | --- | --- |
| `metadata_used_for_raw_data_loading` | shown, read-only | — |
| `metadata_used_for_processing` | editable | clears `Processed`, lists the experiment for reprocessing |
| anything else (`color`, `Notes`, …) | editable | none |

Both keys must be present for the editor to be offered. That is the
backward-compatibility gate — `processing_parameters` reaches a dataset from
the file it was loaded from, so an older file carries neither and its metadata
stays read-only — and the safety rule: a dataset declaring nothing locked can
never expose a filename column as editable by omission.

See [docs/metadata_editing.md](../../docs/metadata_editing.md).

## What the pages show

`plotting_instruction` carries one entry per page, all in `parameters.py`:

- `time_series_instructions` — one curve per entry, with `unit_x` / `unit_y`
  giving curves their own axes
- `kinetic_results_instructions` — one scalar per entry for the Analysis
  Results page
- `results_table_instructions` — one column per entry, with optional `unit`,
  `format` (a Python format spec) and `error`

See [docs/plotting_instructions.md](../../docs/plotting_instructions.md).

## Provenance

`external_version` on `DataUploadConfig` records the version declared in the
nearest `pyproject.toml` above `config.py` — for a real external repository,
its own — in every dataset the app processes. Bump it when the processing
behaviour changes, or the stamp says nothing. See
[docs/versioning_and_reprocessing.md](../../docs/versioning_and_reprocessing.md).

## Adapting it to your data

1. Rewrite the readers in `raw_data_functions.py` for your instruments.
2. Rewrite `processing_functions.py`; `pyKES.utilities` has the analysis
   building blocks (`max_rate`, `calculate_efficiency`, `offset_correction`,
   `time_series_resampling`, `calculate_absorption`).
3. Point the declarations and instructions in `parameters.py` at your own
   column names and result keys.
4. Declare one `FileUploadHandler` per instrument setup in `config.py`, and
   set `external_version` to your repository's own.

Long-running page work has to be chunked across reruns rather than looped
inline, or it delivers nothing to the screen in the stlite browser build —
see [docs/browser_deployment.md](../../docs/browser_deployment.md).
