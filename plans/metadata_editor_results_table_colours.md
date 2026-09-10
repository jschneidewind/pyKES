# pyKES: metadata editor, results-table selection & sorting, plot-colour guard

## Context

Three independent shortcomings in the Streamlit component library, all reported from real use:

1. **Metadata can only be changed by re-uploading an Excel sheet.** `overview_df` is the
   single source of every experiment's metadata (the reference
   `metadata_retrival_function` literally dumps the overview row into
   `Experiment.metadata`), yet the Data Upload page offers no way to correct a value
   in place. Worse, nothing tracks whether an edit invalidated the derived
   `processed_data`. The `Processed` flag today only distinguishes "never ingested"
   from "ingested"; reprocessing never touches it, and `update_overview_df` silently
   turns it into `NaN` for any row the incoming sheet changes.

2. **The Analysis Results Table sorts wrongly and shows a fixed set of columns.**
   `build_cell_text` formats every cell into a string, so `st.dataframe`'s header sort
   is lexicographic — an apparent quantum yield of `9` sorts above `18`. The columns
   are also fixed by the app's `results_table_instructions`, with no way to hide
   results or to bring metadata alongside them.

3. **A colour the user typed into the overview sheet can take a whole page down.**
   `Experiment.color` goes straight into `go.Scatter(marker=dict(color=...))`. A typo
   raises `ValueError` and Streamlit shows a traceback instead of the plot. An *empty*
   colour cell is worse: `metadata_dict.get('color', 'black')` finds the key present,
   so `experiment.color` becomes the float `nan` and the documented `'black'` default
   never applies.

Intended outcome: metadata is editable in-app for datasets that declare which columns
are safe to edit, the `Processed` flag is correct at every transition, the results
table is configurable and sorts by magnitude, and an unusable colour degrades to blue
with a visible warning instead of an exception.

Decisions already taken with the user: `st.data_editor` grid for the editor; columns
in neither declared list are editable without invalidating processing; the optional
`error` path becomes its own sortable column; the editor flags experiments and the
existing reprocessing section gains a shortcut to select exactly those.

---

## 1. Flag constants and correct `Processed` transitions

### `src/pyKES/database/database_experiments.py`

Add three module-level constants next to `SCHEMA_VERSION` — the flag is compared as a
string in five places once this work lands, and `'Processed'` / `'True'` / `'False'`
are currently repeated as literals across two modules:

```python
# The processed flag lives in overview_df as text, not as a bool: it round-trips
# through Excel and HDF5 and comes back as these strings (see ensure_processed_column).
PROCESSED_FLAG_COLUMN = 'Processed'
PROCESSED_TRUE = 'True'
PROCESSED_FALSE = 'False'
```

They belong here, not in `data_processing.py`, because `update_overview_df`
(`database_experiments.py:531`) also needs them and `data_processing` imports *from*
this module.

Give `update_overview_df` an optional third parameter:

```python
def update_overview_df(self, incoming_df, key_column, processing_relevant_columns=None)
```

In the differing-row branch (`:573-575`) the incoming row currently replaces the
existing one wholesale, so an uploaded sheet without a `Processed` column leaves the
flag as `NaN` — which `select_unprocessed_experiments` happens to read as
"unprocessed". Replace that accident with an explicit rule:

* `processing_relevant_columns is None` (default, existing callers) — any difference
  sets `PROCESSED_FALSE`, i.e. today's effective behaviour, made explicit.
* list given — the flag is reset only when one of *those* columns actually differs;
  otherwise the existing flag is carried over.

The Data Upload page passes `raw_data_loading + processing` columns from the dataset's
own declarations, so re-uploading a sheet with a corrected comment no longer
invalidates a processing run.

### `src/pyKES/database/data_processing.py`

* Add `mark_experiment_processed(database, experiment_name, overview_df_experiment_column='Experiment')`
  and `mark_experiments_unprocessed(database, experiment_names, overview_df_experiment_column='Experiment')`,
  both calling the existing `ensure_processed_column` first.
* Replace the inline `.loc[..., "Processed"] = 'True'` write in `ingest_experiment`
  (`:337-340`) with `mark_experiment_processed`.
* **`reprocess_experiment_by_name` must set the flag back to `'True'` on success** —
  it does not touch it today, which was harmless only because nothing ever cleared the
  flag for an ingested experiment. Add
  `overview_df_experiment_column: str = 'Experiment'` to its signature and call
  `mark_experiment_processed` when `result['success']`. `reprocess_single_experiment`
  stays flag-free (it takes an `Experiment`, not a dataset), and
  `reprocess_experiments` (`:679`) passes the column through to the by-name helper.
* Add `select_experiments_needing_reprocessing(database, overview_df_experiment_column='Experiment')`
  next to its siblings `select_unprocessed_experiments` (`:246`) and
  `select_experiments_to_reprocess` (`:646`): experiments **held by
  `database.experiments`** whose overview row is not flagged `PROCESSED_TRUE`. The
  `in database.experiments` half is what separates "edited, needs reprocessing" from
  "never ingested, needs processing", so no extra session state is needed to remember
  the warning across reruns.

Resulting flag table — this is the acceptance criterion for "correct in all cases":

| Transition | Flag |
| --- | --- |
| Row seeded from a sheet / `ensure_processed_column` | `'False'` |
| `ingest_experiment` succeeds | `'True'` |
| `ingest_experiment` fails | unchanged (`'False'`) |
| Processing-relevant metadata edited | `'False'` |
| Non-processing metadata edited | unchanged |
| `reprocess_experiment_by_name` succeeds | `'True'` |
| Reprocessing fails | unchanged (`'False'`) — previous `processed_data` is kept |
| Sheet re-upload changing a declared column | `'False'` |
| Sheet re-upload changing only other columns | unchanged |

`read_in_experiments_multiprocessing` (`:443`) neither reads nor writes the flag, which
already contradicts `docs/guide/dataset.rst:173`. Out of scope here; note it in the
changelog rather than changing the multiprocessing path.

---

## 2. Metadata editing logic — `src/pyKES/database/metadata_editing.py` (new)

Pure, Streamlit-free, so it is testable and reusable outside the page.

```python
METADATA_LOADING_KEY = 'metadata_used_for_raw_data_loading'
METADATA_PROCESSING_KEY = 'metadata_used_for_processing'
```

Functions, each with one nameable job:

* `metadata_editing_available(dataset) -> bool` — **both** keys present in
  `dataset.processing_parameters`. This is the backward-compatibility gate: files
  written before the external app declared them load with whatever
  `processing_parameters` the file held (`database_experiments.py:715`), the config's
  copy is only applied by "Start Fresh Dataset", and requiring both keys means a
  dataset that declares nothing locked can never expose a raw-data-loading column as
  editable.
* `locked_metadata_columns(dataset, experiment_column)` — declared loading columns that
  exist in `overview_df`, plus `PROCESSED_FLAG_COLUMN`. Shown, never editable: changing
  a filename or a well number means re-uploading the sheet and the raw files.
* `processing_relevant_metadata_columns(dataset)` — declared processing columns present
  in `overview_df`.
* `undeclared_metadata_columns(dataset)` — declared names *absent* from `overview_df`,
  surfaced as a warning so a config/sheet mismatch is visible rather than silent.
* `changed_metadata_cells(original_df, edited_df, editable_columns) -> dict` —
  `{experiment_name: {column: new_value}}`. Compare column-wise with an explicit
  NaN-equals-NaN mask rather than `DataFrame.compare`, which cannot distinguish
  "unchanged" from "cleared to NaN":

  ```python
  changed = ~(original[column].eq(edited[column])
              | (original[column].isna() & edited[column].isna()))
  ```

  Restricted to `editable_columns` so a locked column can never be written even if the
  widget were to hand one back.
* `apply_metadata_edits(database, changed_cells, experiment_column, processing_relevant_columns) -> list`
  — for each changed experiment: write the values into `database.overview_df`; update
  `experiment.metadata` in place; re-derive `experiment.color` / `experiment.group`
  from the refreshed metadata exactly as `reprocess_single_experiment` does
  (`data_processing.py:582-583`), so a corrected colour or group takes effect at once.
  Then `mark_experiments_unprocessed` for the experiments whose changed columns
  intersect `processing_relevant_columns`, and return that list.

This is why the editor writes both places rather than deferring to reprocessing:
`overview_df` is authoritative, but `Experiment.metadata` is what the plotting and
table pages resolve paths against, and the page cannot assume a
`metadata_retrival_function` is configured.

---

## 3. Metadata editor UI — `src/pyKES/streamlit_app/components/metadata_editor.py` (new)

`render_metadata_editor(config: DataUploadConfig, dataset: ExperimentalDataset) -> None`,
exported from `components/__init__.py` alongside the existing `render_*` functions.
A separate module rather than another `_render_*` inside the already 641-line
`data_upload_component.py`, and it gets its own test target.

Structure:

1. `if not metadata_editing_available(dataset)` → one `st.info` naming the two keys the
   dataset would have to declare, then return. This is the old-file path.
2. Build the view: `dataset.overview_df.set_index(experiment_column)`, restricted to the
   locked + processing + other columns that exist, with the locked ones ordered first.
3. `st.warning` for `undeclared_metadata_columns`, when any.
4. The grid:

   ```python
   edited = st.data_editor(
       view,
       key=f"metadata_editor_{st.session_state.metadata_editor_revision}",
       num_rows="fixed",
       disabled=locked_columns,
       width='stretch',
   )
   ```

   `num_rows="fixed"` keeps this an editor, not a row/column builder. The revision
   counter in the widget key is load-bearing: `st.data_editor` keeps its `edited_rows`
   delta in session state and re-applies it on top of whatever data it is handed, so
   after applying edits (or after a new sheet upload changes the same cell) a stale
   delta would silently reassert an old value. Bump the counter on apply and the
   editor is re-seeded clean.
5. `st.button("Apply metadata changes")` → `changed_metadata_cells` →
   `apply_metadata_edits` → bump the revision → report how many cells changed and, when
   the returned list is non-empty, `st.warning` naming the experiments that now need
   reprocessing → `st.rerun()`. Nothing is written until the button is pressed, so an
   in-progress edit never touches the dataset.

Streamlit's data editor already supports multi-cell selection, clipboard paste from
Excel and drag-fill, which is what makes the "edit many experiments at once"
requirement need no extra code.

### Wiring into `data_upload_component.py`

* Insert the call **after** the `any_active_job` guard (`:127-128`) as
  `st.subheader("4. ✏️ Edit Metadata")`, renumbering "📦 Merge HDF5 Files" to 5 and
  "💾 Download Dataset" to 6. Behind the guard on purpose: a grid over `overview_df`
  costs about what the guarded `st.dataframe(overview_df)` costs, and it must not
  render while a job is flipping the very flags it displays.
* In `_render_reprocessing_section`, before the form, render the standing warning from
  `select_experiments_needing_reprocessing` — this sits *before* the guard so it is
  always visible, and right next to the control that clears it.
* Add a checkbox to that form, **"Only experiments needing reprocessing (n)"**,
  disabled when the list is empty. When checked it overrides the multiselect with that
  list; the existing "all experiments when left empty" behaviour is unchanged.
* Pass the declared columns to the metadata merge:
  `dataset.update_overview_df(incoming_df, config.metadata_excel_experiment_column, loading + processing)`
  in `_render_metadata_uploader` (`:218`).
* Pass the experiment column into the reprocessing job context so
  `reprocess_experiment_by_name` can set the flag:
  `handler.overview_df_experiment_column or config.metadata_excel_experiment_column`.
  Every key in a chunked-job `context` is splatted into the step function
  (`chunked_processing.py:215`), so the new kwarg must exist on the step function —
  which it will.

No new `DataUploadConfig` field is needed: the column lists come from the *dataset*,
which is what makes the feature unavailable for old files. The only config values used
are `metadata_excel_experiment_column` and the handlers' `overview_df_experiment_column`.

---

## 4. Analysis Results Table — `src/pyKES/streamlit_app/components/results_table_component.py`

### Numeric cells (the sorting fix)

Split the current `build_cell_text` (`:201`) into value extraction and display
formatting, and keep the DataFrame numeric:

* `resolve_result_number(experiment, result_config)` — reuse the existing
  `resolve_result_value` (`:104`), `coerce_to_scalar` (`:125`) and `convert_quantity`
  (`:155`) unchanged, and simply *stop* calling `format_result_value`. Returns the
  number, the raw string when the value genuinely is one, or `None` when unresolvable.
* `build_results_table(...)` returns one numeric column per selected instruction, plus
  a `f"{label} (±)"` column for each instruction that defines `error`. Missing values
  become `None`, so pandas infers `float64` and the cell renders blank instead of `—`.
* `build_number_format(format_spec, unit)` translates the instruction's Python format
  spec into the printf spec `st.column_config.NumberColumn` expects:

  ```python
  printf_spec = f"%{format_spec}"            # '.2f' -> '%.2f', default '.4g' -> '%.4g'
  if unit:
      # A literal '%' in a unit has to be escaped in a printf-style spec.
      printf_spec = f"{printf_spec} {unit.replace('%', '%%')}"
  ```
* `build_column_config(table, results_table_instructions, selected_results)` returns
  `{column: st.column_config.NumberColumn(format=...)}`, and only for columns where
  `pandas.api.types.is_numeric_dtype` holds — a result that resolves to a string keeps
  a plain text column instead of raising.
* Render with `st.dataframe(table, column_config=..., width='stretch')`.

Display is preserved (same digits, same unit suffix); the header sort now compares
numbers because the Arrow column is `float64`. This is the whole fix — no sorting
widget, no client-side workaround. `st.column_config` needs Streamlit ≥ 1.23 and the
floor is already `streamlit>=1.49`, so nothing new is required, and no third-party grid
is introduced (an AgGrid dependency would also break the stlite bundle).

### Column selection

Two multiselects above the table, inside `col2`:

* **"Results to show"** — options `list(results_table_instructions.keys())`, default all,
  key `results_table_selected_results`. Drives which instruction columns (and their
  `(±)` companions) are built.
* **"Metadata to show"** — options `overview_df` columns minus the experiment column,
  default empty so the current appearance is unchanged. Joined on the left of the
  results columns, straight from `overview_df` so each column keeps its real dtype and
  therefore sorts correctly with no extra handling. When `overview_df` is empty, offer
  no options and say so in a caption.

Both feed the CSV download unchanged (`table.to_csv()`), which now exports numbers
rather than pre-formatted strings.

The other `st.dataframe` calls in the package are already `overview_df` slices with
real dtypes and sort fine; the only string-formatted table besides this one is the
provenance table in `data_upload_component.py:605`, whose columns are timestamps and
version strings where lexicographic order is correct. So no other table needs the fix.

---

## 5. Plot-colour guard — `src/pyKES/plotting/plot_colors.py` (new)

```python
FALLBACK_PLOT_COLOR = 'blue'

def resolve_plot_color(color):
    # CSS functional notation is valid for plotly but not for matplotlib, so it is
    # passed through rather than run past is_color_like.
    if isinstance(color, str) and color.strip().lower().startswith(('rgb', 'hsl')):
        return color
    if matplotlib.colors.is_color_like(color):
        return matplotlib.colors.to_hex(color)
    return FALLBACK_PLOT_COLOR


def unrecognized_plot_colors(experiments) -> dict:
    """Map experiment name to the unusable colour it declares."""
```

`matplotlib.colors.is_color_like` is public, exception-free (so no `try/except`, unlike
`plotting/lighten_colors.py:23`), and matplotlib is already a hard dependency.
Converting to hex on the way out also neutralises the matplotlib-only names
(`'tab:blue'`, `'C0'`) that would otherwise pass the check and then be rejected by
plotly. `is_color_like(nan)` is `False`, so the empty-colour-cell case that today
yields a float `nan` also lands on blue.

Call sites — the resolver is applied where the colour is *read for plotting*, leaving
the value the user typed intact in `overview_df`, `Experiment.color` and the metadata
listings:

* `time_series_component.py:244` — `'color': resolve_plot_color(experiment.color)` in
  `build_trace_specifications`; covers `marker.color`, `line.color` and
  `hoverlabel.font_color` at `:477-480` in one place.
* `analysis_results_component.py:793` — `exp_color = resolve_plot_color(first_exp_data.color)`
  in `create_plotly_figure`. The JSON export at `:1010` keeps the raw metadata value:
  it is a record of the sheet, not a plot instruction.
* `fitting_ODE.py:403-408` — the matplotlib consumer; resolve once before the three
  `ax.*` calls.

Each Streamlit page renders one `st.warning` from `unrecognized_plot_colors` naming the
experiments and their unusable colours, so the fallback is visible rather than silent —
the fail-fast rule cannot be honoured literally here (the user asked for blue), but it
should not be quiet either.

---

## 6. Tests

New:

* `src/tests/test_metadata_editing.py` — the whole flag table from §1 driven through
  `apply_metadata_edits` on a synthetic dataset built with
  `ExperimentalDataset(overview_df=...)` + `add_experiment(Experiment(...))`, following
  `test_reprocessing.py:48`. Cover: processing column edited → `'False'` + returned in
  the list; other column edited → flag untouched; locked column never written;
  `experiment.metadata` / `.color` / `.group` refreshed; `metadata_editing_available`
  false for a dataset with empty `processing_parameters`; NaN→value and value→NaN both
  detected as changes.
* `src/tests/test_plot_colors.py` — names, hex, `'tab:blue'`, `'rgb(1,2,3)'`,
  `float('nan')`, `''`, a typo; plus `unrecognized_plot_colors` on a small mapping.

Updated:

* `src/tests/test_results_table.py` — the four assertions pinning string cells
  (`'12.35 ± 0.68'`, `'18.00'`, `MISSING_VALUE_PLACEHOLDER`) become numeric assertions
  plus a `(±)` column; add a test that `table.sort_values(column)` orders `9` before
  `18` (the regression the user reported) and one for `build_number_format` including
  the `'%'`-unit escape.
* `src/tests/test_reprocessing.py` — successful reprocessing sets `Processed` to
  `'True'`; a failing one leaves it `'False'` and keeps the previous `processed_data`.
* `src/tests/test_data_processing.py` — unchanged behaviour, but re-run: it pins the
  seeding and selection semantics the new constants replace literals for.
* `src/tests/test_chunked_processing.py` — its `AppTest` run of the real page asserts
  `"Download Dataset" in element.value`, so renumbering the subheaders is safe; re-run
  to confirm the added section does not disturb the job lifecycle.
* `src/tests/data/processing_parameters.py` — add the two declaration lists to
  `PROCESSING_PARAMETERS` and a `results_table_instructions` block (with one `error`
  path) to `PLOTTING_INSTRUCTIONS`, so `examples/external_repo` exercises both new
  features end to end. The example currently hits
  `st.error("No 'results_table_instructions' found…")`.

---

## 7. Documentation & changelog

* `docs/metadata_editing.md` (new) — the two declaration lists, the policy and why
  raw-data-loading columns are locked, the full flag table, the reprocessing shortcut,
  and the backward-compatibility gate for old files.
* `docs/plotting_instructions.md` §2 — the `error` path now produces its own
  `label (±)` column, unresolvable cells render blank rather than `—`, and cells carry
  numbers formatted by `column_config`; document the two new multiselects.
* `docs/versioning_and_reprocessing.md` — reprocessing now clears the reprocessing
  flag; cross-link the metadata editor as the other thing that sets it.
* `docs/guide/streamlit_app.rst` — mention the new Data Upload section.
* `CHANGELOG.md` `[Unreleased]` (currently empty) — one `### Added` entry per feature
  and `### Fixed` for the sort and the colour guard, in the existing long-prose,
  source-linked style, marking the results-table cell types as
  `**Breaking (display only)**` and noting that
  `read_in_experiments_multiprocessing` still ignores the flag.
* `CLAUDE.md` is stale on two points touched here (`streamlit_app/pages/` and
  `streamlit run src/pyKES/streamlit_app/Home.py` no longer exist). Correcting it is a
  one-line fix worth folding in.

---

## 8. Verification

Nothing is installed in this container and there is no `.venv`, so first:

```bash
cd /home/user/pyKES && uv sync --extra dev     # or: pip install -e ".[dev]"
```

1. **Unit tests** — `pytest src/tests -q`. The suite must be green, including the
   `AppTest`-driven `test_chunked_processing.py`.
2. **Sorting, headless** — in `test_results_table.py`, assert
   `list(build_results_table([...]).sort_values('Apparent quantum yield (%)').index)`
   puts the `9` row before the `18` row. This is the reported bug expressed as a test.
3. **The real app** — `streamlit run examples/external_repo/Home.py` (per the
   `pykes-streamlit` skill; there is no entry script inside the package):
   * Start Fresh Dataset → upload `src/tests/data/260507_Complete.xlsx` → the
     **Edit Metadata** grid appears with `File name H2/O2` disabled and the irradiance /
     volume / offset columns editable.
   * Upload the raw CSVs and process → flags go `'True'`.
   * Change `Irradiance [mW/cm2]` for two experiments → Apply → warning names exactly
     those two, section 3's checkbox offers them, reprocessing clears the flags.
   * Change a `Comment`-style column → Apply → no warning, flags stay `'True'`.
   * Download the HDF5, reload it on Home → edits and flags survive the round trip.
   * Load an HDF5 written *before* the declarations exist (save one from the current
     `main`) → the section shows the "not available for this dataset" info instead.
4. **Results table** — on the Results Table page, deselect a result, add
   `Irradiance [mW/cm2]` as a metadata column, sort by the AQY column both ways and
   confirm magnitude order; download the CSV and confirm it holds numbers.
5. **Colour guard** — set one experiment's `color` cell to `ligthblue` (typo) and
   another to empty, reprocess, and open the Time Series and Analysis Results pages:
   both plot, the two curves are blue, and one warning names them.
