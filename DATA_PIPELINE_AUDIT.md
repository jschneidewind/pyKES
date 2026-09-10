# Data pipeline audit — loading, merging, repairing, updating, downloading, saving

Audit of the data layer on `claude/pykes-data-pipeline-bugs-lkbiu1` (PR #11), focused on
paths that can produce **unreliable data** — results that no longer match the metadata
beside them, values silently changed or dropped, and work silently lost.

Every finding below was **reproduced by running the code**, either against
`ExperimentalDataset` directly or by driving the shipped example app
(`examples/external_repo/`) and a minimal synthetic app through
`streamlit.testing.v1.AppTest`. No finding is speculative; where a finding is reasoned
from code rather than executed, it says so.

The existing suite passes in full (300 passed) — none of these are caught by it.

**No code was changed.** This document is the deliverable.

---

## Contents

| # | Finding | Severity | Introduced |
|---|---|---|---|
| [1](#1) | `overview_df` round-trip through HDF5 silently changes values and dtypes | **Critical** | pre-existing, newly load-bearing |
| [2](#2) | Numeric experiment names detach experiments from their rows — both guards go blind | **Critical** | this PR (invariant + guards) |
| [3](#3) | `mark_experiment_processed` never matches a numeric name — flag can never be set | **Critical** | pre-existing, newly fatal |
| [4](#4) | Re-uploading a narrower sheet wipes columns, but only on changed rows | **High** | this PR (`merge_overview_row`) |
| [5](#5) | Merge resurrects a stale `Processed` flag out of `Experiment.metadata` | **High** | this PR (`complete_overview_rows`) |
| [6](#6) | `read_in_experiments_multiprocessing` produces a dataset that cannot be saved | **High** | this PR (save guard) |
| [7](#7) | A file with no `Processed` column becomes permanently undownloadable on open | **High** | this PR (save guard) |
| [8](#8) | A chunked job that outlives a dataset swap fills an orphan and reports success | **High** | this PR (chunked jobs) |
| [9](#9) | An experiment name containing `/` writes a file that cannot be read back | **High** | pre-existing |
| [10](#10) | Stale results are withheld from the HDF5 download but exported freely as CSV/JSON | **High** | this PR (guard scope) |
| [11](#11) | Re-uploading corrected raw data is silently ignored | **Medium** | pre-existing |
| [12](#12) | A failed save destroys the target and leaves a truncated file that loads cleanly | **Medium** | pre-existing |
| [13](#13) | Home reloads the uploaded file on every rerun, discarding everything since | **Medium** | pre-existing |
| [14](#14) | A failed HDF5 load relabels the still-loaded dataset with the failed file's name | **Medium** | pre-existing |
| [15](#15) | The app ignores `dataset.experiment_column`; a valid file kills the page | **Medium** | this PR (editor) |
| [16](#16) | Duplicate experiment rows: crash on upload, cross-writes in the editor, doubled exports | **Medium** | mixed |
| [17](#17) | `last_modified` is bumped by rendering the page, not by changing the data | **Low** | this PR |
| [18](#18) | "auto-save" is documented but does not exist | **Low** | pre-existing |
| [19](#19) | Smaller round-trip and robustness issues | **Low** | mixed |

---

<a name="1"></a>
## 1. `overview_df` round-trip through HDF5 silently changes values and dtypes — **Critical**

`write_df_to_hdf` serialises the overview sheet as pandas JSON
([`database_experiments.py:390`](src/pyKES/database/database_experiments.py#L390)) and
`read_df_from_hdf` reads it back with `pd.read_json` and default type inference
([`:425`](src/pyKES/database/database_experiments.py#L425)). Both directions are lossy.

### 1a. Floats are truncated to 10 decimal places, and small ones are destroyed

`DataFrame.to_json` defaults to `double_precision=10`, meaning ten digits *after the
decimal point* — not ten significant digits.

```
Concentration [mol/L] 0.001234567890123456      -> 0.0012345679
Tiny                  1e-12                     -> 0.0
Tiny                  2.5e-15                   -> 0.0
Big                   1.7976931348623157e+308   -> inf
```

For a kinetics package this is a live hazard: any overview column in mol/L, any rate
constant, any quantum-yield fraction below ~5e-11 becomes exactly zero on save.

**The repair machinery then launders the corruption.** The per-experiment metadata is
stored natively in HDF5 and survives *exactly*; on load, `add_experiment` compares it to
the corrupted sheet, declares the exact value a divergence, and overwrites it with the
corrupted one — reporting this to the user as a fix:

```
Stored metadata disagreed with overview_df and was re-derived from it.
  A: Scale 2.5e-15 -> 0.0;  B: Scale 0.001234567890123456 -> 0.0012345679
Their stored results were derived from the replaced values and are flagged
for reprocessing: A, B
```

After one save/load cycle both copies agree on the wrong number, and the user is told to
reprocess — which would compute results from `Scale = 0.0`.

### 1b. Types change

```
'Experiment'  ['001','002','003'] (str)   -> [1, 2, 3] (int64)      # leading zeros gone
'Date'        ['2024-01-01', …]  (str)   -> Timestamp (datetime64)
'Scale'       [1.0, 2.0]        (float64)-> [1, 2] (int64)
```

`pd.read_json` coerces numeric-looking string columns to integers and date-named columns
to timestamps. Consequences in [#2](#2).

### Why it is worse now than before

The round-trip predates this PR, but this PR made `overview_df` **the single source of
every experiment's metadata** and built two guards (`metadata_divergences`,
`stale_processed_experiments`) on top of it. A lossy channel is now the authority, and
the exact copy is discarded in its favour.

### Direction

Store the frame in a form that round-trips exactly: per-column HDF5 datasets with an
explicit dtype map, or at minimum `to_json(double_precision=15)` plus
`read_json(..., dtype=<recorded dtypes>, convert_axes=False, convert_dates=False)` with
the original `df.dtypes` written alongside the payload. A round-trip test asserting
`df.equals(read_df_from_hdf(...))` over a frame with small floats, zero-padded string
IDs and a `Date` column would have caught all of this.

**Reproduced:** `write_df_to_hdf`/`read_df_from_hdf` directly, and end to end through
`save_to_hdf5` → `load_from_hdf5`.

---

<a name="2"></a>
## 2. Numeric experiment names detach experiments from their rows — both guards go blind — **Critical**

Given [1b](#1), an overview sheet naming experiments `001, 002, 003` comes back from
HDF5 naming them `1, 2, 3`, while `dataset.experiments` is still keyed `'001', …`.

Every lookup in `overview_metadata_for_experiment`
([`:768`](src/pyKES/database/database_experiments.py#L768)) and
`experiments_with_stale_results` ([`:619`](src/pyKES/database/database_experiments.py#L619))
compares `astype(str)` against the held key: `'1' != '001'`, so **no row is found**. Both
functions treat "no row" as *"the sheet makes no claim"* rather than *"this experiment is
unverifiable"* — so both report clean.

Full reproduction, end to end:

```
--- after save + reload ---
overview:  Experiment=[1, 2, 3]        experiments dict: ['001','002','003']
divergences seen by the guard : {}          <-- invariant guard blind
stale_processed_experiments   : []          <-- staleness guard blind

# user corrects Scale 1.0 -> 10 in the metadata editor
editor index: [1, 2, 3]  (int)
apply_metadata_edits -> overview Scale = 10, Processed = False
stored metadata Scale         : 1.0         <-- never synchronised
                                                (`1 not in database.experiments`)
divergences seen by the guard : {}
stale_processed_experiments   : []
*** SAVE ACCEPTED ***
downloaded file: overview Scale = 10 | metadata Scale = 1.0 | result = 98.0
                 (result for Scale=10 would be 980.0)
```

The delivered file states an irradiance/scale of 10 next to a result computed from 1, and
its own per-experiment metadata still says 1. This is exactly the artefact
`docs/metadata_editing.md` §4b calls "a quiet trap" — produced with both guards active and
silent, and the file offered for download.

Contributing cause in `apply_metadata_edits`
([`metadata_editing.py:368`](src/pyKES/database/metadata_editing.py#L368)): the row mask
uses raw `.eq(experiment_name)` (matches the int row) but the follow-up
`synchronize_experiment_metadata` filters with `name in database.experiments` (misses the
str key), so the sheet is written and the metadata is not.

### Direction

Two separable fixes: (a) stop the coercion ([#1](#1)); (b) make "held experiment with no
overview row" an explicit, reported state rather than a silent pass — the metadata
invariant cannot be checked for such an experiment, and `save_to_hdf5` should say so.

**Reproduced:** directly and through the Data Upload page.

---

<a name="3"></a>
## 3. `mark_experiment_processed` never matches a numeric name — the flag can never be set — **Critical**

Experiment identity is compared inconsistently across the module:

| function | comparison |
|---|---|
| `select_unprocessed_experiments` ([`:336`](src/pyKES/database/data_processing.py#L336)) | returns `.astype(str)` |
| `experiments_with_stale_results` | `.astype(str)` both sides |
| `overview_metadata_for_experiment` | `.astype(str)` both sides |
| `mark_experiment_processed` ([`:272`](src/pyKES/database/data_processing.py#L272)) | raw `.eq(name)` |
| `mark_experiments_unprocessed` ([`:305`](src/pyKES/database/data_processing.py#L305)) | raw `.isin(names)` |
| `apply_metadata_edits` ([`:368`](src/pyKES/database/metadata_editing.py#L368)) | raw `.eq(name)` |

With an `Experiment` column of numbers — which Excel produces for `001`, `1`, `20240101`,
and which [1b](#1) produces from any saved file — the two halves disagree:

```
metadata function returns experiment_name as str (as the docstring requires):
  processed ok            : [True, True]
  experiments dict keys   : [('1', str), ('2', str)]
  overview Processed      : ['False', 'False']     <-- .eq('1') never matched int 1
  select_unprocessed again: ['1', '2']             <-- reprocesses everything, forever
  save                    : ValueError "Refusing to save: 2 experiment(s) hold results
                            that no longer follow from their metadata"

metadata function returns the raw cell value instead:
  overview Processed      : ['True', 'True']       <-- flag works
  save                    : TypeError: A name should be string or bytes, not <class 'int'>
```

Either way a dataset with numeric experiment names **cannot be saved at all**, and the
error message blames a metadata change that never happened. Confirmed through the app:
uploading a sheet named `001, 002`, ingesting successfully, and finding the download
button absent with `Processed` still `False` on every row.

### Direction

One shared helper for "does this row name this experiment", used by all six call sites,
plus a normalisation of the experiment column to `str` at the point the sheet enters the
dataset.

**Reproduced:** directly and through the Data Upload page.

---

<a name="4"></a>
## 4. Re-uploading a narrower sheet wipes columns — but only on the rows that changed — **High**

`merge_overview_row` has two branches
([`database_experiments.py:457`](src/pyKES/database/database_experiments.py#L457)):

```python
if existing_row[shared_columns].equals(incoming_row[shared_columns]):
    return existing_row.combine_first(incoming_row)   # keeps existing-only columns
...
merged_row = incoming_row.copy()                       # <-- drops them
```

The agreeing branch keeps every column the stored row has; the disagreeing branch keeps
only the incoming sheet's columns. So a corrected workbook that omits a column silently
empties it — for the corrected rows only:

```
BEFORE
  Experiment   File  Scale      Comment Operator Processed
  A            A.csv 1.0        keep me  jacob    True
  B            B.csv 2.0    keep me too  jacob    True

re-upload a sheet without 'Comment'/'Operator', correcting only A's Scale

AFTER
  A            A.csv 5.0            NaN    NaN    False     <-- wiped
  B            B.csv 2.0    keep me too  jacob    True      <-- kept
A metadata: {'Scale': 5.0, 'Comment': nan, 'Operator': nan}
```

`synchronize_experiment_metadata` then propagates the NaN into `Experiment.metadata`, so
the value is gone from both copies. Reproduced against the shipped example app: correcting
`NB-319`'s irradiance with a sheet that omits `Notes` erases `NB-319`'s note
(`'Reproduction NB-316'` → `nan`) while every other row keeps its note.

The inconsistency between the two branches is the tell — the disagreeing branch wants
`incoming_row.combine_first(existing_row)`.

**Reproduced:** synthetic dataset and the shipped example app.

---

<a name="5"></a>
## 5. Merge resurrects a stale `Processed` flag out of `Experiment.metadata` — **High**

Two facts combine:

1. The documented metadata pattern (and the shipped
   `examples/external_repo/metadata_functions.py`) copies **the whole overview row** into
   `Experiment.metadata` — including `Processed`. `apply_overview_metadata` deliberately
   never refreshes it, because `Processed` is in `NON_METADATA_OVERVIEW_COLUMNS`. So the
   value frozen at ingestion (or at the last "Refresh metadata" reprocessing run) lives on
   forever inside the experiment.
2. `complete_overview_rows`
   ([`:1130`](src/pyKES/database/database_experiments.py#L1130)) fills any blank merged
   cell from that stored metadata, with **no exclusion for the derived columns**:

```python
for column in merged_columns:
    if column in stored_metadata and pd.isna(row.get(column, np.nan)):
        row[column] = stored_metadata[column]
```

Both directions reproduce.

**False stale (blocks the download).** Merging a file whose sheet has no `Processed`
column: the blank cell is filled with the frozen `'False'`, the fully-processed experiment
is declared stale, and the merged dataset cannot be saved —

```
merged sheet:  C  … Processed=False
stale after merge: ['C']
save refused: Refusing to save: 1 experiment(s) hold results that no longer follow
              from their metadata (C).
```

**False current (ships stale results).** The master file's sheet explicitly flags `C` as
`Processed = False`; the file that actually holds `C` has no flag column and a frozen
`metadata['Processed'] = 'True'` left by an earlier reprocessing run. The merge prefers
the contributing source's row, finds the cell blank, and fills it from the metadata:

```
master sheet says :  C  Processed=False
MERGED sheet says :  C  Processed=True      <-- explicit False silently overwritten
stale after merge : []
merged file saved and downloadable.
```

The user's explicit statement that `C` needs reprocessing is discarded in favour of a copy
of the flag that nothing has maintained since ingestion.

### Direction

Exclude `NON_METADATA_OVERVIEW_COLUMNS` in `complete_overview_rows` (the same exclusion
`overview_metadata_for_experiment` already applies), and keep the pipeline-owned flag out
of `Experiment.metadata` in the first place.

**Reproduced:** both directions, via `merge_hdf5_files` and via the app's merge section.

---

<a name="6"></a>
## 6. `read_in_experiments_multiprocessing` produces a dataset that cannot be saved — **High**

`read_in_experiments_multiprocessing`
([`data_processing.py:505`](src/pyKES/database/data_processing.py#L505)) calls neither
`ensure_processed_column` nor `mark_experiment_processed`. That it "neither reads nor
writes the flag" is stated in `docs/metadata_editing.md` and in the changelog as a known,
tolerated quirk — but the new save-time refusal turns it into a hard failure:

```
results: [True, True]        # every experiment processed fine
overview Processed: ['False', 'False']
stale: ['A', 'B']
SAVE REFUSED: Refusing to save: 2 experiment(s) hold results that no longer follow
              from their metadata (A, B). Their metadata changed after they were
              processed, …
```

Nothing changed after processing; the flag was simply never written. The suggested remedy
(`reprocess_experiments`) works, but it means running the whole analysis a second time.

This breaks the documented workflow verbatim: `docs/guide/dataset.rst:169-191` reads the
sheet with `dtype={'Processed': str}` — i.e. a sheet that *has* the column — runs the
parallel pipeline and calls `dataset.save_to_hdf5('experiments.h5')`. That example now
raises.

**Reproduced:** the documented snippet's exact shape.

---

<a name="7"></a>
## 7. A file with no `Processed` column becomes permanently undownloadable on open — **High**

`experiments_with_stale_results` is carefully read-only ("a save must not change the
dataset it is writing") and returns `[]` for a sheet with no flag column, because such a
sheet makes no claim. But `select_experiments_needing_reprocessing`
([`:757`](src/pyKES/database/data_processing.py#L757)), which the Data Upload page calls on
**every render**, begins with `ensure_processed_column(database)` — writing
`Processed = 'False'` into the sheet the save-time function refused to touch.

```
saved without a Processed column: [Experiment, File, Scale, Comment]
reopened in the app:
  page reports needing reprocessing: ['A', 'B']
  download withheld: True
```

So merely opening any pre-flag file — including every file produced by [#6](#6) — converts
"nothing is being tracked" into "everything is stale", and the only exit is a full
reprocessing run. The mutation happens as a side effect of rendering.

### Direction

`select_experiments_needing_reprocessing` should be as read-only as its save-time twin;
seeding the column belongs to the ingestion path, which knows it is about to set it.

**Reproduced:** directly, along the page's own call path.

---

<a name="8"></a>
## 8. A chunked job that outlives a dataset swap fills an orphan and reports success — **High**

`start_chunked_job` captures the `ExperimentalDataset` **object** in
`context['database']` ([`data_upload_component.py:382`](src/pyKES/streamlit_app/components/data_upload_component.py#L382)
and [`:646`](src/pyKES/streamlit_app/components/data_upload_component.py#L646)), and
`run_job_step` / `finish_chunked_job` keep using it
([`chunked_processing.py:215`](src/pyKES/streamlit_app/chunked_processing.py#L215),
[`:237`](src/pyKES/streamlit_app/chunked_processing.py#L237)) — never re-reading
`st.session_state.experimental_dataset`.

The Data Upload page hides the merge and download sections while a job runs, but nothing
stops the user leaving the page. Driven through the real pages:

```
job in flight: 3/4 done, writing into dataset id=…424
user goes to Home and loads a different HDF5   -> session dataset is now id=…728 ['Z']
job still registered: True
user returns to Data Upload; the job resumes and finishes

page reports : ['✓ Processed 4 experiment(s) successfully']
loaded dataset (what the user sees and downloads) : ['Z']
orphaned dataset the job actually filled          : ['A','B','C','D']
```

Four experiments and all their processing time are lost, and the page reports success.
The same applies to the reprocessing job.

### Direction

Store the dataset's session-state *key* in the job context, not the object, and resolve it
at each step; or refuse to resume a job whose dataset is no longer the loaded one.

**Reproduced:** through the real `render_home` / `render_data_upload` pages.

---

<a name="9"></a>
## 9. An experiment name containing `/` writes a file that cannot be read back — **High**

`sanitize_key` / `KEY_SLASH_PLACEHOLDER` exist precisely so a `/` in a key cannot be read
as an HDF5 path separator — but experiment names are passed to `create_group` unescaped
([`:1638`](src/pyKES/database/database_experiments.py#L1638)) and read back from
`f.keys()`:

```
'Run A/B'      -> save succeeds, load raises
                  KeyError: can't locate attribute: 'experiment_name'
'plate1/well3' -> same
'overview_df'  -> "Experiment overview_df already exists. Overwriting..."
                  the overview group is DELETED and replaced by the experiment;
                  the file then loads with 0 experiments and an empty overview
```

`Run A/B` and `plate1/well3` are ordinary lab names. The save reports success, so the user
learns the file is unreadable only when they try to open it — by which time the session is
gone. The `overview_df` collision is rarer but silently destroys both the sheet and the
experiment.

### Direction

Route experiment names through `sanitize_key` on write and `restore_key` on read (the
convention is already public API), and namespace experiments under their own group so no
name can collide with `overview_df`.

**Reproduced:** save/load round trip.

---

<a name="10"></a>
## 10. Stale results are withheld from the HDF5 download but exported freely as CSV/JSON — **High**

`docs/metadata_editing.md` §4b: a file is withheld because "the results look like every
other result, and nothing reading it later can tell". The guard is on the HDF5 download
only. The Results Table page exports the same numbers as CSV
([`results_table_component.py:839`](src/pyKES/streamlit_app/components/results_table_component.py#L839))
and the Analysis Results page as JSON
([`analysis_results_component.py:1269`](src/pyKES/streamlit_app/components/analysis_results_component.py#L1269)),
with no check and no warning:

```
Data Upload page:      stale = ['NB-318']    HDF5 download offered = False
                       ⚠️ Download withheld: 1 experiment(s) hold results that no
                          longer follow from their metadata …

Results Table page:    warnings = []         downloads = ['📥 Download Table as CSV']
Analysis Results page: warnings = []
Time Series page:      warnings = []
```

The CSV is the worst case: `join_metadata_columns` puts the **new** overview metadata
beside the **old** results in one table, which is precisely the misleading artefact the
HDF5 guard exists to prevent — and it is one page click away.

### Direction

Surface staleness wherever `processed_data` is read: a banner on the three analysis pages,
and the same refusal (or at least a marked column) on their exports.

**Reproduced:** through the real pages with a stale `NB-318`.

---

<a name="11"></a>
## 11. Re-uploading corrected raw data is silently ignored — **Medium**

`_start_ingestion_job` selects work with `select_unprocessed_experiments`, which skips any
experiment already flagged processed. So uploading a corrected raw file for an experiment
that has already been ingested does nothing:

```
first ingestion results: {'A': 38.0, 'B': 38.0}
user uploads A's corrected raw file (a completely different measurement)
  infos on page   : ['No new experiments to process']
  results now     : {'A': 38.0, 'B': 38.0}
  raw data changed: False
```

It is an `st.info`, not a warning, and it does not mention that the file was discarded.
The page's whole purpose is uploading raw data; silently ignoring an upload is a trap —
the user walks away believing the correction landed. "Reprocess" does not help: it re-runs
the processing function against the **stored** raw data.

The only working escape is to edit a processing-relevant metadata cell (clearing the flag)
and re-upload — a side effect the user has no reason to connect to this. Confirmed
working, and confirmed to then drag in every other unprocessed experiment: uploading one
file while two are unprocessed floods the page with a full traceback for the missing one.

### Direction

An explicit "re-ingest these experiments from the uploaded files" control, and a warning —
not an info — naming the uploaded files that were not used.

**Reproduced:** through the real Data Upload page.

---

<a name="12"></a>
## 12. A failed save destroys the target and leaves a truncated file that loads cleanly — **Medium**

`save_to_hdf5` opens with `h5py.File(filename, 'w')`, truncating the destination before
anything is written. If the write then raises — an unpicklable object in `processed_data`,
a key collision, a full disk — what is left is a valid-looking HDF5 file containing only
what was written before the failure:

```
good file holds: ['A', 'B', 'C']
save raised: TypeError cannot pickle '_io.TextIOWrapper' object

the file that replaced it holds: ['A', 'B']
  B processed_data keys: ['r']      # the failing key is missing
  C present? False                  # never written
  -> loads without complaint, silently missing C
```

The original file is unrecoverable and the replacement gives no sign that it is partial.

### Direction

Write to a temporary file in the destination directory and `os.replace` it into place on
success — atomic, and a failed save then costs nothing.

**Reproduced:** save over an existing file with an unserializable value.

---

<a name="13"></a>
## 13. Home reloads the uploaded file on every rerun, discarding everything since — **Medium**

`render_home` reloads unconditionally whenever the uploader holds a file
([`home_component.py:56-68`](src/pyKES/streamlit_app/components/home_component.py#L56)) —
there is no `file_id` guard of the kind `_render_metadata_uploader` uses
(`MERGED_METADATA_FILE_KEY`, added on this branch for exactly this reason).

```
run 1: 6 experiments; user adds an experiment, a column and a marker (7 experiments)
run 2 (a plain rerun of the Home page, no user action):
  same dataset object : False
  MARKER survived     : False
  NEW_EXPERIMENT      : False
  n_experiments       : 6
```

**Scope, tested honestly:** navigating away to another page and back does *not* trigger
this — Streamlit clears the uploader's widget state when the page is not rendered, and a
round trip Home → Data Upload → Home preserved everything. The reachable triggers are a
rerun **while still on Home**: pressing `R` / *Rerun*, a websocket reconnect, or uploading
again. Given the app has no persistence at all ([#18](#18)), losing an ingestion run this
way is unrecoverable.

**Reproduced:** via `AppTest`, including the negative page-switch case.

---

<a name="14"></a>
## 14. A failed HDF5 load relabels the still-loaded dataset with the failed file's name — **Medium**

`st.session_state.hdf5_filename` is set from the upload *before* the load is attempted
([`home_component.py:57`](src/pyKES/streamlit_app/components/home_component.py#L57)) and is
never rolled back when the load fails:

```
after a failed upload of 'important_results_v2.h5':
  errors on page          : ['Error loading HDF5 file: Unable to synchronously open file …']
  filename shown to user  : important_results_v2.h5
  dataset actually in memory: 6 experiments -> ['NB-316','NB-318','NB-319']
  caption on page         : 'File: important_results_v2.h5'
```

The page shows the error *and*, below it, "Loaded dataset — File: important_results_v2.h5"
over the previous dataset's overview table. The Data Upload page shows the same wrong name
in its header. The user then edits and downloads the wrong dataset under the right name.

**Reproduced:** via `AppTest`.

---

<a name="15"></a>
## 15. The app ignores `dataset.experiment_column`; a valid file kills the page — **Medium**

Schema 1.2 stores `experiment_column` on the file precisely so a reader can line the sheet
up with the experiments without being told. The pages ignore it and use
`config.metadata_excel_experiment_column` everywhere (`render_metadata_editor`,
`_render_reprocessing_section`), as does `results_table_component`
(`EXPERIMENT_NAME_COLUMN = 'Experiment'`, hardcoded).

Loading a perfectly valid dataset built with `experiment_column='Sample ID'`:

```
Data Upload page exception: "None of ['Experiment'] are in the columns"
dataset.experiment_column : Sample ID
download offered          : False
```

Section 3 raises, so sections 4–6 (reprocess, merge, download) never render — the user
cannot even export their way out. Separately, `select_experiments_needing_reprocessing`
called with the wrong column returns `[]`, so staleness tracking silently stops rather
than failing.

### Direction

Read the column from the dataset, falling back to the config only for a fresh dataset; and
tell the user plainly when the configured column and the file's disagree.

**Reproduced:** through the real Home + Data Upload pages.

---

<a name="16"></a>
## 16. Duplicate experiment rows: crash on upload, cross-writes in the editor, doubled exports — **Medium**

Duplicate names in the overview sheet are handled three different ways, none of them well.

**Upload crashes opaquely.** A sheet listing `A` twice:

```
ValueError: setting an array element with a sequence. The requested array has an
inhomogeneous shape after 2 dimensions.
```

`update_overview_df` does `existing_df.loc[key]`, which returns a DataFrame rather than a
Series for a duplicated index, and `merge_overview_row` then works on it. Fail-fast is the
repo's principle, but the message names neither the sheet nor the duplicate.

**Stored duplicates are silently collapsed.** If the *stored* sheet already has two `A`
rows, the next upload leaves one — the other row disappears without a word.

**Duplicates are easy to create by accident.** A trailing/leading space in Excel produces a
second row for the same experiment; `overview_metadata_for_experiment` matches the first
one exactly, so the corrected row is silently ignored while the stale one wins.

**The editor cross-writes.** Editing the first `A` row writes to both:

```
edit 'Comment' on the FIRST A row only:
  A  a   1.0  EDITED          <-- intended
  A  a2  2.0  EDITED          <-- second row's 'y' silently overwritten
```

`apply_metadata_edits`'s row mask matches every row with that name, and
`Experiment.metadata` takes only `iloc[0]`, so the second row's data is orphaned.

**Exports double the experiment.** `join_metadata_columns` right-joins on a non-unique
index:

```
            Irradiance  H2 rate
Experiment
A                   50      1.0
A                   60      1.0     <-- one measurement, two irradiances
B                  100      2.0
```

The CSV reads as two independent experiments at different irradiance with identical rates.

### Direction

Validate uniqueness of the experiment column where the sheet enters the dataset
(`update_overview_df`, `merge_hdf5_files`) and refuse with a message naming the duplicates.

**Reproduced:** all four behaviours.

---

<a name="17"></a>
## 17. `last_modified` is bumped by rendering the page, not by changing the data — **Low**

`_render_download_section` serialises the whole dataset on every render to produce the
download bytes, and `save_to_hdf5` calls `stamp_version()`:

```
after load                                : last_modified 2026-09-10T21:08:53+00:00
after one idle rerun (no user action)     : last_modified 2026-09-10T21:08:54+00:00
```

`last_modified` records when the page was last drawn. Any downloaded file says "modified
just now" regardless of whether anything changed, which makes the provenance field
useless for the thing it exists for. `save_to_hdf5` also mutates `self.schema_version` as
a side effect. The same render also rewrites the entire HDF5 (~0.5 MB for the example
dataset) on every rerun.

**Reproduced:** via `AppTest`.

---

<a name="18"></a>
## 18. "auto-save" is documented but does not exist — **Low**

`DataUploadConfig.output_hdf5_name` is documented as *"Filename used for both the download
button and auto-save"* ([`config_interface.py:105`](src/pyKES/streamlit_app/config_interface.py#L105)),
and `render_data_upload`'s docstring mentions *"download / auto-save behavior"*
([`data_upload_component.py:111`](src/pyKES/streamlit_app/components/data_upload_component.py#L111)).
A grep over `src/`, `examples/` and `docs/` finds no auto-save anywhere — those two
docstrings are the only hits.

Everything lives in `st.session_state` and is lost when the session ends (and, in the
stlite build, when the tab closes). Combined with [#13](#13) and [#11](#11), a user who
believes the app is saving their work will lose it.

---

<a name="19"></a>
## 19. Smaller round-trip and robustness issues — **Low**

Each verified, each minor on its own.

- **Empty nested dicts vanish.** `{'fit': {}, 'ok': 1.0}` → `{'ok': 1.0}`. A processing
  function that returns an empty sub-result loses the key entirely.
- **Object arrays of strings come back as bytes.** `np.array(['a','b'], dtype=object)` →
  `array([b'a', b'b'])`. String operations downstream then break, or `b'...'` appears in
  the results table.
- **Non-scalar metadata values crash the invariant.** `values_agree`
  ([`:703`](src/pyKES/database/database_experiments.py#L703)) does `bool(stored == overview)`;
  a list or array metadata value raises `ValueError: The truth value of an array with more
  than one element is ambiguous` from `add_experiment`, `metadata_divergences`,
  `synchronize_experiment_metadata` and therefore `save_to_hdf5`.
- **Row order changes on every partial upload.** `update_overview_df` concatenates
  `[existing_only, incoming_only, merged]`, so re-uploading a sheet moves every overlapping
  row to the bottom. Harmless to the data, disorienting in the grid.
- **dtype-driven false invalidation.** `merge_overview_row` and `processing_relevant_change`
  compare with `Series.equals`, which is dtype-strict. When the incoming sheet is
  numerically homogeneous and the stored one is not, `1` vs `1.0` reads as a change and
  clears the processed flag for rows where nothing changed. (Re-uploading a typical mixed
  sheet is unaffected — verified.)
- **`read_in_experiments_multiprocessing` passes the unresolved `external_version` to
  `finalize_processing_run`** ([`:587`](src/pyKES/database/data_processing.py#L587)), unlike
  the single-threaded loop which passes the resolved one. *(Code-level; not separately
  reproduced.)*
- **Nothing records which pipeline produced an experiment's `processed_data`.** The
  reprocessing form applies one selected handler to every selected experiment. In the
  example app a mismatch fails loudly on a missing raw-data key, but an app whose handlers
  share raw-data keys would get silently wrong results, marked `Processed = True`.
  *(Code-level; not separately reproduced.)*

---

## What was checked and found sound

So the report is read as a list of exceptions, not a verdict on the whole layer:

- Nested-dict HDF5 serialisation of arrays, scalars, bools, `None`, NaN, ragged lists,
  string lists, unicode keys and slash-containing *keys* round-trips correctly. The
  `KEY_SLASH_PLACEHOLDER` convention works exactly as documented — for keys.
- Merging two files with disjoint experiments and different extra columns produces the
  right union, and `complete_overview_rows` correctly fills a column one source lacks from
  the stored metadata (the case it was written for).
- Re-uploading an unchanged workbook over the shipped example dataset is a genuine no-op:
  no flags cleared, no spurious staleness, download still offered.
- Correcting a processing-relevant column via upload clears exactly that experiment's flag
  and withholds the download; correcting an undeclared column (`Notes`) correctly leaves
  the flag alone. The declared-column policy in `docs/metadata_editing.md` behaves as
  written.
- The chunked ingestion job, the metadata editor's changed-cell detection, the
  locked-column policy, `widen_integer_columns` and the merge report all behave as
  documented when the dataset stays put.
- `reprocess_experiment_by_name` correctly re-imposes the sheet on the metadata before
  flagging the experiment processed.
- The `MERGED_METADATA_FILE_KEY` guard does prevent the re-merge-on-every-rerun problem it
  was added for; [#13](#13) is the same class of bug in the one uploader that lacks it.

---

## Suggested order of work

1. **[#1](#1) `overview_df` round-trip.** Everything else that reads the sheet inherits it,
   and it is the only finding that corrupts numbers outright.
2. **[#3](#3) one shared experiment-name comparison**, which also closes [#2](#2)'s
   contributing half.
3. **[#2](#2) make "held experiment, no overview row" a reported state**, so no future
   name mismatch can blind both guards at once.
4. **[#4](#4), [#5](#5)** — two localised fixes in the merge code added by this PR.
5. **[#6](#6), [#7](#7)** — reconcile the new save guard with the paths that never write
   the flag, before it blocks real work.
6. **[#8](#8), [#10](#10)** — the two remaining app-level ways to lose or misrepresent data.
7. The rest as convenient; [#12](#12) (atomic save) is cheap and prevents an
   unrecoverable loss.

---

## Method

- Python 3.11, `pip install -e ".[dev]"`; `pytest` → **300 passed**.
- Direct exercises of `ExperimentalDataset`, `data_processing` and `metadata_editing`
  against synthetic datasets with known ground truth.
- `streamlit.testing.v1.AppTest` driving:
  - the shipped example app (`examples/external_repo/`) over
    `examples/example_data/example_dataset.h5` and its overview workbook, across the Home,
    Data Upload, Analysis Results, Time Series and Results Table pages; and
  - a minimal synthetic app using the same `render_home` / `render_data_upload` components
    with a fast pipeline, so full ingestion, reprocessing and merge runs could be driven
    end to end.
- Findings marked *code-level* in [#19](#19) were read rather than executed; everything
  else was reproduced.
