# Editing metadata in the app

A metadata sheet is written before the experiments are processed, and it is
usually wrong somewhere: an irradiance typed from the wrong logbook line, a
volume in the wrong unit, a colour spelled `ligthblue`. Until now the only way
to correct one was to edit the Excel sheet and upload it again — and nothing in
the dataset recorded that the results sitting beside the corrected value had
been computed from the old one.

Section **4. ✏️ Edit Metadata** of the Data Upload page makes the overview
sheet editable in place, and makes the consequence of each edit explicit.

---

## 1. What the dataset has to declare

The editor is only offered for datasets that say which columns mean what.
Two lists in `processing_parameters` do that:

```python
PROCESSING_PARAMETERS = {
    'metadata_used_for_raw_data_loading': ['Experiment',
                                           'group',
                                           'File name H2',
                                           'File name O2'],
    'metadata_used_for_processing': ['Irradiance A [mW/cm2]',
                                     'Irradiation wavelength A [nm]',
                                     'Irradiated area [cm2]',
                                     'Liquid phase volume [mL]',
                                     'Offset',
                                     'Pyroscience Irradiation start [s]',
                                     'Pyroscience Irradiation end [s]',
                                     'Catalyst concentration [g/L]'],
    ...
}
```

`metadata_used_for_raw_data_loading` names the columns the *raw-data reader*
consults — which file to open, which sensor channel to read.
`metadata_used_for_processing` names the columns the *processing function*
consults.

**Both keys must be present.** A dataset carrying only the processing list gets
no editor either: without a declaration of what is locked, a filename column
would become editable by omission, which is the one thing this feature must
never do.

### Old files keep working

`processing_parameters` reaches a dataset from the HDF5 file it was loaded
from. A file written before an app declared these lists carries neither, so its
Data Upload page shows an explanatory note where the grid would be and nothing
else changes. There is no migration and no flag day: such a dataset is edited
the way it always was, by uploading a corrected sheet.

To make an existing file editable, load the sheet into a fresh dataset created
by an app whose configuration declares the two lists ("Start Fresh Dataset"),
or merge it into one.

---

## 2. The policy

| Kind of column | In the grid | Effect of an edit |
| --- | --- | --- |
| Declared under `metadata_used_for_raw_data_loading` | shown, **read-only** | — |
| Declared under `metadata_used_for_processing` | editable | clears `Processed`; the experiment is listed for reprocessing |
| Anything else (comments, labels, `color`, `group`) | editable | none |
| The `Processed` flag | not shown | — |

Raw-data-loading columns are locked because changing one does not correct the
experiment, it describes a different one: a new filename means different raw
data, which the stored `raw_data` no longer is. Correcting one means uploading
the corrected sheet **and the raw-data files** again.

The `Processed` flag is not a column of the grid at all. It is derived and the
pipeline owns it, for the same reason a thermometer is not adjustable — and an
edit changes it *after* the grid for that run has been drawn, so showing it put
a stale `True` on screen directly above the warning saying the experiment
needed reprocessing. The **Dataset Overview** table further down the page
shows the flag.

Everything not named in either list is editable and costs nothing, which is
the common case — a comment, a label, a colour. `color` and `group` are
re-derived onto the `Experiment` immediately, so a corrected colour takes
effect on the plotting pages without a reprocessing run.

A declared column the sheet does not actually have is reported as a warning
above the grid. It is neither locked nor invalidating — it is not there to be
edited — and the mismatch between the app's configuration and the uploaded
sheet is worth seeing rather than guessing at.

---

## 3. Editing several experiments at once

The grid is `st.data_editor`, so the spreadsheet gestures work: drag a
selection, fill down from a cell's corner, and paste a block copied straight
out of Excel. Correcting one column across forty experiments is a paste, not
forty edits. Measured on the real page: dragging one corrected value down three
rows arrived as one four-cell delta, and all four experiments were stored and
flagged together.

Nothing is stored until **Apply metadata changes** is pressed, and *that is
what makes the spreadsheet gestures work*. The grid sits in an `st.form`, and
inside one a committed cell sends nothing and triggers no rerun — so a
drag-fill or a pasted block is left alone until the button is pressed, and
arrives as a single delta covering every cell it touched.

Saving each cell as it was committed instead, with no form, is what an earlier
version did, and it was measurably worse: the grid was re-rendered in the
middle of the gesture, so a drag dropped rows and single edits went missing.
There is no way to have both — either the editor reports every cell as it
happens, or it leaves the gesture undisturbed and reports on submit.

Whole-number columns are widened rather than rounded, so an irradiation start
corrected to `605.5` stays `605.5` — Streamlit's editor takes its field type
from the column dtype, and an Excel column of whole numbers arrives as
`int64`.

### The page refreshes, and does not move

Applying an edit changes something section 3 shows — its standing warning and
its **"Only experiments needing reprocessing (n)"** checkbox — and section 3 is
drawn by the page body, above the editor. So applying ends in
`st.rerun(scope="app")`: the warning appears and the checkbox becomes
selectable on the same press, rather than staying stale until the next page
interaction.

An app-scoped rerun costs a page run — the uploaded workbook is checked, and
the whole HDF5 file is rewritten to feed the download button — but that is what
every page-level interaction already costs, and it buys a page that tells the
truth about its own state.

What it does *not* cost is the scroll position. Measured in headless Chromium
with the button already in view: 686 → 686 px in an isolated layout, and
1492 → 1499 px on the real page, where the 7 px is the reprocessing warning
appearing in section 3 and pushing what follows down by its own height. Two
things are what keep it still, and neither is the scope of the rerun:

* **The grid's widget key does not change on submit.** Horizontal scroll
  survives a rerun of either scope (1200 → 1200 px) and is lost only to a new
  key (1200 → 0), which is why only an uploaded workbook bumps it.
* **The status area is always the same two captions.** An `st.success` box that
  comes and goes changes the height of the section and shifts everything below
  it, which reads as the page moving under the reader.

Because the applying run ends in a rerun, everything it drew is discarded — so
what was stored is parked in session state under `METADATA_EDIT_SUMMARY_KEY`
and rendered by the run that follows.

```{note}
An earlier version was measured as jumping ~300 px on an app-scoped rerun, and
avoided one for that reason. That measurement was wrong: the browser automation
had scrolled the off-screen button into view before clicking it, and the
scrolling was its own. Re-measured with the button in view, the scope of the
rerun makes no difference to the scroll position — the jumping came from the
widget key and from the status box changing height.
```

Rows cannot be added or deleted here. New experiments come from the metadata
sheet, which is also what keeps the sheet and the dataset in step.

---

## 3a. An uploaded sheet wins

Uploading a metadata workbook overrides anything edited in the app: for every
row both sides hold, the sheet's values replace the stored ones. That is what
makes re-uploading a corrected workbook the way to undo an editing session,
and it is why the grid is re-seeded on a merge — its client-side edits must not
replay over the sheet that just replaced them.

The sheet is merged **once per uploaded file**, tracked by the uploader's
`file_id`. It used to be re-read and re-merged on every rerun, because
`st.file_uploader` keeps its file for the whole session — which turned "the
sheet wins" into "the sheet wins again one rerun after every edit". Clearing
the widget lets the same workbook be uploaded again.

---

## 4. The `Processed` flag, at every transition

The flag lives in the `Processed` column of `overview_df` as the text `'True'`
or `'False'` — text, because it round-trips through Excel and HDF5. It answers
one question: *does the stored `processed_data` follow from the metadata
sitting next to it?*

| Transition | Flag |
| --- | --- |
| Row seeded from a sheet, or by `ensure_processed_column` | `'False'` |
| `ingest_experiment` succeeds | `'True'` |
| `ingest_experiment` fails | unchanged (`'False'`) |
| Processing-relevant metadata edited | `'False'` |
| Non-processing metadata edited | unchanged |
| `reprocess_experiment_by_name` succeeds | `'True'` |
| Reprocessing fails | unchanged — the previous `processed_data` is kept |
| Sheet re-upload changing a declared column | `'False'` |
| Sheet re-upload changing only other columns | unchanged |

Two of these are new, and both were wrong before:

* **Reprocessing now raises the flag.** It never touched it, which was harmless
  only because nothing ever lowered it for an experiment that had been
  ingested. The editor does lower it, so reprocessing has to be able to
  clear it again.
* **A re-uploaded sheet no longer loses the flag.** A row the sheet changed was
  replaced wholesale by the incoming row, which carries no `Processed` column;
  the flag came back as `NaN`, which `select_unprocessed_experiments` happened
  to read as "unprocessed". It is now set explicitly — and only when a
  *declared* column differs, so re-uploading a sheet with a corrected comment
  no longer invalidates a processing run.

```{note}
`read_in_experiments_multiprocessing` neither reads nor writes the flag: it
processes every file its keywords match, regardless. That predates this work
and is unchanged by it. The Streamlit page does not use that entry point.
```

---

## 4a. The two places metadata lives cannot disagree

An edit has to reach both `overview_df` and the `Experiment` it belongs to. It
used not to: an edit reverted in the overview table could survive in the
experiment's own metadata, so the grid and the Dataset Overview showed one
number while the time-series and results pages showed another — and a
reprocessing run with metadata refresh then resolved it by pulling the reverted
value back over the stored one.

The sheet now owns those columns. The editor writes only `overview_df`, and the
stored metadata follows from it through
`ExperimentalDataset.synchronize_experiment_metadata`, so an edit cannot reach
one and miss the other. `save_to_hdf5` refuses to write a dataset where the two
disagree, and `load_from_hdf5` repairs files written before the guarantee
existed and reports what it corrected — which the Home page shows, since the
corrected values are the ones the analysis pages use. See
{doc}`guide/dataset`.

Keys a `metadata_retrival_function` adds that are not overview columns are the
app's own and are untouched; a transformed overview column, on the other hand,
is put back.

---

## 5. Reprocessing what the editor invalidated

Section **3. ♻️ Reprocess Existing Experiments** carries the other half:

* a standing warning naming every experiment whose results no longer match its
  metadata. It sits above the section, so it stays visible while a job runs;
* a checkbox, **"Only experiments needing reprocessing (n)"**, which overrides
  the experiment multiselect with exactly that list, and is disabled when the
  list is empty.

An experiment appears on that list only if the dataset actually **holds** it.
A row flagged `'False'` that has never been ingested needs *processing*, not
reprocessing — there is no stored raw data to rerun the processing function
against — and belongs to the raw-data uploader in section 2.

Leave **"Refresh metadata from the overview table"** checked. Unchecked, the
metadata stored inside the file is reused and the very edit that caused the
reprocessing is ignored.

Reprocessing needs no raw-data files: metadata and `raw_data` both come from
the dataset. See {doc}`versioning_and_reprocessing`.

---

## 6. Where the code lives

| Module | Responsibility |
| --- | --- |
| [`pyKES.database.metadata_editing`](https://github.com/jschneidewind/pyKES/blob/main/src/pyKES/database/metadata_editing.py) | the policy, Streamlit-free: which columns are locked, what changed, what an edit invalidates |
| [`pyKES.streamlit_app.components.metadata_editor`](https://github.com/jschneidewind/pyKES/blob/main/src/pyKES/streamlit_app/components/metadata_editor.py) | the grid, the apply button and the messages |
| [`pyKES.database.data_processing`](https://github.com/jschneidewind/pyKES/blob/main/src/pyKES/database/data_processing.py) | `mark_experiment_processed`, `mark_experiments_unprocessed`, `select_experiments_needing_reprocessing` |
| [`ExperimentalDataset.update_overview_df`](https://github.com/jschneidewind/pyKES/blob/main/src/pyKES/database/database_experiments.py) | the same policy applied to a re-uploaded sheet |

The editor needs no new configuration field: the column lists come from the
*dataset*, which is exactly what makes the feature unavailable for old files.
The only configuration it reads is `metadata_excel_experiment_column`.
