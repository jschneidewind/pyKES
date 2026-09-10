# pyKES examples

A fully configured pyKES application and a small real dataset to drive it with.
Together they exercise every feature the package ships: ingestion, in-app
metadata editing, reprocessing, merging, all four analysis pages, and the HDF5
round trip.

```
examples/
├── external_repo/          # the app: how an external repository configures pyKES
│   ├── Home.py             # entry point
│   ├── config.py           # the one object the pyKES pages are given
│   ├── parameters.py       # declarations, analysis settings, plotting instructions
│   ├── metadata_functions.py
│   ├── raw_data_functions.py
│   ├── processing_functions.py
│   └── pages/              # one-line delegations to the pyKES components
└── example_data/           # the data: one series of six experiments
    ├── metadata/example_overview.xlsx
    ├── raw_data/           # 12 sensor files, two per experiment
    ├── example_dataset.h5  # the same series, already processed
    └── build_example_dataset.py
```

## Run it

```bash
pip install -e .            # from the pyKES repository root
streamlit run examples/external_repo/Home.py
```

## The example data

Six runs of a photocatalytic water-splitting series (`NB-316` … `NB-325`),
measured with a UniAmp hydrogen sensor and a FireStingO2 oxygen sensor. Two
groups: `Reference` runs, and `Intensity` runs that vary
`Irradiance [mW/cm2]` — so the Analysis Results page has a parameter to plot
against.

| File | What it is |
| --- | --- |
| `metadata/example_overview.xlsx` | one row per experiment: which raw files belong to it, the irradiation window, the reactor and lamp, plus display `color` and free-text `Notes` |
| `raw_data/*.csv` | UniAmp hydrogen exports, named in the sheet's `File name H2` column |
| `raw_data/*.txt` | FireStingO2 oxygen exports, named in `File name O2` |
| `example_dataset.h5` | the finished dataset — load it to skip straight to the analysis pages |

`example_dataset.h5` is produced by the app's own pipeline. Rebuild it after
changing the processing functions:

```bash
python examples/example_data/build_example_dataset.py
```

The processing is real, not a stub: each trace is cut to its irradiation
window, averaged into 10 s bins, converted to an amount of substance, and
handed to `pyKES.utilities.max_rate`, whose Gaussian-process fit separates the
kinetics from correlated sensor drift and returns the largest sustained rate
with an uncertainty. That rate becomes an apparent quantum yield through
`pyKES.utilities.calculate_efficiency`. Six experiments take a few seconds in
total.

Expected results, which is what makes this usable as a regression check:

| Experiment | H2 max rate / µmol h⁻¹ | O2 max rate / µmol h⁻¹ | H2 AQY / % |
| --- | --- | --- | --- |
| NB-316 | 7.24 | 5.12 | 0.93 |
| NB-318 | 12.63 | 8.47 | 0.81 |
| NB-319 | 6.83 | 4.54 | 0.88 |
| NB-320 | 7.19 | 4.92 | 0.93 |
| NB-322 | 12.73 | 10.33 | 0.82 |
| NB-325 | 2.25 | 1.90 | 0.72 |

H2 comes out at roughly twice O2, which is the stoichiometry of water
splitting and a sign the pipeline is wired up correctly.

---

# End-to-end walkthrough

Each step names what to look for. Steps 1–4 build a dataset from scratch;
step 5 onwards can be done on `example_dataset.h5` instead.

## 1. Start a fresh dataset

Open **Data Upload**. With nothing loaded the page offers only **Start Fresh
Dataset** — press it. The dataset it creates carries the app's group mapping,
plotting instructions and processing parameters, which is what makes the
metadata editor available later.

## 2. Upload the metadata sheet

Section **1. 📋 Upload Metadata (Excel)** → `metadata/example_overview.xlsx`.

*Look for:* "✅ Metadata merged successfully", and six rows under **Dataset
Overview** with `Processed` = `False`.

The sheet is merged **once per uploaded file**, not once per rerun. Leave it
in the uploader for the rest of the walkthrough; that is the state in which
edits used to be silently reverted.

## 3. Upload the raw data

Section **2. 📥 Upload Raw Data** → the **💧 Liquid phase** uploader → select
all 12 files in `raw_data/` → **🚀 Process data**.

*Look for:* a progress bar naming each experiment, then "✓ Processed 6
experiment(s) successfully". `Processed` flips to `True` for every row.

The second uploader (**💨 Gas phase**) is the same wiring for a different
instrument mode. The example series is liquid-phase, so it stays empty here —
it is there to show how an app declares more than one pipeline, and it appears
in the reprocessing pipeline selector in step 6.

## 4. Download the dataset

Section **6. 💾 Download Dataset** → **📥 Download HDF5**. This is the only
way to persist the session; the app holds the dataset in memory.

*Look for:* a file that loads on the Home page and shows the same six
experiments.

## 5. Edit metadata

Section **4. ✏️ Edit Metadata** is a spreadsheet grid over the overview sheet.

1. **Read-only columns.** `group`, `File name H2` and `File name O2` are
   greyed out: they select *which* measurement an experiment is, so correcting
   one means a corrected sheet and the raw files again. The caption above the
   grid lists them.
2. **Edit a processing column.** Change `Irradiance [mW/cm2]` for `NB-316`
   from `50` to `40` and press Enter.
   *Look for:* "✅ Saved 1 change(s)…" immediately, and "⚠️ 1 experiment(s)
   need reprocessing…". Nothing else on the page moves, and the grid stays
   scrolled where it was — the grid saves from inside a fragment, so the page
   itself does not re-run.
3. **Edit several at once.** Select a block of cells and paste a column
   straight out of Excel, or drag-fill from one cell. Streamlit's editor does
   this natively.
4. **Edit a free column.** Change a `Notes` cell.
   *Look for:* it saves, and **no** reprocessing warning — `Notes` is read by
   nobody but the reader.
5. **Try a fraction.** Set `Unisense Irradiation start [s]` to `605.5`.
   *Look for:* `605.5`, not `605`. Whole-number columns are widened rather
   than rounded.

## 6. Reprocess

Section **3. ♻️ Reprocess Existing Experiments** shows the standing warning
and a **"Only experiments needing reprocessing (n)"** checkbox.

Pick the **💧 Liquid phase** pipeline, tick the checkbox, leave **Refresh
metadata from the overview table** on, and press **♻️ Reprocess**.

*Look for:* "✓ Reprocessed 1 experiment(s) successfully", the warning gone,
`Processed` back to `True`, and — on the Results Table — a changed AQY for
`NB-316`, because the quantum yield is computed from the irradiance you
edited.

## 7. Uploading a sheet overrides in-app edits

Remove the workbook from the uploader in section 1, then upload
`example_overview.xlsx` again.

*Look for:* `NB-316`'s irradiance back at `50`, both in the grid and in
`Experiment.metadata`. The sheet is authoritative, which is what makes
re-uploading it the way to undo an editing session.

## 8. Analysis pages

**Results Table.** Select all six experiments.
- *Column selection:* **Results to show** and **Metadata to show**. Add
  `Irradiance [mW/cm2]`; it joins on the left of the results.
- *Sorting:* click a column header → **Sort ascending**. `H2 max. rate` orders
  `2.248` before `12.730` — by magnitude, not by first digit. The table holds
  numbers; the digits you see come from each instruction's `format`.
- *Uncertainties:* `H2 max. rate (±)` is a sortable column of its own.
- *Quality flags:* the `Rate quality flags` column carries whatever the
  max-rate fit wants a human to look at, empty for a clean series.
- *Export:* **📥 Download Table as CSV** writes numbers, not formatted text.

**Time Series.** Expand a group, tick experiments, and choose curves. `H2
amount` and `H2 rate` have different units and get separate y axes. Each
experiment is drawn in its `color` from the sheet.

**Analysis Results.** Plots one scalar per experiment, against experiment name
or against a metadata column — choose `Irradiance [mW/cm2]` to see the
intensity dependence. Only experiments whose `Active` cell is true are
included.

## 9. Colour guard

In the metadata editor, set `NB-318`'s `color` to `ligthblue` (a typo), and
clear `NB-319`'s `color` entirely.

*Look for:* the Time Series and Analysis Results pages still plot. Both
experiments are drawn in blue, and a warning names them and the values they
declare. Before the guard, the typo raised a `ValueError` out of plotly and
replaced the whole figure with a traceback, while an empty cell became the
float `nan` and silently bypassed the documented `black` default.

## 10. Merging files

Process a subset into one file and merge it into another:

1. Start a fresh dataset, upload the sheet, upload only `NB-316`'s two raw
   files, process, and download as `partial.h5`.
2. Load `example_dataset.h5` on the Home page.
3. Section **5. 📦 Merge HDF5 Files** → upload `partial.h5` → **🚀 Merge**.

*Look for:* the merged dataset reporting six experiments, duplicates skipped
with a warning naming them.

## 11. Provenance

**🧾 Dataset Provenance** shows which pyKES version and which version of *this
app* produced the results, when the dataset was created, last modified and
last processed, and a per-experiment table of the same. Reprocess one
experiment and its row updates while the others keep their earlier stamp.

## 12. Loading an older file

Load an HDF5 file written before `overview_df` became the single source of
experiment metadata and whose stored metadata contradicts its overview table.

*Look for:* a warning on the Home page naming the experiments that were
corrected, with an expander listing the columns and both values. The overview
values are adopted, because those are the ones the analysis pages resolve
against; download the dataset again to store the corrected file.

Datasets whose `processing_parameters` declare neither
`metadata_used_for_raw_data_loading` nor `metadata_used_for_processing` — every
file written before those declarations existed — get an explanatory note in
section 4 instead of a grid. Their metadata is corrected by editing the Excel
sheet and uploading it.
