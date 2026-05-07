# External Repository Example

A minimal Streamlit app built on top of pyKES. It customizes only what is
domain-specific (file types, parsing functions, branding) and uses the pyKES
components verbatim for the rest.

## Layout

```
external_repo/
├── Home.py                 # Entry point — renders the configurable home page
├── config.py               # The single config object (PYKES_CONFIG)
├── data_processing.py      # Domain-specific metadata / raw / processing fns
└── pages/
    ├── 01_Data_Upload.py
    ├── 02_Analysis_Results.py
    └── 03_Time_Series.py
```

## How it fits together

1. `data_processing.py` exposes three callables with the signatures that
   `FileUploadHandler` expects:
   - `get_metadata_from_excel(file_name, overview_df) -> Dict[str, Any]`
   - `read_raw_data_from_csv(file_name, metadata_dict) -> Dict[str, Any]`
   - `process_photocatalysis_data(raw_data_dict, metadata_dict) -> Dict[str, Any]`

   Because ingestion runs in `ProcessPoolExecutor`, these must be importable
   at module top level (no closures, no lambdas).

2. `config.py` builds a `PyKESStreamlitConfig` containing one `HomeConfig` and
   one `DataUploadConfig`. The processing callables are attached to the
   `FileUploadHandler` that owns them — there are no top-level processing
   functions on `DataUploadConfig` itself.

3. `Home.py` and the `pages/*.py` files are one-line delegations to the
   reusable pyKES components (`render_home`, `render_data_upload`,
   `render_analysis_results`, `render_time_series`).

## Run

```bash
pip install -e /path/to/pyKES
streamlit run examples/external_repo/Home.py
```

## Adapting to your data

- Update the column names / parsing logic in `data_processing.py`.
- Adjust the `FileUploadHandler` entries in `config.py` (file types, labels,
  whether multiple files are accepted, which handler owns the processing
  pipeline).
- For ingestion driven by an overview spreadsheet, set
  `read_in_experiments_kwargs={"overview_df_based_processing": True,
  "overview_df_experiment_column": "<column>"}` on the data handler. The
  upload component stages files in a temp directory and resolves bare
  filenames in that column against it, so processing functions receive
  absolute paths.
- Customize titles, icons, and the home-page intro through `HomeConfig` and
  `PyKESStreamlitConfig`.

## Notes / gaps

- The Excel handler in this example is a placeholder — it accepts uploads
  but has no processing pipeline, so it does not populate
  `dataset.overview_df` on its own. The `overview_df_based_processing=True`
  flag on the CSV handler therefore assumes the overview is already present
  (e.g. loaded from an HDF5 file on the Home page). To drive the entire
  flow from a fresh dataset you would either pre-populate `overview_df` via
  the Excel handler's own pipeline, or switch the CSV handler to
  keyword-based ingestion.
