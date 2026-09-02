# Dataset versioning and reprocessing

This document describes two closely related features of the pyKES data layer:

1. **Version information** — every dataset records which code produced it and
   when, and carries that record into its HDF5 file.
2. **Reprocessing** — an existing HDF5 file can be pushed through the
   processing step again, without the original raw-data files, and the version
   information is updated accordingly.

Together they answer the question a dataset otherwise cannot: *"which version
of the analysis produced these numbers, and can I redo it?"*

---

## 1. Why a version record

An HDF5 file written by pyKES holds three kinds of content per experiment:

| Content | Source |
| --- | --- |
| `metadata` | the overview Excel sheet |
| `raw_data` | the raw measurement files |
| `processed_data` | the app's `processing_function` |

Only the first two are inputs. `processed_data` is *derived*, and it changes
whenever the processing code changes — for instance when the maximum-rate
determination is reworked. Without a record of the code version, two files
holding the same experiment can disagree with no way to tell which one is
current.

## 2. The version dictionary

`ExperimentalDataset.version` is a top-level dictionary, alongside
`plotting_instruction`, `group_mapping` and `processing_parameters`:

```python
{
    'pykes_version':    '0.1.7',
    'schema_version':   '1.1',
    'created':          '2026-08-31T14:03:57+02:00',
    'last_modified':    '2026-09-02T09:21:04+02:00',
    'last_processed':   '2026-09-02T09:20:11+02:00',
    'external_version': {'app': 'photocat', 'commit': '9f3c1ab…'},
}
```

| Key | Meaning |
| --- | --- |
| `pykes_version` | version of the pyKES code that last wrote the file |
| `schema_version` | on-disk layout version (see below) |
| `created` | first save of the dataset; preserved across all later saves |
| `last_modified` | most recent save |
| `last_processed` | most recent run of a processing function |
| `external_version` | free-form provenance of the embedding app |

`ExperimentalDataset.describe_version()` renders it as a one-line summary; the
Streamlit Home page and the Data Upload page both show it.

The version is written into the HDF5 root as a JSON attribute, so it travels
with the file. Files written before this feature existed simply have no
attribute; they load with an empty dictionary and are stamped on their next
save (with `created` set to that moment, since the true creation date is not
recoverable).

### Per-experiment provenance

Each `Experiment` carries the same kind of dictionary in `Experiment.version`,
stamped whenever *that* experiment is processed. This matters after a partial
reprocessing run: the dataset-level `last_processed` says when something was
last processed, the per-experiment entries say *what*. The Data Upload page
shows them as a table under **Dataset Provenance**.

### Where the pyKES version comes from

`get_pykes_version()` prefers the `version` declared in the `pyproject.toml` of
a source checkout and falls back to the installed distribution metadata. The
detour exists because an editable install (`pip install -e`) keeps reporting
the version it was installed at, which would stamp datasets with a version the
running code no longer has.

### Recording the external app's version

The processing functions live in the *external* repository, so pyKES's own
version only tells half the story. An app records its own with:

```python
from pyKES.utilities.version_information import get_git_commit

EXTERNAL_VERSION = {'app': 'photocat', 'commit': get_git_commit(__file__)}
```

and hands it to pyKES in one of two ways:

```python
# once, on the dataset — inherited by every later processing run
dataset.set_external_version(EXTERNAL_VERSION)

# or through the Streamlit configuration
DataUploadConfig(file_handlers=[...], external_version=EXTERNAL_VERSION)
```

`get_git_commit` returns the commit hash, suffixed with `-dirty` when the
working tree holds uncommitted changes, and `None` outside a git work tree
(a deployment from a source archive, say).

### Schema version

`SCHEMA_VERSION` in `pyKES.database.database_experiments` describes the *file
layout*, not the code. It is bumped only when older readers cannot ignore a
change — renamed or removed groups, changed required attributes. Version `1.1`
added the two optional `version` attributes described above, which older
readers ignore; `1.0` files load unchanged.

---

## 3. Reprocessing an existing file

Reprocessing reruns the processing step against the metadata and raw data
**already stored in the dataset**. The raw-data files are not needed; only
`processed_data` is rebuilt. This is the path for applying an improved
algorithm to a finished file:

```python
from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.database.data_processing import reprocess_experiments

dataset = ExperimentalDataset.load_from_hdf5('experiments.h5')

results = reprocess_experiments(
    database = dataset,
    processing_function = processing_function,
    metadata_retrival_function = metadata_retrival_function,  # optional
)

dataset.save_to_hdf5('experiments.h5')
```

### Metadata: stored or refreshed

* **Without** `metadata_retrival_function`, the metadata stored in the file is
  reused unchanged — the processing code alone decides the outcome.
* **With** it, the metadata is rebuilt from `dataset.overview_df` first, so
  edits to the overview sheet (a corrected catalyst loading, a new group) take
  effect. `color` and `group` follow the refreshed metadata.

Uploading a corrected overview sheet on the Data Upload page merges it into
`overview_df`; a reprocessing run with metadata refresh then propagates it into
the experiments.

### Selecting experiments and handling failures

`experiment_names` restricts the run to a subset; the default is every
experiment in the dataset. Naming an experiment that is not in the dataset
raises immediately, before anything is touched.

An experiment whose processing raises is reported as a failure and **keeps its
previous processed data** — the experiment is only mutated once the processing
function has returned. The result list mirrors that of the ingestion pipeline:

```python
{'success': True,  'data': experiment}
{'success': False, 'file': experiment_name, 'error': message_with_traceback}
```

`progress_callback` follows the same convention as
`read_in_experiments_single_threaded`: called as `(completed, total, name)`,
once before the loop with `(0, total, None)` and again after each experiment.

### From the Streamlit app

Section **3. ♻️ Reprocess Existing Experiments** of the Data Upload page
exposes the same run:

* **Processing pipeline** — which `FileUploadHandler`'s `processing_function`
  to use (handlers without one are not offered).
* **Experiments to reprocess** — empty means all of them.
* **Refresh metadata from the overview table** — toggles the metadata
  retrieval described above.

A progress bar names the experiment being processed; failures are reported
individually with their traceback. The result lives in the session dataset, so
**download the dataset afterwards** to persist it.

The page does not call `reprocess_experiments` for this: it steps through the
experiments one Streamlit rerun at a time, via
`pyKES.streamlit_app.chunked_processing`, which is what keeps the progress bar
visible in the browser deployment. See
[browser_deployment.md](browser_deployment.md).

### What reprocessing writes

On a successful run the dataset's `last_processed`, `last_modified` and
`pykes_version` are refreshed, as are the per-experiment version dictionaries,
each inheriting the dataset's `external_version` unless one is passed
explicitly. `created` is left alone: the file is the same file, reprocessed.
