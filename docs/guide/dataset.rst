Experimental datasets
=====================

:mod:`pyKES.database` is the storage layer everything else reads from. An
:class:`~pyKES.database.database_experiments.ExperimentalDataset` holds a set of
experiments, the overview sheet that describes them, and the configuration the
embedding application supplies — all in one HDF5 file.


What is in a dataset
--------------------

.. mermaid::

   flowchart TB
       subgraph DS["ExperimentalDataset"]
           direction TB
           OV["overview_df<br/><i>the experiment list</i>"]
           EXPS["experiments<br/><i>name → Experiment</i>"]
           CFG["plotting_instruction<br/>group_mapping<br/>processing_parameters"]
           VER["version<br/><i>provenance</i>"]
       end

       subgraph EX["Experiment"]
           direction TB
           MD["metadata<br/><i>from the overview sheet</i>"]
           RAW["raw_data<br/><i>as measured</i>"]
           PROC["processed_data<br/><i>derived results</i>"]
           EV["version · color · group"]
       end

       EXPS --> EX

       style RAW fill:#e1f5e1,stroke:#5a9,stroke-width:2px

The highlighted box is the design decision that matters most: **the raw data
stays in the file.** It costs space, and it buys the ability to rerun an
improved processing algorithm over a finished dataset without going back to the
original instrument files — often years later, on a machine that no longer has
them. See :doc:`/versioning_and_reprocessing`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Holds
   * - ``experiments``
     - ``{name: Experiment}``. The dataset is addressed by experiment name
       throughout.
   * - ``overview_df``
     - The overview sheet, one row per experiment, plus a ``Processed`` column
       maintained by the ingestion pipelines.
   * - ``plotting_instruction``
     - What the Streamlit time-series page should plot. See
       :doc:`/plotting_instructions`.
   * - ``group_mapping``
     - Display names and grouping used by the Streamlit pages.
   * - ``processing_parameters``
     - Whatever the processing functions were configured with — and, after a
       fit, the whole fitting model under ``'fitting_model'``.
   * - ``version``
     - pyKES version, schema version, timestamps, and the embedding app's own
       version.


Getting data in
---------------

pyKES does not know how to read your instrument, and does not try to. You
supply three callables; pyKES supplies the loop, the error handling, the
storage and the provenance.

.. code-block:: python

   def metadata_retrival_function(experiment_name, overview_df):
       """Return the metadata dict for one experiment."""
       row = overview_df[overview_df['Experiment'] == experiment_name].iloc[0]

       return {'experiment_name': experiment_name,      # required
               'color': row['color'],                   # optional
               'group': row['group'],                   # optional
               'file_name': row['file_name'],
               'ru_concentration_uM': float(row['Ru concentration'])}


   def raw_data_reading_function(directory, metadata_dict):
       """Read the raw measurement for one experiment."""
       data = np.genfromtxt(Path(directory) / metadata_dict['file_name'],
                            delimiter=',', skip_header=1)

       return {'time': data[:, 0], 'signal': data[:, 1]}


   def processing_function(raw_data_dict, metadata_dict):
       """Turn raw data into results."""
       result = extract_max_rate(Quantity(raw_data_dict['time'], 's'),
                                 Quantity(raw_data_dict['signal'], 'umol'))

       return {'max_rate': result.max_rate,
               'time_series': {'x': raw_data_dict['time'],
                               'y': raw_data_dict['signal']}}

Only ``'experiment_name'`` is required of the metadata; ``'color'`` and
``'group'`` are used by the Streamlit pages and default to ``'black'`` and
``'default'``.

Then run one of the three entry points:

.. code-block:: python

   from pyKES.database.database_experiments import (ExperimentalDataset,
                                                    import_overview_excel)
   from pyKES.database.data_processing import read_in_experiments_multiprocessing

   overview_df = import_overview_excel('overview.xlsx', 'Sheet1',
                                       dtype={'Processed': str})

   dataset = ExperimentalDataset(overview_df=overview_df,
                                 group_mapping=GROUP_MAPPING,
                                 plotting_instruction=PLOTTING_INSTRUCTIONS,
                                 processing_parameters=PROCESSING_PARAMETERS)

   results = read_in_experiments_multiprocessing(
       database=dataset,
       metadata_retrival_function=metadata_retrival_function,
       raw_data_reading_function=raw_data_reading_function,
       processing_function=processing_function,
       overview_df_based_processing=True,
       directory='data_files')

   dataset.save_to_hdf5('experiments.h5')

.. warning::

   The multiprocessing entry point sends the three callables to worker
   processes, so they must be importable at **module top level** — no closures,
   no lambdas, nothing defined inside another function. Put them in their own
   module and import them. Use
   :func:`~pyKES.database.data_processing.read_in_experiments_single_threaded`
   where that is not possible.

A file whose processing raises is reported with its full traceback and skipped;
the rest of the run continues. With a plate of a hundred wells, one unreadable
file should not cost the other ninety-nine. Check the results:

.. code-block:: python

   failed = [r for r in results if not r['success']]
   for result in failed:
       print(result['file'], result['error'])


Ingestion modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Selects files by
   * - ``keywords`` + ``directory``
     - Substring match on the file names in ``directory``. Convenient for a
       directory that *is* the dataset.
   * - ``overview_df_based_processing=True``
     - The names in ``overview_df[overview_df_experiment_column]``. The right
       mode when the overview sheet, not the directory listing, defines what
       belongs to the dataset. If ``directory`` is also given, relative names
       are resolved against it — which is how the Streamlit uploader stages
       browser uploads in a temporary directory.

Both skip experiments already flagged ``Processed`` in the overview sheet, so
re-running an ingestion adds only what is new.


Getting data out
----------------

.. code-block:: python

   dataset = ExperimentalDataset.load_from_hdf5('experiments.h5')

   dataset.print_experiments()
   dataset.list_experiments()          # sorted names
   dataset.describe_version()          # one-line provenance summary

   experiment = dataset.experiments['NB-316']
   experiment.metadata['ru_concentration_uM']
   experiment.processed_data['max_rate']

Selecting subsets by metadata:

.. code-block:: python

   from pyKES.utilities.get_experiments import (get_experiments_by_metadata,
                                                get_unique_metadata_values)

   intensity_series = get_experiments_by_metadata(dataset.experiments,
                                                  type='intensity')

   intensities = get_unique_metadata_values(intensity_series, 'intensity')

Merging datasets:

.. code-block:: python

   merged = ExperimentalDataset.merge_hdf5_files(['run_a.h5', 'run_b.h5'],
                                                 output_filename='merged.h5')

Duplicate experiment names are reported and skipped rather than overwritten,
and the merged dataset records which files it came from under
``version['merged_from']``.


What HDF5 will store
--------------------

Nested dictionaries become nested groups, so ``processed_data`` may be as
deeply structured as the processing function likes. The value types fall back in
order:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Value
     - Stored as
   * - NumPy array
     - A native dataset. Efficient and portable.
   * - ``str``, ``int``, ``float``, ``bool``
     - A scalar dataset.
   * - ``list``, ``tuple``
     - Converted to an array where possible, JSON otherwise.
   * - Anything else
     - JSON, and failing that a pickle.

The fallbacks exist because a dataset that cannot be written is worse than one
written inefficiently — but a pickled entry is only readable by a compatible
Python, so prefer arrays and plain types in a processing function's output.

.. note::

   Keys containing ``'/'`` are escaped on write and restored on read, because
   HDF5 would otherwise read them as path separators. Metadata columns such as
   ``'Catalyst loading [wt% Rh/Cr]'`` make this a real case rather than a
   theoretical one.


Reprocessing
------------

The reason the raw data is kept:

.. code-block:: python

   from pyKES.database.data_processing import reprocess_experiments

   dataset = ExperimentalDataset.load_from_hdf5('experiments.h5')

   results = reprocess_experiments(
       database=dataset,
       processing_function=improved_processing_function,
       metadata_retrival_function=metadata_retrival_function)  # optional

   dataset.save_to_hdf5('experiments_v2.h5')

``processed_data`` is rebuilt from the stored raw data; ``metadata`` is either
reused as stored or refreshed from ``overview_df`` when a metadata function is
given — which is how an edited overview sheet takes effect. An experiment whose
processing raises **keeps its previous results**, so a partial failure leaves a
usable file. See :doc:`/versioning_and_reprocessing`.


Provenance
----------

Every save stamps the dataset, and every processing run stamps the experiments
it touched. Applications embedding pyKES should record their own version too:

.. code-block:: python

   from pyKES.utilities.version_information import get_project_version

   dataset.set_external_version({'app': 'photocat',
                                 'version': get_project_version(__file__)})

so that a file six months later says not only which pyKES produced it, but
which release of the processing code did.


Reference
---------

* :mod:`pyKES.database.database_experiments` — the dataset and experiment
  classes, and the HDF5 layer.
* :mod:`pyKES.database.data_processing` — ingestion and reprocessing.
* :doc:`/versioning_and_reprocessing` — the provenance model in detail.
