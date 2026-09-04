The Streamlit application
=========================

pyKES ships a set of reusable Streamlit pages. They are meant to be **embedded
and configured, not forked**: an external repository supplies its own
processing functions and branding through dataclasses, and imports the pages
unchanged. When pyKES improves a page, every embedding repository gets the
improvement with a version bump.


The five pages
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Does
   * - :func:`~pyKES.streamlit_app.components.render_home`
     - Loads an existing HDF5 dataset, or starts a fresh one. Shows the
       dataset's provenance.
   * - :func:`~pyKES.streamlit_app.components.render_data_upload`
     - Uploads the metadata sheet and the raw data files, runs the processing
       pipeline, reprocesses existing experiments, merges datasets, and
       downloads the result.
   * - :func:`~pyKES.streamlit_app.components.render_analysis_results`
     - Plots derived results against a chosen metadata axis, with grouping and
       subset selection.
   * - :func:`~pyKES.streamlit_app.components.render_time_series`
     - Plots the measured traces, with multiple y axes when the curves carry
       different units. See :doc:`/plotting_instructions`.
   * - :func:`~pyKES.streamlit_app.components.render_results_table`
     - Tabulates numerical results across experiments, in configurable units
       and formats.


Wiring up an application
------------------------

A complete embedding repository is four files. The full version is in
:source:`examples/external_repo <examples/external_repo>`.

**config.py** — everything the application customizes:

.. code-block:: python

   from pyKES.streamlit_app.config_interface import (
       DataUploadConfig, FileUploadHandler, HomeConfig, PyKESStreamlitConfig)
   from pyKES.utilities.version_information import get_project_version

   from my_app.processing import (metadata_retrival_function,
                                  raw_data_reading_function,
                                  processing_function)

   raw_data_handler = FileUploadHandler(
       label='📊 Upload Raw Data',
       file_type=['csv', 'txt'],
       help_text='Raw data files referenced from the metadata sheet.',
       overview_df_experiment_column='Experiment',
       metadata_retrival_function=metadata_retrival_function,
       raw_data_reading_function=raw_data_reading_function,
       processing_function=processing_function)

   DATA_UPLOAD_CONFIG = DataUploadConfig(
       file_handlers=[raw_data_handler],
       page_title='Data Upload & Processing',
       metadata_excel_experiment_column='Experiment',
       group_mapping=GROUP_MAPPING,
       plotting_instruction=PLOTTING_INSTRUCTIONS,
       processing_parameters=PROCESSING_PARAMETERS,
       external_version={'app': 'photocat',
                         'version': get_project_version(__file__)})

   PYKES_CONFIG = PyKESStreamlitConfig(
       home_config=HomeConfig(main_title='Photocatalysis Data Portal'),
       data_upload_config=DATA_UPLOAD_CONFIG,
       app_title='Photocatalysis Data Analysis System',
       app_icon=':test_tube:')

**Home.py** and the pages — each delegates and does nothing else:

.. code-block:: python

   # Home.py
   from config import PYKES_CONFIG
   from pyKES.streamlit_app.components import render_home

   render_home(PYKES_CONFIG.home_config)

.. code-block:: python

   # pages/01_Data_Upload.py
   from config import PYKES_CONFIG
   from pyKES.streamlit_app.components import render_data_upload

   render_data_upload(PYKES_CONFIG.data_upload_config)

The remaining three pages take no configuration at all — they read what they
need from the dataset:

.. code-block:: python

   # pages/03_Time_Series.py
   from pyKES.streamlit_app.components import render_time_series

   render_time_series()

Run it with:

.. code-block:: bash

   streamlit run Home.py


The single-dataset invariant
----------------------------

.. mermaid::

   flowchart TB
       SS["st.session_state.experimental_dataset<br/><i>one ExperimentalDataset</i>"]

       H["Home<br/><i>loads / creates</i>"]
       DU["Data Upload<br/><i>adds experiments</i>"]
       AR["Analysis Results<br/><i>reads</i>"]
       TS["Time Series<br/><i>reads</i>"]
       RT["Results Table<br/><i>reads</i>"]

       H --> SS
       DU --> SS
       SS --> AR
       SS --> TS
       SS --> RT

       style SS fill:#ffe6cc,stroke:#d79b00,stroke-width:2px

Every page reads and mutates the *same* object. A page that adds experiments
and a page that plots them are looking at the same dataset, so nothing has to
be passed between pages and there is no copy to fall out of date. A new page
should follow the same rule: read
``st.session_state.experimental_dataset``, mutate it in place.


Configuration is the extension point
------------------------------------

The rule that keeps the pages reusable: **adding a capability should mean
adding a field to a config dataclass, not editing a component.** If a change to
a page can only be expressed by editing the page, that is a signal the config
interface is missing something — and the fix belongs in pyKES, so every
embedding repository benefits.

The four dataclasses:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Dataclass
     - Configures
   * - :class:`~pyKES.streamlit_app.config_interface.FileUploadHandler`
     - One uploader and, optionally, the three-callable pipeline behind it. A
       handler is "processing-enabled" only when all three callables are given;
       one without them uploads files but produces no experiments.
   * - :class:`~pyKES.streamlit_app.config_interface.DataUploadConfig`
     - The Data Upload page: its handlers, its wording, the metadata sheet's
       name and key column, the defaults for a fresh dataset, and the
       application's own version stamp.
   * - :class:`~pyKES.streamlit_app.config_interface.HomeConfig`
     - Home page branding and wording.
   * - :class:`~pyKES.streamlit_app.config_interface.PyKESStreamlitConfig`
     - The two above, plus the application title and icon.

Two more pieces of configuration live on the dataset rather than in these
dataclasses, because they describe the *data* rather than the application:
``plotting_instruction`` (see :doc:`/plotting_instructions`) and
``group_mapping``. The Data Upload config seeds them onto a freshly created
dataset; thereafter they travel with the HDF5 file.


Long-running work must be chunked
---------------------------------

The pages are also deployed as a **static browser page** via stlite, which runs
Streamlit inside Pyodide. There, Python, Streamlit and the UI share the
browser's single event loop, with no script-runner thread — so a loop that
processes every experiment inside one script run delivers nothing to the
screen until it has finished. A progress bar written inside such a loop is
invisible for the whole run and then replaced by the result.

:mod:`pyKES.streamlit_app.chunked_processing` is the answer: a job is
registered once, and advanced **one experiment per rerun** by a fragment on a
timer.

.. mermaid::

   flowchart LR
       START["start_chunked_job<br/><i>register</i>"]
       A1["advance_job<br/><i>one experiment</i>"]
       P1["render_job_progress<br/><i>draw the bar</i>"]
       DONE["finish_job<br/><i>stamp, clean up</i>"]

       START --> A1
       A1 -->|"run ends,<br/>elements delivered"| P1
       P1 -->|"run_every timer"| A1
       A1 -->|"last experiment"| DONE

Each run **ends normally**, which is what lets its elements reach the browser
before the next one starts. This is not a detail to work around: using
``st.rerun`` instead shows nothing at all in the browser, because Streamlit
clears the unflushed message queue at the start of every script run. Both
halves of the design were established by measuring the deployed page in
headless Chrome.

When adding a long-running section to a page, use this module rather than
looping inline, and use
:func:`~pyKES.streamlit_app.chunked_processing.any_active_job` to skip
sections too expensive to re-render on every step. The full account is in
:doc:`/browser_deployment`.

.. warning::

   :func:`~pyKES.database.data_processing.read_in_experiments_multiprocessing`
   cannot be used from the pages. Pyodide has no processes to fork, and the
   uploaded-file staging is not visible to subprocesses anyway. The upload page
   steps through
   :func:`~pyKES.database.data_processing.ingest_experiment` instead.


Adding a page
-------------

1. Write a component function in
   :mod:`pyKES.streamlit_app.components`, reading the dataset from
   ``st.session_state.experimental_dataset``.
2. Export it from ``components/__init__.py``.
3. If it needs configuration, add a dataclass or a field to
   :mod:`pyKES.streamlit_app.config_interface` — do not read module-level
   globals from the embedding repository.
4. If it does anything long-running, chunk it.
5. Add a delegating page file to
   :source:`examples/external_repo/pages <examples/external_repo/pages>` so the
   wiring stays demonstrated.


Reference
---------

* :mod:`pyKES.streamlit_app.config_interface` — the configuration dataclasses.
* :mod:`pyKES.streamlit_app.chunked_processing` — the chunked-job machinery.
* :doc:`/browser_deployment` — why chunking is necessary, in detail.
* :doc:`/plotting_instructions` — configuring the time-series page.
