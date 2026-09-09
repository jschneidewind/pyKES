Architecture and code flow
==========================

pyKES is built around one idea: a mechanism is *text*, an experiment is *data*,
and everything else is a transformation between the two. This page shows the
transformations and how they connect, so that the rest of the guide can be read
in any order.


The two halves of the package
-----------------------------

There is a modelling half and a data half, and they meet in the fitting module.

.. mermaid::

   flowchart LR
       subgraph modelling["Modelling"]
           direction TB
           M1["Mechanism<br/><i>list of strings</i>"]
           M2["Parsed network<br/><i>reactants, products, k</i>"]
           M3["ODE system<br/><i>dc/dt</i>"]
           M4["Concentration traces"]
           M5["Photon budget"]
           M1 --> M2 --> M3 --> M4 --> M5
       end

       subgraph data["Experimental data"]
           direction TB
           D1["Instrument files<br/>+ overview sheet"]
           D2["Experiment<br/><i>raw + metadata</i>"]
           D3["processed_data<br/><i>rates, traces</i>"]
           D4["ExperimentalDataset<br/><i>HDF5</i>"]
           D1 --> D2 --> D3 --> D4
       end

       FIT["Fitting_Model<br/><i>rate constants</i>"]

       M3 --> FIT
       D4 --> FIT
       FIT --> M4

       style FIT fill:#ffe6cc,stroke:#d79b00,stroke-width:2px

The modelling half needs rate constants it does not have; the data half has
measurements it cannot interpret. ``Fitting_Model`` closes the loop: it
integrates the network once per experiment, compares against the measured
trace, and searches the rate constants that minimize the disagreement.


From mechanism text to a solution
---------------------------------

Everything in the modelling half starts with
:func:`~pyKES.reaction_ODE.parse_reactions`.

.. mermaid::

   flowchart TD
       A["'2 [RuIII] > [Ru-Dimer], k3'"]
       B["parse_reactions"]
       C["{'reactants': {'[RuIII]': 2.0},<br/>&nbsp;'products': {'[Ru-Dimer]': 1.0},<br/>&nbsp;'rate_constant': 'k3',<br/>&nbsp;'other_multipliers': []}"]
       D["build_ode_system"]
       E["ode_system(y, t) → dy/dt"]
       F["solve_ode_system<br/><i>scipy odeint</i>"]
       G["solution<br/><i>(times × species)</i>"]

       A --> B --> C --> D --> E --> F --> G

       H["calculate_reaction_rate"]
       I["resolve_other_multipliers"]
       E -.->|per reaction,<br/>per step| H
       H -.->|per multiplier| I
       I -.->|"e.g. light absorption"| H

       style I fill:#e1f5e1,stroke:#5a9,stroke-width:2px

The dashed path is what distinguishes pyKES from a generic ODE solver. A
multiplier may be a plain number, or a *function specification* whose arguments
are resolved against the concentrations at the current integration step. A
light-absorption term therefore sees the instantaneous composition of the
sample, and competitive absorption between species is handled without writing
it into the rate law by hand. See :doc:`light_absorption`.


From instrument files to a dataset
----------------------------------

The data half is deliberately generic: pyKES does not know how to read your
instrument. You supply three callables, and pyKES supplies the loop, the error
handling, the storage and the provenance.

.. mermaid::

   flowchart TD
       OV["Overview sheet<br/><i>Excel</i>"]
       RAW["Raw instrument files"]

       MD["metadata_retrival_function<br/><i>yours</i>"]
       RD["raw_data_reading_function<br/><i>yours</i>"]
       PF["processing_function<br/><i>yours</i>"]

       EXP["Experiment<br/>metadata · raw_data · processed_data · version"]
       DS["ExperimentalDataset"]
       H5[("HDF5 file")]

       OV --> MD
       RAW --> RD
       MD --> RD --> PF --> EXP --> DS --> H5
       MD --> EXP

       H5 -.->|"reprocess_experiments"| PF

       style MD fill:#ffe6cc,stroke:#d79b00
       style RD fill:#ffe6cc,stroke:#d79b00
       style PF fill:#ffe6cc,stroke:#d79b00

The dashed arrow is the reason the raw data is stored in the file rather than
discarded after processing: an improved algorithm can be applied to a finished
dataset without going back to the original instrument files. See
:doc:`dataset` and :doc:`/versioning_and_reprocessing`.

Three entry points drive that pipeline, differing only in *how* the loop is
run:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Entry point
     - Concurrency
     - Use when
   * - :func:`~pyKES.database.data_processing.read_in_experiments_multiprocessing`
     - One process per core
     - Bulk ingestion from a script. The callables must be importable at module
       level.
   * - :func:`~pyKES.database.data_processing.read_in_experiments_single_threaded`
     - Sequential
     - Progress reporting matters, or subprocesses are unavailable.
   * - :func:`~pyKES.database.data_processing.ingest_experiment`
     - One experiment per call
     - The caller drives the loop — which is what the Streamlit page does, one
       experiment per rerun. See :doc:`/browser_deployment`.


Module map
----------

.. mermaid::

   flowchart TB
       subgraph core["Core modelling"]
           RO["reaction_ODE"]
           RM["reaction_model"]
           FO["fitting_ODE"]
       end

       subgraph pw["pathways"]
           P1["pathways"]
           P2["transform_pathways_data"]
       end

       subgraph db["database"]
           DB1["database_experiments"]
           DB2["data_processing"]
       end

       subgraph util["utilities"]
           U1["calculate_absorption"]
           U2["max_rate"]
           U3["unit_handler"]
           U4["resolve_attributes"]
           U5["version_information"]
       end

       subgraph app["streamlit_app"]
           S1["config_interface"]
           S2["components"]
           S3["chunked_processing"]
       end

       subgraph plot["plotting"]
           PL1["plotting_pathways_transformed"]
           PL2["plotting_tools"]
       end

       RM --> RO
       RM --> P1
       RM --> P2
       FO --> RO
       FO --> DB1
       FO --> U4
       RO --> U1
       P1 --> RO
       P1 --> U1
       P2 --> PL1
       DB2 --> DB1
       DB1 --> U5
       U2 --> U3
       S2 --> DB2
       S2 --> S1
       S2 --> S3
       S2 --> PL2

The dependency direction never reverses: utilities know nothing about the
database, the database knows nothing about the Streamlit pages, and the
Streamlit pages know nothing about the repository that embeds them.


Two invariants worth knowing
----------------------------

**One dataset, mutated in place.** The Streamlit pages keep exactly one
:class:`~pyKES.database.database_experiments.ExperimentalDataset` in
``st.session_state.experimental_dataset``, and every page mutates that object
rather than passing copies around. A page that adds experiments and a page that
plots them are looking at the same object.

**Configuration is the extension point.** An external repository embedding the
Streamlit pages does not fork them; it constructs the dataclasses in
:mod:`pyKES.streamlit_app.config_interface` and passes them in. Adding a
capability should mean adding a field to a config dataclass, not editing a
component. See :doc:`streamlit_app`.


Reading order
-------------

.. mermaid::

   flowchart LR
       A["Reaction networks"] --> B["Light absorption"]
       B --> C["Pathway analysis"]
       A --> D["Fitting"]
       E["Datasets"] --> D
       E --> F["Maximum rates"]
       G["Units"] --> F

For simulation work follow the top path — :doc:`reaction_networks`,
:doc:`light_absorption`, :doc:`pathways`. For data-analysis work start at
:doc:`dataset`, then :doc:`/max_rate` and :doc:`fitting`. :doc:`units`
underpins both.
