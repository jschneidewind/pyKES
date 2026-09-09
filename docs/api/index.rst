API reference
=============

Generated from the NumPy-style docstrings in the source. Every public function
and class of pyKES appears here; the :doc:`user guide </guide/architecture>`
explains how they fit together.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Module group
     - Contains
   * - :doc:`reaction_ODE`
     - Parsing mechanisms, building and solving the ODE system.
   * - :doc:`reaction_model`
     - The object interface to simulation and pathway analysis.
   * - :doc:`fitting_ODE`
     - Fitting rate constants to experimental data.
   * - :doc:`pathways`
     - Photon-budget propagation and its plotting layout.
   * - :doc:`database`
     - Datasets, HDF5 storage, ingestion and reprocessing.
   * - :doc:`utilities`
     - Absorption, maximum rates, units, efficiencies, path resolution,
       provenance and small helpers.
   * - :doc:`plotting`
     - Pathway diagrams and analysis-result figures.
   * - :doc:`streamlit_app`
     - Configuration dataclasses, page components and chunked processing.

.. toctree::
   :maxdepth: 2

   reaction_ODE
   reaction_model
   fitting_ODE
   pathways
   database
   utilities
   plotting
   streamlit_app
