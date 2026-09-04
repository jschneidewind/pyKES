Utilities (``pyKES.utilities``)
===============================

Small, focused modules the rest of the package builds on.

Light absorption
----------------

Competitive absorption in a mixture of chromophores, used as a multiplier
inside a reaction network. See :doc:`/guide/light_absorption`.

.. automodule:: pyKES.utilities.calculate_absorption
   :members:

Maximum rates
-------------

Robust extraction of maximum rates from noisy kinetic traces. The algorithm is
described stage by stage in :doc:`/max_rate`.

.. automodule:: pyKES.utilities.max_rate
   :members:

Efficiencies
------------

Apparent quantum yield and light-to-hydrogen efficiency.

.. automodule:: pyKES.utilities.calculate_efficiency
   :members:

Units
-----

The lightweight quantity type. See :doc:`/guide/units`.

.. automodule:: pyKES.utilities.unit_handler.quantity
   :members:

.. automodule:: pyKES.utilities.unit_handler.config
   :members:

Attribute resolution
--------------------

Resolving slash-separated paths into experiment objects, which is what lets one
fitting model cover a whole dataset. See :doc:`/guide/fitting`.

.. automodule:: pyKES.utilities.resolve_attributes
   :members:

Version information
-------------------

Provenance stamps recorded in every dataset. See
:doc:`/versioning_and_reprocessing`.

.. automodule:: pyKES.utilities.version_information
   :members:

Time-series helpers
-------------------

.. automodule:: pyKES.utilities.harmonize_time_series
   :members:

.. automodule:: pyKES.utilities.time_series_resampling
   :members:

.. automodule:: pyKES.utilities.offset_correction
   :members:

.. automodule:: pyKES.utilities.find_nearest
   :members:

Dataset helpers
---------------

.. automodule:: pyKES.utilities.get_experiments
   :members:

.. automodule:: pyKES.utilities.make_json_serializable
   :members:
