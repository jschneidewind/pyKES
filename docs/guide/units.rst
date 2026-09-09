Units and quantities
====================

Kinetic data arrives in whatever unit the instrument happened to write. A
dissolved-oxygen logger reports µmol L\ :sup:`-1` every second, a hydrogen
sensor reports mL min\ :sup:`-1`, and the rate that goes into a paper is in
µmol h\ :sup:`-1`. Between those lie half a dozen conversion factors, none of
which is visible in a bare float.

:class:`~pyKES.utilities.unit_handler.quantity.Quantity` carries the unit along
with the number so the conversions happen where they can be checked.

.. note::

   ``Quantity`` is a small, self-contained replacement for Pint, written so
   that pyKES has no heavy unit dependency — which matters for the browser
   deployment, where every megabyte of wheel is a download. It supports what
   this package needs, not the whole of dimensional analysis.


Creating and converting
-----------------------

.. code-block:: python

   from pyKES.utilities.unit_handler import Quantity

   rate = Quantity(5.2e-5, 'mol / h')

   rate.unit['umol / s']       # 0.01444...
   rate.unit['mmol / min']     # 0.000866...
   rate.base_value             # 1.444e-8, in the base unit
   rate.base_unit              # 'mol / s'
   rate.dimension              # 'substance / time'

Conversion happens through the ``unit`` mapping, keyed by the unit you want.
The lookup is lazy and cached, so converting the same array repeatedly costs
nothing after the first time. Values may be scalars or NumPy arrays:

.. code-block:: python

   time = Quantity(np.arange(0, 8000, 1.0), 's')
   time.unit['h']              # the whole array, in hours

Composite expressions accept ``*``, ``/`` and parentheses:

.. code-block:: python

   Quantity(44, 'mW / cm2')
   Quantity(1.5, '(kWh * m) / m2')
   Quantity(2.4, '1 / month')

A unit token that is not in the configuration raises ``ValueError`` on
construction rather than producing a silently wrong number.

Requesting a unit of the wrong dimension is also an error, which is the point:

.. code-block:: python

   rate.unit['kg']
   # ValueError: dimension mismatch


Supported dimensions
--------------------

Defined in :mod:`pyKES.utilities.unit_handler.config`:

.. hlist::
   :columns: 3

   * energy
   * power
   * length
   * area
   * volume
   * time
   * mass
   * substance
   * temperature_diff
   * absolute_temperature
   * voltage
   * current
   * charge
   * resistance
   * pressure
   * force
   * frequency
   * angle
   * currency
   * dimensionless

Adding a unit means adding an entry to the ``conversions`` dictionary of the
relevant dimension — the multiplier that converts it to that dimension's base
unit.

.. note::

   **Absolute temperature is a special case.** Kelvin and degrees Celsius
   differ by an offset, not a factor, so they are handled separately from every
   other dimension and cannot appear inside a composite expression. A
   *difference* of temperatures is a different dimension —
   ``temperature_diff`` — and does compose normally.


Reference labels
----------------

A quantity may carry descriptive labels saying *what* the unit refers to. Two
rates both in ``mol / s`` are not interchangeable if one counts H\ :sub:`2` and
the other O\ :sub:`2`:

.. code-block:: python

   hydrogen_rate = Quantity(5.84e-8, 'mol[H2] / s')
   oxygen_rate = Quantity(2.92e-8, 'mol[O2] / s')

   hydrogen_rate
   # Quantity(5.84e-08, 'mol[H2] / s')

Labels are descriptive only — they never affect the arithmetic — but they
survive conversion and appear in the representation, so a stored result says
what it counted. Requesting a unit with a mismatched label raises, which
catches the case of reading an O\ :sub:`2` rate as if it were H\ :sub:`2`.

They can also be given as a separate list, one entry per unit token:

.. code-block:: python

   Quantity(1.5e5, 'J / kg', reference=['electricity', 'H2'])

Not both at once — bracketed labels and a ``reference`` argument together raise
``ValueError``.


Where quantities are required
-----------------------------

Two parts of pyKES take and return ``Quantity`` objects rather than floats:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Module
     - Contract
   * - :mod:`pyKES.utilities.max_rate`
     - ``time`` must have dimension *time*, ``values`` dimension *substance*.
       Anything else raises immediately — there is no "assume seconds"
       fallback. Every physical field of the result comes back as a
       ``Quantity``.
   * - :mod:`pyKES.utilities.calculate_efficiency`
     - Every argument and the result. Efficiency formulas mix wavelengths,
       irradiances, areas and rates, each habitually reported in a different
       unit.

Everything else — the ODE modules, the absorption functions, the fitting code —
works on plain floats in units the caller keeps track of. See
:doc:`light_absorption` for the units those expect.

.. code-block:: python

   from pyKES.utilities.calculate_efficiency import calculate_apparent_quantum_yield

   aqy = calculate_apparent_quantum_yield(
       irradiation_wavelength=Quantity(365, 'nm'),
       irradiation_area=Quantity(2.5, 'cm2'),
       irradiance_power=Quantity(44, 'mW/cm2'),
       reaction_rate=Quantity(5.2e-5, 'mol[O2]/h'),
       fraction_of_photons_reaching_inside=Quantity(0.86, '-'),
       electron_transfer_per_reaction=4)

   print(f'AQY: {aqy.unit["%"]:.2f}%')

Six quantities in six different units, and no conversion factor written by
hand.


Storing quantities
------------------

A ``Quantity`` in ``processed_data`` is stored by the HDF5 layer through its
JSON or pickle fallback, which works but is neither compact nor portable. For
results meant to be read back years later, store the magnitude in an explicit
unit and the unit string beside it:

.. code-block:: python

   return {'max_rate': result.max_rate.unit['umol / h'],
           'max_rate_unit': 'umol / h'}

The Streamlit results table takes the same approach: its configuration names
the unit each column is displayed in, and the conversion happens at render
time.


Reference
---------

* :mod:`pyKES.utilities.unit_handler.quantity` — the class and its parsing.
* :mod:`pyKES.utilities.unit_handler.config` — the dimensions and conversions.
* :doc:`/max_rate` — the main consumer.
