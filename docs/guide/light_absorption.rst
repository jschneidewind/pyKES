Light absorption and competing chromophores
===========================================

A thermal rate law needs concentrations. A photochemical one needs to know how
many photons a species actually caught — which is not a property of that
species alone, but of everything else in the cuvette that was competing for the
same light. :mod:`pyKES.utilities.calculate_absorption` computes that quantity,
and :doc:`reaction_networks` shows how to plug it into a mechanism.


Why a constant will not do
--------------------------

Consider Ru(bpy)\ :sub:`3`\ :sup:`2+` being oxidized to Ru(bpy)\ :sub:`3`\
:sup:`3+` during a photocatalytic run. The two absorb very differently at the
irradiation wavelength — :math:`\varepsilon \approx 8500` against
:math:`540\ \mathrm{M^{-1}cm^{-1}}` — so as the reaction proceeds:

* the total absorbance of the sample falls, and more light is transmitted
  unused;
* of the light still absorbed, a growing share goes to the Ru(III) that cannot
  use it productively;
* so the *per-molecule* excitation rate of Ru(II) changes even at constant lamp
  power.

Writing the excitation rate as a constant makes the model wrong in a way that
is easy to miss: it fits fine over a short window and drifts systematically
over a long one.


What the functions compute
--------------------------

Two steps, in every variant.

**How much light is absorbed at all** — the Beer–Lambert law over the total
absorbance of the mixture:

.. math::

   A_{\text{tot}} = \ell \sum_i c_i \varepsilon_i
   \qquad
   f_{\text{abs}} = 1 - 10^{-A_{\text{tot}}}

**How that light is divided** — in proportion to each species' contribution to
the total absorbance:

.. math::

   f_i = \frac{c_i \varepsilon_i}{\sum_j c_j \varepsilon_j}

The excitations per molecule of species *i* per second then follow by
normalizing the absorbed photons to the amount of that species present:

.. math::

   \text{excitations}_i = \frac{\Phi \, f_{\text{abs}} \, f_i}{V c_i}

with :math:`\Phi` the photon flux in mol s\ :sup:`-1` and *V* the illuminated
volume. The result is a per-molecule rate, which is exactly what multiplies a
first-order rate constant in the mechanism.

.. mermaid::

   flowchart LR
       C["concentrations<br/>at this step"] --> AB["total absorbance<br/>ℓ Σ cᵢ εᵢ"]
       E["extinction<br/>coefficients"] --> AB
       AB --> FA["absorbed fraction<br/>1 − 10⁻ᴬ"]
       AB --> FI["share per species<br/>cᵢεᵢ / Σ cⱼεⱼ"]
       PF["photon flux"] --> EX
       FA --> EX["excitations per<br/>molecule per second"]
       FI --> EX
       C --> EX

       style EX fill:#e1f5e1,stroke:#5a9,stroke-width:2px

The two limits behave as they should. At low absorbance,
:math:`f_{\text{abs}} \to A_{\text{tot}}`, the volume terms cancel and the
excitation rate becomes independent of concentration — the optically thin case.
At high absorbance, :math:`f_{\text{abs}} \to 1`, all light is caught and adding
more chromophore only dilutes the photons over more molecules.


Choosing a variant
------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Function
     - Use when
   * - :func:`~pyKES.utilities.calculate_absorption.calculate_excitations_per_second`
     - A single absorbing species. No competition term at all.
   * - :func:`~pyKES.utilities.calculate_absorption.calculate_excitations_per_second_competing`
     - Exactly two species share the light. Positional arguments ``A`` and
       ``B``; ``A`` is the one being excited.
   * - :func:`~pyKES.utilities.calculate_absorption.calculate_excitations_per_second_multi_competing`
     - Any number of species. NumPy implementation; can also return the full
       per-species absorption breakdown via ``return_full=True``, which is what
       the pathway analysis uses.
   * - :func:`~pyKES.utilities.calculate_absorption.calculate_excitations_per_second_multi_competing_fast`
     - Any number of species, called from inside an ODE solve. Same result,
       pure-Python dictionaries, no NumPy overhead.

.. important::

   Inside an integration, use the ``_fast`` variant. It is called once per
   light-driven reaction per integration step — of the order of 10\ :sup:`4`
   times per solve and 10\ :sup:`7` times over a fit — and at these array sizes
   NumPy's per-call overhead dominates the actual arithmetic. The two agree to
   floating-point precision; :func:`~pyKES.utilities.calculate_absorption.test_function`
   benchmarks them against each other.


Wiring it into a mechanism
--------------------------

The multiplier specification names, for each parameter of the absorption
function, where its value comes from:

.. code-block:: python

   from pyKES.utilities.calculate_absorption import (
       calculate_excitations_per_second_multi_competing_fast)

   other_multipliers = {
       'photon_flux': 1e17,               # photons cm^-2 s^-1
       'pathlength': 2.25,                # cm
       'Ru_II_extinction_coefficient': 8500,
       'Ru_III_extinction_coefficient': 540,

       # A literal string, passed through unresolved — it names a species
       # rather than pointing at a value.
       'hv_functionA_species_of_interest': '[RuII]',

       'hv_functionA': {
           'function': calculate_excitations_per_second_multi_competing_fast,
           'arguments': {
               'photon_flux': 'photon_flux',
               'concentration_[RuII]': '[RuII]',      # ← current concentration
               'concentration_[RuIII]': '[RuIII]',    # ← current concentration
               'extinction_coefficient_[RuII]': 'Ru_II_extinction_coefficient',
               'extinction_coefficient_[RuIII]': 'Ru_III_extinction_coefficient',
               'pathlength': 'pathlength',
               'species_of_interest': 'hv_functionA_species_of_interest'}}}

and the reaction references it after a semicolon:

.. code-block:: python

   '[RuII] > [RuII-ex], k1 ; hv_functionA'

The rate of that step is then ``k1 × excitations × [RuII]``. Since
``excitations`` is already per-molecule, ``k1`` is the quantum yield of the
excitation event — dimensionless and usually close to 1.

The ``concentration_<name>`` / ``extinction_coefficient_<name>`` prefixes are
how the multi-species variants accept an arbitrary number of chromophores
through a flat argument mapping. Every species meant to compete for the light
has to appear in **both**; one listed with a concentration but no extinction
coefficient contributes nothing and is silently absent from the competition.

A second light-driven step reuses everything and changes one line:

.. code-block:: python

   'hv_functionB_species_of_interest': '[RuIII]',
   'hv_functionB': {
       'function': calculate_excitations_per_second_multi_competing_fast,
       'arguments': {**shared_arguments,
                     'species_of_interest': 'hv_functionB_species_of_interest'}}


Units
-----

These functions take **plain floats in fixed units**, unlike
:mod:`pyKES.utilities.max_rate`, which takes
:class:`~pyKES.utilities.unit_handler.quantity.Quantity` objects. Nothing
checks them, so they are worth stating:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Quantity
     - Unit
   * - ``photon_flux``
     - photons cm\ :sup:`-2` s\ :sup:`-1`
   * - ``concentrations``
     - µM by default; pass ``concentration_unit='M'`` (or another unit) to the
       multi-species variants to change it. **The unit must match the one the
       mechanism's concentrations are in.**
   * - ``extinction_coefficients``
     - M\ :sup:`-1` cm\ :sup:`-1`
   * - ``pathlength``
     - cm
   * - result
     - excitations per molecule per second

A unit mismatch between the mechanism's concentrations and
``concentration_unit`` is the most likely cause of an excitation rate that is
off by a clean factor of 10\ :sup:`6`.


Sanity checks
-------------

Two quick ones before trusting a photochemical fit.

**Does the absorbed fraction make sense?** Ask for the full breakdown:

.. code-block:: python

   from pyKES.utilities.calculate_absorption import (
       calculate_excitations_per_second_multi_competing)

   excitations, absorbed = calculate_excitations_per_second_multi_competing(
       species_of_interest='[RuII]',
       photon_flux=1e17,
       concentrations={'[RuII]': 10.0, '[RuIII]': 2.0},
       extinction_coefficients={'[RuII]': 8500, '[RuIII]': 540},
       pathlength=2.25,
       return_full=True)

   absorbed
   # {'[RuII]': 0.38..., '[RuIII]': 0.005..., 'transmitted': 0.61...}

The entries sum to 1. If ``transmitted`` is near 1 the sample barely absorbs;
if it is near 0 the sample is optically dense and adding chromophore will not
speed anything up.

**Is the excitation rate physically plausible?** For a 10 µM sensitizer at
10\ :sup:`17` photons cm\ :sup:`-2` s\ :sup:`-1`, expect of the order of one
excitation per molecule per second. Several orders of magnitude away from that
usually means a unit mismatch rather than unusual photophysics.


Reference
---------

* :mod:`pyKES.utilities.calculate_absorption` — all four variants.
* :doc:`pathways` — the same absorption calculation, used to trace where the
  absorbed photons end up.
* :mod:`pyKES.utilities.calculate_efficiency` — apparent quantum yield and
  light-to-hydrogen efficiency from measured rates.
