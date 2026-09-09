Reaction networks and ODE simulation
====================================

A reaction network in pyKES is written the way it is written on a whiteboard,
as a list of strings. :mod:`pyKES.reaction_ODE` turns that list into a coupled
system of ordinary differential equations and integrates it.


The mechanism syntax
--------------------

One reaction per string::

    reactants  >  products ,  rate_constant  ;  multiplier, multiplier, ...

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Element
     - Rule
   * - Species
     - Enclosed in square brackets: ``[RuII]``, ``[Ru-Dimer]``, ``[H2O2]``.
       Anything is allowed inside them except a closing bracket, so
       ``[RuII-ex]`` and ``[Catalyst (oxidized)]`` are both fine.
   * - Stoichiometry
     - An integer or decimal *in front of* the species, separated by a space:
       ``2 [RuIII]``, ``0.5 [O2]``. Omitted means 1.
   * - Direction
     - A single ``>``. Reversible steps are written as two reactions, which is
       what lets them carry different multipliers.
   * - Rate constant
     - An identifier after the comma. It is looked up in ``rate_constants``;
       the same identifier may appear in several reactions.
   * - Multipliers
     - Optional, after a semicolon, comma-separated. Each is looked up in
       ``other_multipliers`` and multiplied into the rate.

A worked mechanism — Ru(bpy)\ :sub:`3`-photosensitized water oxidation:

.. code-block:: python

   reactions = ['[RuII] > [RuII-ex], k1 ; hv_functionA',
                '[RuII-ex] > [RuII], k8',
                '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
                '[RuIII] > [H2O2] + [RuII], k2 ; hv_functionB',
                '2 [RuIII] > [Ru-Dimer], k3',
                '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
                '[H2O2] > [O2], k5',
                '[RuIII] > [Inactive], k6']

Read top to bottom: Ru(II) is excited (``k1``, light-driven), the excited state
either decays (``k8``) or is oxidatively quenched by persulfate to Ru(III)
(``k7``). Ru(III) turns over to peroxide and back to Ru(II) (``k2``, also
light-driven), and peroxide releases O\ :sub:`2` (``k5``). The last three
reactions are the loss channels: dimerization (``k3``), autocatalytic loss on
the dimer (``k4``) and plain decomposition (``k6``).


What parsing produces
---------------------

:func:`~pyKES.reaction_ODE.parse_reactions` returns the parsed reactions and
the sorted species list:

.. code-block:: python

   from pyKES.reaction_ODE import parse_reactions

   parsed_reactions, species = parse_reactions(['[A] + 2 [B] > [C], k1',
                                                '[C] > [A] + [B], k2 ; hv1'])

   parsed_reactions[0]
   # {'reactants': {'[A]': 1.0, '[B]': 2.0},
   #  'products': {'[C]': 1.0},
   #  'rate_constant': 'k1',
   #  'other_multipliers': []}

   species
   # ['[A]', '[B]', '[C]']

The species list matters beyond bookkeeping: **its order is the order of the
state vector**, and therefore the column order of every solution array. Column
``i`` of the solution is the trace of ``species[i]``. Because the list is
sorted alphabetically, adding a species to a mechanism can shift the columns —
always index through ``species.index(...)`` rather than by a remembered number.


The rate law
------------

Each reaction gets a mass-action rate: the rate constant, times every
multiplier, times each reactant concentration raised to its stoichiometric
coefficient.

.. math::

   r_j \;=\; k_j \;\cdot\; \prod_{m \in M_j} \mu_m \;\cdot\; \prod_{i} c_i^{\,\nu_{ij}}

That rate then enters the derivative of every participating species, negatively
for reactants and positively for products:

.. math::

   \frac{\mathrm{d}c_i}{\mathrm{d}t} \;=\; \sum_j \left( \nu^{\text{prod}}_{ij} - \nu^{\text{react}}_{ij} \right) r_j

So ``2 [RuIII] > [Ru-Dimer], k3`` contributes :math:`-2 k_3 [\mathrm{RuIII}]^2`
to Ru(III) and :math:`+k_3 [\mathrm{RuIII}]^2` to the dimer. The stoichiometric
coefficient appears twice — once as the reaction order, once as the amount
consumed — and pyKES applies both, which is the usual place to make an
arithmetic slip by hand.

.. note::

   The rate law is mass action with the stoichiometric coefficients as orders.
   A step whose observed order differs from its stoichiometry (a saturating
   catalytic step, say) is not expressible directly — write it as the
   elementary steps it decomposes into, or push the non-mass-action part into a
   multiplier function.


Multipliers: where light comes in
---------------------------------

An entry of ``other_multipliers`` is either a number or a **function
specification**:

.. code-block:: python

   other_multipliers = {
       # a plain number
       'photon_flux': 1e17,
       'pathlength': 2.25,

       # a function specification
       'hv_functionA': {
           'function': calculate_excitations_per_second_multi_competing_fast,
           'arguments': {
               'photon_flux': 'photon_flux',                    # → another multiplier
               'concentration_[RuII]': '[RuII]',                # → a concentration
               'extinction_coefficient_[RuII]': 'Ru_II_epsilon',
               'pathlength': 'pathlength',
               'species_of_interest': 'hv_functionA_species'}}}

Every value in ``arguments`` is a *name*, resolved at each integration step
against — in this order — the current concentrations, the other (non-function)
multipliers, and the rate constants. A name that resolves nowhere raises
``KeyError`` rather than defaulting silently.

This is what makes photochemistry work properly. The excitation rate of Ru(II)
depends on how much of the light Ru(III) took first, and that ratio changes
throughout the run; because the arguments are resolved per step, the multiplier
tracks it. See :doc:`light_absorption`.

.. warning::

   A multiplier function is called once per reaction per integration step —
   tens of thousands of times per solve, and millions of times during a fit.
   Keep it cheap. This is why
   :func:`~pyKES.utilities.calculate_absorption.calculate_excitations_per_second_multi_competing_fast`
   exists alongside the NumPy version: at these array sizes, NumPy's per-call
   overhead dominates.


Solving
-------

.. code-block:: python

   import numpy as np
   from pyKES.reaction_ODE import parse_reactions, solve_ode_system, plot_solution

   parsed_reactions, species = parse_reactions(reactions)

   solution = solve_ode_system(parsed_reactions,
                               species,
                               rate_constants,
                               initial_conditions={'[RuII]': 10, '[S2O8]': 6000},
                               times=np.linspace(0, 300, 1000),
                               other_multipliers=other_multipliers)

   plot_solution(species, times, solution, exclude_species=['[S2O8]', '[SO4]'])

.. image:: ../_static/images/reaction_network_simulation.png
   :alt: Simulated concentration-time traces of Ru(bpy)3-photosensitized water oxidation
   :align: center
   :width: 90%

The figure shows what the mechanism was built to explain: O\ :sub:`2` evolution
tails off long before the persulfate is exhausted, because the photosensitizer
is being consumed by the dimerization and decomposition channels.

Species absent from ``initial_conditions`` start at zero, so only the species
actually present at *t* = 0 need to be listed. An initial condition naming a
species that is not in the network is reported and ignored — worth watching for
in the console, since a typo there silently changes the simulated experiment.

.. note::

   Photochemical networks are stiff: excited-state decay happens on
   nanoseconds, the chemistry it feeds on seconds. ``solve_ode_system``
   therefore sets tolerances well below the ``odeint`` defaults
   (``rtol=1e-8``, ``atol=1e-10``) and raises the step limit to 5000. A run
   that still reports excess work has a genuinely extreme separation of time
   scales; the usual fix is to remove the fastest step from the mechanism by
   assuming it to be at steady state.


The object interface
--------------------

:class:`~pyKES.reaction_model.Reaction_Model` bundles the same thing into one
object and keeps every intermediate result as an attribute, which is what makes
it convenient in a notebook:

.. code-block:: python

   from pyKES.reaction_model import Reaction_Model

   model = Reaction_Model(reaction_network=reactions,
                          rate_constants=rate_constants,
                          initial_conditions={'[RuII]': 10, '[S2O8]': 6000},
                          other_multipliers=other_multipliers,
                          times=np.linspace(0, 300, 1000))

   model.solve_ode()
   model.plot_solution(exclude_species=['[S2O8]', '[SO4]'])

   oxygen = model.solution[:, model.species.index('[O2]')]

It also carries the pathway analysis of :doc:`pathways`, which the functional
interface does not.


Common pitfalls
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symptom
     - Cause
   * - ``KeyError`` on a species name
     - A species appears in ``initial_conditions`` or in a multiplier's
       ``arguments`` but is spelled differently in the mechanism. Brackets are
       part of the name: ``'[RuII]'``, not ``'RuII'``.
   * - Concentrations go negative
     - Almost always a tolerance problem on a stiff network. Check that the
       fastest rate constant is really needed at that magnitude.
   * - ``Warning: [X] not in species list``
     - An initial condition for a species the mechanism never mentions. The
       simulation runs without it.
   * - A multiplier has no effect
     - It was defined in ``other_multipliers`` but not referenced after a
       semicolon in the reaction that should use it.


Reference
---------

* :mod:`pyKES.reaction_ODE` — the functional interface.
* :mod:`pyKES.reaction_model` — the object interface.
