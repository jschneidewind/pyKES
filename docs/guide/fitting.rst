Fitting rate constants to experimental data
===========================================

:mod:`pyKES.fitting_ODE` answers the inverse question of
:doc:`reaction_networks`: given a mechanism and a set of measurements, which
rate constants reproduce them?

The design goal is a fit across a **whole dataset at once**. A single curve
rarely constrains a mechanism — several rate constants can trade off against
each other and still fit it. Experiments run at different concentrations,
different light intensities and different durations break those correlations,
and a mechanism that reproduces all of them with one parameter set is a much
stronger claim than one that fits any of them alone.


How it works
------------

.. mermaid::

   flowchart TD
       OPT["differential_evolution<br/><i>proposes rate constants</i>"]
       OBJ["objective_function"]

       subgraph per["for each experiment"]
           direction TB
           RES["resolve_experiment_attributes<br/><i>paths → values</i>"]
           SOLVE["solve_ode_system"]
           LOSS["loss_function"]
           RES --> SOLVE --> LOSS
       end

       SUM["Σ weightᵢ × lossᵢ"]

       OPT -->|"trial vector"| OBJ
       OBJ --> per
       LOSS --> SUM
       SUM -->|"total error"| OPT

       style RES fill:#e1f5e1,stroke:#5a9,stroke-width:2px

The highlighted step is the one that makes a dataset-wide fit practical.


Conditions are paths, not numbers
---------------------------------

Every experiment was run under its own conditions. Rather than building one
model per experiment, the model declares *where to find* each condition, and
:func:`~pyKES.utilities.resolve_attributes.resolve_experiment_attributes`
resolves that path against each
:class:`~pyKES.database.database_experiments.Experiment` in turn:

.. code-block:: python

   model.initial_conditions = {
       '[S2O8]': 'metadata/oxidant_concentration_uM',
       '[RuII]': 'metadata/ru_concentration_uM'}

   model.other_multipliers = {
       'pathlength': 2.25,                        # a literal, same for all
       'photon_flux': 'metadata/photon_flux',     # a path, per experiment
       'Ru_II_extinction_coefficient': 8500}

   model.times = {'times': 'processed_data/time_reaction'}

   model.data_to_be_fitted = {
       '[O2]': {'x': 'processed_data/x_diff',
                'y': 'processed_data/y_diff'}}

A value containing ``/`` is treated as a path; anything else is passed through
unchanged. Paths follow attributes and dictionary keys alike, and cope with
keys that themselves contain ``/`` — ``'metadata/Catalyst loading [wt% Rh/Cr]'``
resolves as the two-part path it was meant to be. See :doc:`dataset`.

The resolution modes differ by field, deliberately:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Field
     - Mode
     - Why
   * - ``other_multipliers``, ``times``
     - ``strict``
     - Every entry must resolve. A missing photon flux is a mistake, not a
       measurement that happens to be absent.
   * - ``initial_conditions``, ``data_to_be_fitted``
     - ``semi-strict``
     - At least one entry must resolve. An experiment where one product was not
       measured still contributes the products that were.


Choosing what to compare
------------------------

The loss function decides *which* quantity the fit is judged on, and the choice
matters more than the optimizer settings.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Loss function
     - Compares
   * - :func:`~pyKES.fitting_ODE.square_loss_time_series`
     - The integrated trace, point by point. The obvious choice when the
       measurement really is an amount.
   * - :func:`~pyKES.fitting_ODE.square_loss_time_series_normalized`
     - The same, divided by the mean squared magnitude of the data. Use it when
       experiments differ in scale by orders of magnitude, so a large-signal
       experiment does not swamp the rest.
   * - :func:`~pyKES.fitting_ODE.square_loss_ydiff`
     - The *derivative* of the trace. Usually the right choice for evolved-gas
       measurements: a constant offset in accumulated amount says nothing about
       the kinetics, but it dominates a point-by-point comparison.
   * - :func:`~pyKES.fitting_ODE.square_loss_max_rate_ydiff`
     - Only the maximum rate. Use it to fit a mechanism against a table of
       maximum rates — pairs naturally with
       :func:`~pyKES.utilities.max_rate.extract_max_rate`.

A custom loss follows the same contract — ``(model_data, experimental_data,
times, **kwargs)`` returning ``(error, transformed_model_data)``. The second
return value is what
:meth:`~pyKES.fitting_ODE.Fitting_Model.visualize_optimization_results` plots,
so it must live on the same axis as the data it was compared to.


Weighting experiments
---------------------

An entry of ``experiments`` may be a bare experiment or an
``(experiment, weight)`` tuple:

.. code-block:: python

   model.experiments = [dataset.experiments['MRG-059-ZN-1-1'],           # weight 1.0
                        (dataset.experiments['MRG-059-ZN-14-1'], 0.1),   # noisy run
                        (dataset.experiments['MRG-059-ZO-2-1'], 2.0)]    # replicated

This is the alternative to dropping a run outright. A noisy or atypical
experiment still carries information; down-weighting keeps it in the fit
without letting it dominate.


A complete fit
--------------

.. code-block:: python

   import numpy as np

   from pyKES.database.database_experiments import ExperimentalDataset
   from pyKES.fitting_ODE import Fitting_Model, square_loss_ydiff
   from pyKES.utilities.calculate_absorption import (
       calculate_excitations_per_second_competing)

   dataset = ExperimentalDataset.load_from_hdf5('250608_HTE.h5')

   model = Fitting_Model(['[RuII] > [RuII-ex], k1 ; hv_functionA',
                          '[RuII-ex] > [RuII], k8',
                          '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
                          '[RuIII] > [H2O2] + [RuII], k2 ; hv_functionB',
                          '2 [RuIII] > [Ru-Dimer], k3',
                          '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
                          '[H2O2] > [O2], k5',
                          '[RuIII] > [Inactive], k6'])

   model.experiments = list(dataset.experiments.values())

   # Independently measured — not a free parameter.
   model.fixed_rate_constants = {'k8': 1 / 650e-9}

   model.rate_constants_to_optimize = {'k1': (1e-1, 1e0),
                                       'k2': (1e-1, 1e0),
                                       'k3': (1e-3, 1e-1),
                                       'k4': (1e-3, 1e-1),
                                       'k5': (1e-3, 5e-1),
                                       'k6': (1e-3, 5e-1),
                                       'k7': (1e0,  6e1)}

   model.data_to_be_fitted = {'[O2]': {'x': 'processed_data/x_diff',
                                       'y': 'processed_data/y_diff'}}
   model.initial_conditions = {'[S2O8]': 'metadata/oxidant_concentration_uM',
                               '[RuII]': 'metadata/ru_concentration_uM'}
   model.times = {'times': 'processed_data/time_reaction'}
   model.other_multipliers = {...}          # see the light-absorption guide
   model.loss_function = square_loss_ydiff

   model.optimize()
   model.visualize_optimization_results()

   model.add_fit_results_to_database(dataset)
   dataset.save_to_hdf5('250608_HTE_fitted.h5')

.. image:: ../_static/images/fitting_result.png
   :alt: Three experiments fitted simultaneously with one set of rate constants
   :align: center
   :width: 85%


Optimizers
----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - Character
   * - :meth:`~pyKES.fitting_ODE.Fitting_Model.optimize`
     - Differential evolution. The default, and the right first choice: the
       loss surface of a photochemical network has many local minima, and a
       global search is the only reliable way in. Parallel over the population.
   * - :meth:`~pyKES.fitting_ODE.Fitting_Model.optimize_dual_annealing`
     - Dual annealing. Single-process, often better at escaping a narrow local
       minimum.
   * - :meth:`~pyKES.fitting_ODE.Fitting_Model.minimize`
     - Local minimization from a given start. A polishing step, not a search.
       Note that the bounds are **not** applied here.

Restarting a global search from a previous best is done through ``x0``:

.. code-block:: python

   model.x0 = previous_result.x
   model.optimize()

.. warning::

   ``optimize(workers=-1)`` evaluates the population in subprocesses, so the
   loss function and every multiplier function must be importable at module
   top level — no closures, no lambdas, nothing defined inside another
   function. A ``PicklingError`` at the start of a fit is this. Pass
   ``workers=1`` to rule it out.


Bounds
------

Bounds are not a formality here; they are most of the prior knowledge that
makes the fit converge.

* **Span orders of magnitude, not factors.** ``(1e-3, 1e-1)`` is a reasonable
  bound; ``(0.007, 0.009)`` is a fixed value with extra steps.
* **Fix what is known.** An independently measured excited-state lifetime
  belongs in ``fixed_rate_constants``. Every parameter moved there is one fewer
  dimension for the search.
* **Check for parameters at their bound.** A fitted value sitting exactly on a
  bound means the data wanted to go further; widen it and refit rather than
  reporting the boundary as a result.


Storing a fit
-------------

:meth:`~pyKES.fitting_ODE.Fitting_Model.add_fit_results_to_database` writes both
halves of the result back into the dataset:

* per experiment, ``processed_data['<species>_experimental']`` and
  ``processed_data['<species>_fit']``, so the comparison plots from the
  Streamlit pages;
* dataset-wide, ``processing_parameters['fitting_model']`` — the mechanism, the
  bounds, the fitted values, the loss function name, the experiments used and
  the optimizer's own report.

The multipliers are passed through
:func:`~pyKES.utilities.make_json_serializable.make_json_serializable` first, so
a stored fit records *which* absorption function it used by name even though a
callable has no JSON representation.


Diagnosing a bad fit
--------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Symptom
     - Likely cause
   * - Error barely moves from its initial value
     - The bounds exclude the true values, or the loss compares the wrong
       quantity — check with ``square_loss_ydiff`` versus
       ``square_loss_time_series``.
   * - Fits every experiment badly in the same direction
     - Missing chemistry rather than wrong numbers. A systematic residual is a
       mechanism problem.
   * - Fits most experiments and misses a few
     - Suspect the conditions of the outliers first: a wrong entry in the
       overview sheet resolves silently into a wrong initial condition.
   * - Different runs give very different constants
     - The parameters are correlated and the data does not separate them. Add
       experiments that vary the conditions the correlated steps depend on
       differently.
   * - ``ValueError: Semi-strict mode: No entries could be resolved``
     - No path in ``data_to_be_fitted`` or ``initial_conditions`` matched that
       experiment — usually a renamed key in ``processed_data``.


Reference
---------

* :mod:`pyKES.fitting_ODE` — the model class, the loss functions and the
  objective.
* :mod:`pyKES.utilities.resolve_attributes` — the path resolution.
* :doc:`dataset` — where the experiments come from.
