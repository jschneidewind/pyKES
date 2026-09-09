Quickstart
==========

Four short examples, one per part of the package. Each is self-contained and
runs in seconds.


1. Simulate a reaction network
------------------------------

A mechanism is a list of strings. Species go in square brackets, reactants and
products are separated by ``>``, and the rate constant follows a comma:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   from pyKES.reaction_model import Reaction_Model

   model = Reaction_Model(
       reaction_network=['[A] > [B], k1',
                         '[B] > [C], k2'],
       rate_constants={'k1': 0.045, 'k2': 0.011},
       initial_conditions={'[A]': 10.0},
       times=np.linspace(0, 300, 300))

   model.solve_ode()
   model.plot_solution()
   plt.show()

Everything the solver produced stays on the model: ``model.species`` is the
species list, and ``model.solution`` is the ``(len(times), len(species))``
concentration array, so ``model.solution[:, model.species.index('[B]')]`` is
the trace of the intermediate.

Stoichiometric coefficients are written in front of the species
(``2 [RuIII] > [Ru-Dimer], k3``), and anything after a semicolon is an extra
multiplier applied to the rate — which is how light enters. See
:doc:`guide/reaction_networks`.


2. Fit rate constants to measured data
--------------------------------------

The same network, with the rate constants unknown and three experiments to
reproduce. The initial concentration differs between the experiments, so it is
given as a *path* into each experiment rather than as a number:

.. code-block:: python

   from pyKES.fitting_ODE import Fitting_Model, square_loss_time_series

   model = Fitting_Model(['[A] > [B], k1',
                          '[B] > [C], k2'])

   model.experiments = [dataset.experiments['run-1'],
                        dataset.experiments['run-2'],
                        dataset.experiments['run-3']]

   model.rate_constants_to_optimize = {'k1': (1e-3, 5e-1),
                                       'k2': (1e-3, 5e-1)}

   model.data_to_be_fitted = {'[B]': {'x': 'processed_data/intermediate/x',
                                      'y': 'processed_data/intermediate/y'}}
   model.initial_conditions = {'[A]': 'metadata/initial_concentration_uM'}
   model.times = {'times': 'processed_data/times'}
   model.loss_function = square_loss_time_series

   model.optimize()
   model.visualize_optimization_results()

.. image:: _static/images/fitting_result.png
   :alt: Three experiments fitted simultaneously with one set of rate constants
   :align: center
   :width: 85%

One mechanism, one pair of rate constants, three starting concentrations. See
:doc:`guide/fitting`.


3. Extract a maximum rate from a noisy trace
--------------------------------------------

.. code-block:: python

   from pyKES.utilities.max_rate import extract_max_rate, plot_max_rate
   from pyKES.utilities.unit_handler import Quantity

   time = Quantity(time_seconds, 's')
   amount = Quantity(evolved_h2_umol, 'umol')

   result = extract_max_rate(time, amount)

   print(result.max_rate.unit['umol / h'])   # read it in whatever unit you want
   print(result.max_rate_std)                # its standard deviation
   print(result.flags)                       # empty list = nothing suspicious

   plot_max_rate(result, time, amount)

.. image:: _static/images/max_rate_diagnostic.png
   :alt: Maximum-rate diagnostic showing a rejected bubble artifact and the fitted rate
   :align: center
   :width: 85%

The trace above carries a slow baseline wave whose local slope rivals the true
rate and a bubble whose instantaneous slope is seventy-five times it; the
extracted rate is within 8 % of the truth and the disturbances are flagged. See
:doc:`max_rate`.


4. Trace where the light goes
-----------------------------

For a photochemical network, the concentration traces do not say which fraction
of the absorbed light reached the product. Freezing the network at one point in
time answers that:

.. code-block:: python

   model.calculate_reaction_network_propopagation(
       timepoint=10,
       absorbing_species_with_extinction_coefficients={
           '[A]': {'excited_name': '[A-excited]',
                   'extinction_coefficient': 8500},
           '[B]': {'excited_name': '[B-excited]',
                   'extinction_coefficient': 5400}},
       photon_flux=1e17,
       pathlength=2.25,
       concentration_unit='uM')

   model.plot_reaction_network_propagation()

.. image:: _static/images/pathway_diagram.png
   :alt: Pathway diagram of the photon budget of a two-chromophore cascade
   :align: center
   :width: 90%

Bar heights are log-scaled, so a pathway carrying a thousandth of the light is
still visible next to one carrying a third. See :doc:`guide/pathways`.


Where next
----------

* :doc:`guide/architecture` — how the pieces fit together.
* :doc:`guide/dataset` — getting measurement files into an ``ExperimentalDataset``.
* :doc:`api/index` — the full reference.
