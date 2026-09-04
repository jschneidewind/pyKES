Contributing
============

Contributions are welcome — open an issue or a pull request at
`github.com/jschneidewind/pyKES <https://github.com/jschneidewind/pyKES>`_.


Getting set up
--------------

.. code-block:: bash

   git clone https://github.com/jschneidewind/pyKES.git
   cd pyKES
   pip install -e '.[dev]'
   pytest


Coding conventions
------------------

:source:`src/pyKES/reaction_ODE.py <src/pyKES/reaction_ODE.py>` is the style
benchmark for this repository.

**Reduce nesting.** Break logic into small, self-contained functions rather
than deep ``if`` / ``with`` / ``try`` ladders, and let the function name carry
the explanation. Avoid the opposite excess too: a helper needs one clear,
nameable job.

**NumPy-style docstrings on every function**, with brief ``Parameters`` and
``Returns`` blocks. Skip ``Examples`` unless they materially clarify usage, and
do not repeat type hints in prose.

**Meaningful comments only.** Explain the *why* — hidden constraints,
invariants, surprising decisions, references to a bug that motivated the code.
Do not restate what well-named code already shows. Separate logical blocks
within a function with blank lines.

**Fail fast.** Avoid broad ``try``/``except`` and silent fallbacks; let
exceptions propagate, where in Streamlit they surface to the user as a
traceback. Validate inputs at construction boundaries (``__post_init__``) and
trust internal invariants thereafter. ``try``/``finally`` for resource cleanup
is fine; ``try: ... except: pass`` is not.

**Be short.** Three clear lines beat ten lines of speculative robustness.

**Full-word names.** No single-letter or abbreviated variables, even for
mathematical quantities: ``filtered_covariances``, not ``P_f``;
``lengthscale``, not ``ell``. Function names are verbs describing the job —
``parse_reactions``, ``detect_artifacts``.

**No magic numbers.** Statistical factors, thresholds and window sizes become
named module-level constants with a short explanatory comment, grouped into
commented sections at the top of the module.
:source:`src/pyKES/utilities/max_rate.py <src/pyKES/utilities/max_rate.py>`
shows the pattern.

**No nested function definitions.** Keep every function at module level and
pass extra data through parameters — ``scipy.optimize.minimize(..., args=...)``
rather than a closure. The exception is the existing ODE-builder closures that
solver APIs require.

**Avoid** ``while`` **loops.** Prefer vectorized NumPy scans (boolean arrays,
``np.convolve``, ``np.searchsorted``, prefix sums) or bounded ``for`` loops.
Sequential recursions such as a Kalman filter use ``for``.


Tests
-----

Tests live in :source:`src/tests <src/tests>`.

For numerical and analysis code, test against **synthetic data with known
ground truth** — never against files that only exist on one machine. New
numerical algorithms should additionally be validated against a reference
implementation during development.

Runnable analysis modules carry a small ``test_function()`` demo under
``if __name__ == "__main__":``, mirroring
:source:`reaction_ODE.py <src/pyKES/reaction_ODE.py>`. These are demonstrations
rather than assertions; the real checks belong in the pytest suite.


Documentation
-------------

Non-trivial modules get a companion document explaining how the code works in
detail, readable also by non-specialists: motivation, pipeline stages,
parameter guidance, validation. :doc:`max_rate` is the model to follow.

To build the documentation locally:

.. code-block:: bash

   pip install -e '.[docs]'
   sphinx-build -b html docs docs/_build/html

The example figures are committed and are regenerated with

.. code-block:: bash

   python docs/generate_example_figures.py


Changelog
---------

Every user-visible change gets an entry under ``[Unreleased]`` in
:source:`CHANGELOG.md <CHANGELOG.md>`. Entries in this project explain *why* a
change was made and what was measured, not only what changed — read a few
existing ones before writing a new one.


Releasing
---------

See :doc:`releasing`.
