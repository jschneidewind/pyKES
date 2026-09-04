Installation
============

Requirements
------------

pyKES needs **Python 3.9 or newer**. Its runtime dependencies — NumPy, SciPy,
pandas, h5py, matplotlib, plotly, openpyxl and Streamlit — are installed
automatically.


From PyPI
---------

.. code-block:: bash

   pip install pyKES

To check that the installation works:

.. code-block:: bash

   python -c "import pyKES; print('pyKES import OK')"


From source
-----------

.. code-block:: bash

   git clone https://github.com/jschneidewind/pyKES.git
   cd pyKES
   pip install -e .

The editable install is the right one for a repository that embeds pyKES and
tracks its development: changes in the working tree take effect without
reinstalling.


Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Installs
   * - ``docs``
     - Sphinx and the theme and extensions needed to build this documentation.
   * - ``dev``
     - ``docs`` plus pytest, for running the test suite.

.. code-block:: bash

   pip install -e '.[dev]'


Using pyKES from another repository
-----------------------------------

Repositories that embed the Streamlit pages usually depend on a released
version:

.. code-block:: toml

   # pyproject.toml of the embedding repository
   dependencies = ["pyKES>=0.2.4"]

When pyKES itself is being developed alongside such a repository, install it
from the working tree instead:

.. code-block:: bash

   pip install -e /path/to/pyKES

If the editor does not pick up the source tree for autocompletion, point it at
``src`` explicitly — for VS Code, in ``.vscode/settings.json``:

.. code-block:: json

   {
       "python.analysis.extraPaths": ["/path/to/pyKES/src"],
       "python.autoComplete.extraPaths": ["/path/to/pyKES/src"]
   }

After a pyKES version bump, refresh the lock file of the embedding repository:

.. code-block:: bash

   uv lock --refresh
   uv sync


Running the test suite
----------------------

.. code-block:: bash

   pip install -e '.[dev]'
   pytest

The suite runs against synthetic data with known ground truth and against a set
of committed real measurement traces, so it needs no files outside the
repository.


Building the documentation
--------------------------

.. code-block:: bash

   pip install -e '.[docs]'
   sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html``. The example figures are committed, so a
documentation build does not need to run any simulations; regenerate them with

.. code-block:: bash

   python docs/generate_example_figures.py
