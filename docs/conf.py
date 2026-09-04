"""Sphinx configuration for the pyKES documentation.

Built on Read the Docs from ``.readthedocs.yaml`` at the repository root, and
locally with::

    pip install -e '.[docs]'
    sphinx-build -b html docs docs/_build/html
"""

import sys
from importlib import metadata
from pathlib import Path

# The package is normally installed (Read the Docs installs it from
# pyproject.toml), but adding the source tree keeps a plain `sphinx-build` in a
# fresh checkout working as well.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

# -- Project information -------------------------------------------------------

project = 'pyKES'
author = 'Jacob Schneidewind'
copyright = '2026, Jacob Schneidewind'

try:
    release = metadata.version('pyKES')
except metadata.PackageNotFoundError:
    release = '0.0.0+unknown'

version = '.'.join(release.split('.')[:2])

# -- General configuration -----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.extlinks',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.mermaid',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The guides are RST, the deep-dive documents that predate this build are
# Markdown and stay Markdown so they keep rendering on GitHub too.
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

master_doc = 'index'

# -- Autodoc -------------------------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Signatures of the dataclasses carry every field with its default, which is
# both unreadable and redundant with the Parameters section of the docstring.
autodoc_class_signature = 'separated'

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented_params'

autodoc_preserve_defaults = True

autosummary_generate = False

# Heavy or platform-specific imports that need not be installed to build the
# documentation. Everything else is a real dependency and is imported for real,
# so a broken import shows up as a failed build rather than as a missing page.
autodoc_mock_imports = []

# -- Napoleon (NumPy-style docstrings) -----------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
}

# Referencing a name that has no target is a documentation bug worth seeing,
# but third-party types appearing in signatures are not ours to fix.
nitpicky = False

# -- MyST ----------------------------------------------------------------------

myst_enable_extensions = ['colon_fence', 'deflist', 'dollarmath', 'linkify']

myst_heading_anchors = 3

# -- Links ---------------------------------------------------------------------

GITHUB_BLOB = 'https://github.com/jschneidewind/pyKES/blob/main/%s'

extlinks = {
    'source': (GITHUB_BLOB, '%s'),
    'issue': ('https://github.com/jschneidewind/pyKES/issues/%s', 'issue #%s'),
}

# -- HTML output ---------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_title = f'pyKES {version}'
html_logo = None
html_favicon = None

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
    'titles_only': False,
    'style_external_links': True,
}

html_context = {
    'display_github': True,
    'github_user': 'jschneidewind',
    'github_repo': 'pyKES',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# -- Copy button ---------------------------------------------------------------

# Strip prompts so a copied block pastes as runnable code.
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True
