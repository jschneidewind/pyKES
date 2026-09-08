"""
Tests for what actually ends up in an installed copy of the package.

Every failure guarded against here is silent at run time rather than loud. A
missing schema directory makes `load_entity_schemas` return `{}` *by design*,
which quietly removes validation, the download templates and every reference
declaration; a missing `pages/` directory leaves a single-page app with no
navigation; a missing Streamlit configuration reverts the theme to the light
default and drops the upload ceiling back to 200 MB, rejecting exactly the
batches the database is built for. None of them raises, so each is asserted.
"""

from pathlib import Path

import pytest

from pyKES.database.entity_schema import DEFAULT_SCHEMA_DIRECTORY, load_entity_schemas
from pyKES.database.index_schema import ENTITY_TYPES
from pyKES.database_app import launch
from pyKES.utilities.version_information import UNKNOWN_VERSION, get_pykes_version


# =============================================================================
# Paths of the installed package
# =============================================================================

DATABASE_APP_DIRECTORY = Path(launch.__file__).resolve().parent

PACKAGED_STREAMLIT_CONFIG = DATABASE_APP_DIRECTORY / ".streamlit" / "config.toml"

# Repository root, present in a source checkout and absent when the tests run
# against an installed wheel.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Package data
# =============================================================================

def test_every_entity_type_has_a_shipped_schema():
    """
    An absent schema directory is not an error anywhere in the code — it just
    means nothing is validated — so the install has to be checked instead.
    """
    schemas = load_entity_schemas()

    assert set(schemas) == set(ENTITY_TYPES)


def test_schema_files_are_package_data():
    shipped = sorted(path.name for path in DEFAULT_SCHEMA_DIRECTORY.glob("*.yaml"))

    assert len(shipped) == len(ENTITY_TYPES)
    assert DEFAULT_SCHEMA_DIRECTORY.is_relative_to(DATABASE_APP_DIRECTORY.parent)


def test_the_entry_script_and_its_pages_are_installed():
    """
    Streamlit discovers pages relative to the entry script, which is why the
    multipage app works from a wheel at all — and why losing `pages/` shows as
    a working app with no navigation rather than as a failure.
    """
    assert (DATABASE_APP_DIRECTORY / launch.ENTRY_SCRIPT).is_file()

    pages = sorted(path.name for path in
                   (DATABASE_APP_DIRECTORY / "pages").glob("[0-9]*.py"))

    assert pages == ["01_Browse.py", "02_Entity.py", "03_Property_map.py",
                     "04_Contribute.py", "05_Admin.py"]


def test_the_streamlit_configuration_is_installed():
    assert PACKAGED_STREAMLIT_CONFIG.is_file()

    text = PACKAGED_STREAMLIT_CONFIG.read_text(encoding="utf-8")

    assert "maxUploadSize" in text
    assert 'base = "dark"' in text


def test_the_development_copy_matches_the_packaged_one():
    """
    Two copies exist deliberately: the packaged one is what a deployment gets,
    and the repository-root one is what `streamlit run src/.../Home.py` picks
    up during development, because Streamlit resolves the file against the
    working directory. They must not drift, or the app looks different in
    development from the way it looks deployed.
    """
    development_copy = REPOSITORY_ROOT / ".streamlit" / "config.toml"

    if not development_copy.is_file():
        pytest.skip("not a source checkout")

    assert development_copy.read_bytes() == PACKAGED_STREAMLIT_CONFIG.read_bytes()


# =============================================================================
# Provenance
# =============================================================================

def test_the_running_version_is_knowable():
    """
    `get_pykes_version()` is stamped into every payload the server writes, so
    an install it cannot read reports `unknown` and every derived file loses
    its provenance. The trap is a container that copies sources onto
    PYTHONPATH without installing the project.
    """
    assert get_pykes_version() != UNKNOWN_VERSION


# =============================================================================
# The console entry point
# =============================================================================

def test_help_does_not_require_a_data_root(monkeypatch, capsys):
    """
    `main` verifies the data root before handing over, so that an absent or
    unwritable one stops the service instead of surfacing later as a
    traceback. A data root is not a precondition of printing usage, though,
    and requiring one made `photocat-app --help` fail — which the container
    image's own smoke test caught.
    """
    monkeypatch.setenv("PHOTOCAT_DATA_ROOT", "/nonexistent/never-attached")
    monkeypatch.setattr("sys.argv", ["photocat-app", "--help"])

    with pytest.raises(SystemExit) as exit_status:
        launch.main()

    assert exit_status.value.code == 0
    assert "streamlit run" in capsys.readouterr().out
