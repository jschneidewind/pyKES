"""
Provenance stamps for datasets and the files they are written to.

Every `ExperimentalDataset` carries a ``version`` dictionary recording which
code produced its contents and when. The dictionary is written into the HDF5
file, so a file that has been passed around can still answer "which pyKES
version processed this, and when was it last touched?".

The dictionary holds:

``pykes_version``
    Version of the installed pyKES package.
``schema_version``
    On-disk layout version of the HDF5 file (see
    `pyKES.database.database_experiments.SCHEMA_VERSION`).
``created``
    Local ISO timestamp of the first save of the dataset.
``last_modified``
    Local ISO timestamp of the most recent save.
``last_processed``
    Local ISO timestamp of the most recent run of a processing function
    (initial ingestion or reprocessing of an existing file).
``external_version``
    Free-form dictionary an external app fills with its own provenance,
    typically the git commit of the repository holding the processing
    functions (see `get_git_commit`).

Author: pyKES Development Team
Date: 31 August 2026
"""

import re
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# Keys of the version dictionary
# =============================================================================

PYKES_VERSION_KEY = 'pykes_version'
SCHEMA_VERSION_KEY = 'schema_version'
CREATED_KEY = 'created'
LAST_MODIFIED_KEY = 'last_modified'
LAST_PROCESSED_KEY = 'last_processed'
EXTERNAL_VERSION_KEY = 'external_version'

# Reported when pyKES is imported from a source tree that was never installed,
# so no distribution metadata exists to read the version from.
UNKNOWN_VERSION = 'unknown'

# Repository root relative to this module, used to read the version of a source
# checkout (src/pyKES/utilities/version_information.py -> repository root)
PYPROJECT_PATH = Path(__file__).resolve().parents[3] / 'pyproject.toml'

# Matches the `version = "0.1.7"` line of the [project] table
PYPROJECT_VERSION_PATTERN = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)

# Marker appended to a commit hash when the working tree has uncommitted changes
DIRTY_SUFFIX = '-dirty'


# =============================================================================
# Primitive stamps
# =============================================================================

def get_pykes_version() -> str:
    """
    Read the version of the pyKES code that is running.

    The version of a source checkout is read from its ``pyproject.toml`` and
    takes precedence over the installed distribution metadata: an editable
    install keeps reporting the version it was installed at, which would stamp
    datasets with a version the running code no longer has.

    Returns
    -------
    version : str
        Version string, or `UNKNOWN_VERSION` when neither source is available.
    """
    source_version = read_version_from_pyproject()
    if source_version:
        return source_version

    try:
        return installed_version('pyKES')
    except PackageNotFoundError:
        return UNKNOWN_VERSION


def read_version_from_pyproject() -> Optional[str]:
    """
    Read the project version from the ``pyproject.toml`` of a source checkout.

    Returns
    -------
    version : str or None
        Version declared in the file, or None when pyKES runs from an installed
        package (no ``pyproject.toml`` beside it) or the file declares none.
    """
    if not PYPROJECT_PATH.is_file():
        return None

    match = PYPROJECT_VERSION_PATTERN.search(PYPROJECT_PATH.read_text(encoding='utf-8'))

    return match.group(1) if match else None


def current_timestamp() -> str:
    """
    Build a local ISO-8601 timestamp including the UTC offset.

    Returns
    -------
    timestamp : str
        e.g. ``'2026-08-31T14:03:57+02:00'``.
    """
    return datetime.now().astimezone().isoformat(timespec='seconds')


def get_git_commit(repository_path: Optional[str] = None, short: bool = False) -> Optional[str]:
    """
    Read the current git commit of a repository, for external provenance.

    Intended for external apps that want the commit of their own processing
    code recorded in the dataset::

        dataset.set_external_version({'app': 'photocat',
                                      'commit': get_git_commit(__file__)})

    Parameters
    ----------
    repository_path : str, optional
        Any path inside the repository — a file path is resolved to its
        containing directory. Defaults to the current working directory.
    short : bool, default False
        Return the abbreviated hash instead of the full one.

    Returns
    -------
    commit : str or None
        Commit hash, suffixed with ``'-dirty'`` when the working tree holds
        uncommitted changes. None when the path is not inside a git work tree,
        which is the normal case for a deployment from a source archive.
    """
    working_directory = Path(repository_path).resolve() if repository_path else Path.cwd()
    if working_directory.is_file():
        working_directory = working_directory.parent

    describe_command = ['git', 'rev-parse', '--short', 'HEAD'] if short else ['git', 'rev-parse', 'HEAD']
    commit = run_git_command(describe_command, working_directory)

    if commit is None:
        return None

    # An empty `status --porcelain` means a clean tree; None means the status
    # call itself failed, in which case the cleanliness is simply not reported.
    status = run_git_command(['git', 'status', '--porcelain'], working_directory)

    return f"{commit}{DIRTY_SUFFIX}" if status else commit


def run_git_command(command: list, working_directory: Path) -> Optional[str]:
    """
    Run a git command and return its stripped stdout.

    Parameters
    ----------
    command : list of str
        Command and arguments to execute.
    working_directory : Path
        Directory the command is run in.

    Returns
    -------
    output : str or None
        Standard output, or None when git is unavailable or the command fails
        (not a repository, git not installed).
    """
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False
        )
    except (OSError, FileNotFoundError):
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()


# =============================================================================
# Version dictionary handling
# =============================================================================

def build_version_information(schema_version: str,
                              external_version: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a fresh version dictionary for a newly created dataset.

    Parameters
    ----------
    schema_version : str
        On-disk layout version the dataset will be written with.
    external_version : dict, optional
        Provenance of the external app (e.g. its git commit).

    Returns
    -------
    version_information : dict
        Version dictionary with `created` and `last_modified` set to now.
    """
    timestamp = current_timestamp()

    return {
        PYKES_VERSION_KEY: get_pykes_version(),
        SCHEMA_VERSION_KEY: schema_version,
        CREATED_KEY: timestamp,
        LAST_MODIFIED_KEY: timestamp,
        LAST_PROCESSED_KEY: None,
        EXTERNAL_VERSION_KEY: dict(external_version) if external_version else {},
    }


def stamp_version_information(version_information: Dict[str, Any],
                              schema_version: str,
                              processed: bool = False,
                              external_version: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Update a version dictionary in place for a save or a processing run.

    Missing entries are filled in, so version dictionaries of files written
    before a key existed are upgraded on their next save. ``created`` is only
    ever set once — for a dataset loaded from a file it keeps the timestamp of
    the original file.

    Parameters
    ----------
    version_information : dict
        Dictionary to update; may be empty for a legacy dataset.
    schema_version : str
        On-disk layout version the dataset is written with.
    processed : bool, default False
        Whether a processing function was just run, which additionally sets
        `last_processed` and refreshes `pykes_version`.
    external_version : dict, optional
        Provenance of the external app, merged into any existing entry.

    Returns
    -------
    version_information : dict
        The same dictionary, updated.
    """
    timestamp = current_timestamp()

    version_information.setdefault(CREATED_KEY, timestamp)
    version_information.setdefault(LAST_PROCESSED_KEY, None)
    version_information.setdefault(EXTERNAL_VERSION_KEY, {})

    version_information[PYKES_VERSION_KEY] = get_pykes_version()
    version_information[SCHEMA_VERSION_KEY] = schema_version
    version_information[LAST_MODIFIED_KEY] = timestamp

    if processed:
        version_information[LAST_PROCESSED_KEY] = timestamp

    if external_version:
        version_information[EXTERNAL_VERSION_KEY].update(external_version)

    return version_information


def describe_version_information(version_information: Dict[str, Any]) -> str:
    """
    Render a version dictionary as a one-line human-readable summary.

    Parameters
    ----------
    version_information : dict
        Version dictionary to describe.

    Returns
    -------
    description : str
        e.g. ``'pyKES 0.1.7 | created 2026-08-31T… | last modified 2026-09-02T…'``.
    """
    if not version_information:
        return "no version information (file written before versioning existed)"

    parts = [f"pyKES {version_information.get(PYKES_VERSION_KEY, UNKNOWN_VERSION)}"]

    for label, key in (("created", CREATED_KEY),
                       ("last modified", LAST_MODIFIED_KEY),
                       ("last processed", LAST_PROCESSED_KEY)):
        if version_information.get(key):
            parts.append(f"{label} {version_information[key]}")

    external_version = version_information.get(EXTERNAL_VERSION_KEY) or {}
    if external_version:
        parts.append(", ".join(f"{key}: {value}" for key, value in external_version.items()))

    return " | ".join(parts)


def test_function():
    """Demonstrate building and stamping a version dictionary."""
    version_information = build_version_information(schema_version='1.1',
                                                    external_version={'app_commit': 'abc123'})
    print(describe_version_information(version_information))

    stamp_version_information(version_information,
                              schema_version='1.1',
                              processed=True,
                              external_version={'app_commit': 'def456'})
    print(describe_version_information(version_information))

    print(f"git commit of this repository: {get_git_commit(__file__, short=True)}")


if __name__ == '__main__':
    test_function()
