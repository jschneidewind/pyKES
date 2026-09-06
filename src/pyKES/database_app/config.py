"""
Configuration for the database application.

Everything an installation needs to vary lives here, so the components
themselves are not edited per deployment — the same convention the processing
app's `config_interface` follows.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


# =============================================================================
# Where the data lives
# =============================================================================

# Overridden by the systemd unit; the default suits a local prototype run.
DATA_ROOT_VARIABLE = "PHOTOCAT_DATA_ROOT"
DEFAULT_DATA_ROOT = Path.home() / ".photocat"


# =============================================================================
# Identity
# =============================================================================

# Header Authelia sets through nginx. Read from the WebSocket request, which is
# why the proxy must set it on the same location that proxies the upgrade.
USER_HEADER = "Remote-User"
GROUPS_HEADER = "Remote-Groups"

# Group whose members may edit and delete entries they do not own.
ADMIN_GROUP = "admins"

# Used only when no proxy header is present, i.e. running the app directly for
# development. Never reachable through the deployed proxy, which refuses
# unauthenticated requests before they arrive.
DEVELOPMENT_USER = "developer"


@dataclass
class DatabaseAppConfig:
    """
    Settings for one deployment of the database application.

    Parameters
    ----------
    title, icon : str
        Branding shown in the browser tab and page header.
    data_root : Path
        Directory holding ``index.sqlite``, ``payloads/`` and ``uploads/``.
    default_entity_type : str
        Kind of entry the search page opens on.
    default_columns : list of str
        Effective-metadata keys and ``result:`` labels shown as table columns
        before the user chooses their own.
    reference_instructions_by_type : dict
        ``{entity_type: {column: {'role': role}}}`` offered as the default
        reference declaration when uploading a sheet of that type.
    page_size : int
        Rows per page of search results.
    allow_development_login : bool
        Whether to fall back to ``DEVELOPMENT_USER`` when no proxy header is
        present. Must be False in a deployment.
    """

    title: str = "Photocatalysis Database"
    icon: str = ":microscope:"
    data_root: Path = field(default_factory=lambda: Path(
        os.environ.get(DATA_ROOT_VARIABLE, DEFAULT_DATA_ROOT)))
    default_entity_type: str = "experiment"
    default_columns: List[str] = field(default_factory=list)
    reference_instructions_by_type: Dict[str, Any] = field(default_factory=dict)
    page_size: int = 50
    allow_development_login: bool = True

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)


# =============================================================================
# The group's own wiring
# =============================================================================

# Which spreadsheet column of each kind of entry names another entry. This is
# the one piece of the deployment that encodes the group's own chain, and it is
# offered as a default on the upload page rather than hard-coded into ingestion.
GROUP_REFERENCE_INSTRUCTIONS = {
    "experiment": {
        "Catalyst Batch [experiment no.]": {"role": "catalyst_batch"},
    },
    "catalyst_batch": {
        "Finished Semiconductor": {"role": "finished_semiconductor"},
    },
    "finished_semiconductor": {
        "Precursor Chemical A": {"role": "precursor_chemical_a"},
        "Precursor Chemical B": {"role": "precursor_chemical_b"},
    },
    "precursor_semiconductor": {
        "Precursor Chemical A": {"role": "precursor_chemical_a"},
        "Precursor Chemical B": {"role": "precursor_chemical_b"},
    },
    "precursor_chemical": {
        "Commercial Chemical": {"role": "commercial_chemical"},
    },
    "stock_solution": {
        "Commercial Chemical": {"role": "commercial_chemical"},
    },
}

DEFAULT_CONFIG = DatabaseAppConfig(
    reference_instructions_by_type=GROUP_REFERENCE_INSTRUCTIONS,
    default_columns=["result:Max. rate (umol/s)",
                     "result:Apparent quantum yield (%)"],
)
