"""
Configuration for the database application.

Everything an installation needs to vary is read from the environment, so the
deployed unit of code can be an unmodified wheel or container image. A
deployment that had to edit this file would have to re-apply the edit on every
update, and the setting it must never lose is `allow_development_login`:
reverting that one silently turns the application into an unauthenticated
admin console.

Every variable is named `PHOTOCAT_<SETTING>`, so a deployment is one
`Environment=` block in a systemd unit or one `environment:` mapping in a
compose file. See docs/deployment.md.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from pyKES.database.entity_schema import DEFAULT_SCHEMA_DIRECTORY, load_entity_schemas


# =============================================================================
# Reading settings from the environment
# =============================================================================

# Shared prefix of every variable this application reads.
SETTING_PREFIX = "PHOTOCAT_"

# Suits a local prototype run; a deployment always sets PHOTOCAT_DATA_ROOT.
DEFAULT_DATA_ROOT = Path.home() / ".photocat"

# Accepted as true for a boolean setting. Anything else is false, so a
# misspelt value fails closed rather than enabling the thing it names.
TRUE_VALUES = ("1", "true", "yes", "on")


def setting_from_environment(name: str, default: Any,
                             cast: Callable[[str], Any] = str) -> Any:
    """
    Read one deployment setting from the environment.

    Parameters
    ----------
    name : str
        Setting name without the `PHOTOCAT_` prefix.
    default : Any
        Value used when the variable is unset.
    cast : callable, optional
        Converts the raw string to the field's type.

    Returns
    -------
    value : Any
        Converted value, or `default` when the variable is unset.
    """
    raw = os.environ.get(f"{SETTING_PREFIX}{name}")

    return default if raw is None else cast(raw)


def as_boolean(text: str) -> bool:
    """
    Interpret an environment variable as a flag.

    Parameters
    ----------
    text : str
        Raw variable value.

    Returns
    -------
    enabled : bool
        True only for an explicitly affirmative value.
    """
    return text.strip().lower() in TRUE_VALUES


# =============================================================================
# Identity
# =============================================================================

# Headers Authelia sets through nginx. Read from the WebSocket request, which
# is why the proxy must set them on the same location that proxies the upgrade.
USER_HEADER = "Remote-User"
GROUPS_HEADER = "Remote-Groups"

# Group whose members may edit and delete entries they do not own. Overridable
# because an institution's directory names its groups its own way, and pointing
# this at an LDAP group should not require a release.
ADMIN_GROUP = "admins"

# Used only when no proxy header is present, i.e. running the app directly for
# development. Never reachable through the deployed proxy, which refuses
# unauthenticated requests before they arrive.
DEVELOPMENT_USER = "developer"


# =============================================================================
# Deployment identity, for the version caption
# =============================================================================

# Set by the container image and the systemd unit so the running application
# can say which build it is. Unset in a source checkout.
ENVIRONMENT_NAME = setting_from_environment("ENV", "development")
IMAGE_TAG = setting_from_environment("IMAGE_TAG", "")
GIT_SHA = setting_from_environment("GIT_SHA", "")

# The value of PHOTOCAT_ENV that means "this is the real thing"; anything else
# makes the application say so on every page.
PRODUCTION_ENVIRONMENT = "production"


@dataclass
class DatabaseAppConfig:
    """
    Settings for one deployment of the database application.

    Every field reads `PHOTOCAT_<NAME>` from the environment, falling back to
    the default given here.

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
        reference declaration when uploading a sheet of that type. Derived from
        `schema_directory` in `__post_init__` unless given explicitly, so a
        deployment with its own vocabulary cannot end up interpreting the same
        reference column two different ways.
    page_size : int
        Rows per page of search results.
    allow_development_login : bool
        Whether to fall back to ``DEVELOPMENT_USER`` when no proxy header is
        present. Defaults to False: the failure mode of the wrong default is
        silent, since a proxy that stops sending the identity header would
        otherwise make every visitor an admin called `developer`.
    user_header, groups_header : str
        Headers the authenticating proxy sets.
    admin_group : str
        Group whose members may edit entries they do not own.
    schema_directory : Path
        Directory of the per-entity-type metadata schemas. Defaults to the ones
        shipped with pyKES; a deployment maintaining its own copy points this at
        it, so the group can edit its vocabulary without touching the package.
    """

    title: str = field(default_factory=lambda: setting_from_environment(
        "TITLE", "Photocatalysis Database"))
    icon: str = field(default_factory=lambda: setting_from_environment(
        "ICON", ":microscope:"))
    data_root: Path = field(default_factory=lambda: setting_from_environment(
        "DATA_ROOT", DEFAULT_DATA_ROOT, Path))
    default_entity_type: str = "experiment"
    default_columns: List[str] = field(default_factory=list)
    reference_instructions_by_type: Dict[str, Any] = field(default_factory=dict)
    page_size: int = field(default_factory=lambda: setting_from_environment(
        "PAGE_SIZE", 50, int))
    allow_development_login: bool = field(default_factory=lambda:
        setting_from_environment("ALLOW_DEV_LOGIN", False, as_boolean))
    user_header: str = field(default_factory=lambda: setting_from_environment(
        "USER_HEADER", USER_HEADER))
    groups_header: str = field(default_factory=lambda: setting_from_environment(
        "GROUPS_HEADER", GROUPS_HEADER))
    admin_group: str = field(default_factory=lambda: setting_from_environment(
        "ADMIN_GROUP", ADMIN_GROUP))
    schema_directory: Path = field(default_factory=lambda:
        setting_from_environment("SCHEMA_DIRECTORY", DEFAULT_SCHEMA_DIRECTORY, Path))

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.schema_directory = Path(self.schema_directory)

        # Derived rather than passed in, so the declarations an upload is read
        # with and the ones a correction is re-resolved with come from the same
        # schemas. Held apart, they drift the moment a deployment maintains its
        # own vocabulary, and the same column then means two different things.
        if not self.reference_instructions_by_type:
            self.reference_instructions_by_type = reference_instructions(
                self.schema_directory)


# =============================================================================
# The group's own wiring
# =============================================================================

def reference_instructions(schema_directory: Path) -> Dict[str, Any]:
    """
    Read which column of each kind of entry names another entry.

    Derived from the schema files rather than written twice: a field of type
    `reference` already says which column it is and what role the link takes,
    so a second hand-written copy could only ever disagree with it.

    Parameters
    ----------
    schema_directory : Path
        Directory of per-entity-type schema files.

    Returns
    -------
    instructions : dict
        ``{entity_type: {column: {'role': role}}}``, omitting types that
        declare no references.
    """
    schemas = load_entity_schemas(schema_directory)

    return {entity_type: schema.reference_instructions()
            for entity_type, schema in schemas.items()
            if schema.reference_instructions()}


GROUP_REFERENCE_INSTRUCTIONS = reference_instructions(DEFAULT_SCHEMA_DIRECTORY)

DEFAULT_CONFIG = DatabaseAppConfig(
    default_columns=["result:Max. rate (umol/s)",
                     "result:Apparent quantum yield (%)"],
)
