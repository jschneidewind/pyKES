"""
Identity and the shared database connection.

Authentication happens in the reverse proxy, not here: an unauthenticated
request never reaches this application. What the application does is read *who*
the proxy says is present, so uploads can be attributed and the owner-or-admin
rule enforced.
"""

import threading
from dataclasses import dataclass
from typing import List

import streamlit as st

from pyKES.database.index_schema import IndexPaths, open_index
from pyKES.database_app.config import DEVELOPMENT_USER, DatabaseAppConfig


@dataclass
class Identity:
    """
    Who is using the application.

    Parameters
    ----------
    name : str
        Username as supplied by the proxy.
    groups : list of str
        Groups the proxy reports.
    is_admin : bool
        Whether those groups include the admin group.
    authenticated : bool
        False when the identity came from the development fallback rather than
        from a proxy header.
    """

    name: str
    groups: List[str]
    is_admin: bool
    authenticated: bool


def read_identity(config: DatabaseAppConfig) -> Identity:
    """
    Read the signed-in user from the proxy headers.

    ``st.context.headers`` reflects the ``/_stcore/stream`` WebSocket request
    rather than the initial page load, so the proxy has to set the identity
    headers on the location that proxies the upgrade. If it does not, this
    returns the development identity and the page says so rather than silently
    attributing everything to one person.

    Parameters
    ----------
    config : DatabaseAppConfig
        Deployment settings, including whether a development fallback is allowed.

    Returns
    -------
    identity : Identity
        The current user.

    Raises
    ------
    PermissionError
        If no identity header is present and the deployment forbids the
        development fallback.
    """
    headers = st.context.headers or {}
    name = headers.get(config.user_header)

    if not name:
        if not config.allow_development_login:
            raise PermissionError(
                f"No {config.user_header} header: the application is not "
                f"behind its authenticating proxy, and development login is "
                f"disabled."
            )
        return Identity(DEVELOPMENT_USER, [config.admin_group], True, False)

    groups = [group.strip() for group in
              (headers.get(config.groups_header) or "").split(",")
              if group.strip()]

    return Identity(name, groups, config.admin_group in groups, True)


# One connection per thread. A SQLite connection may only be used by the thread
# that created it, and Streamlit runs each session's script on a thread from a
# pool — so a single cached connection raises ProgrammingError as soon as a
# second page is opened. WAL mode is what makes several connections to the same
# file cheap: readers do not block each other or the writer.
_THREAD_STATE = threading.local()


def open_shared_index(data_root: str):
    """
    Open the index for the calling thread, reusing it across reruns.

    Parameters
    ----------
    data_root : str
        Directory holding the index and its file tiers.

    Returns
    -------
    connection : sqlite3.Connection
        Connection owned by this thread, in WAL mode.
    """
    handles = getattr(_THREAD_STATE, "handles", None)
    if handles is None:
        handles = {}
        _THREAD_STATE.handles = handles

    if data_root not in handles:
        connection = open_index(IndexPaths(root=data_root))

        # Ingestion holds the write lock for a second or two on a large batch;
        # a reader that arrives meanwhile should wait rather than fail.
        connection.execute("PRAGMA busy_timeout=5000")
        handles[data_root] = connection

    return handles[data_root]


def index_paths(config: DatabaseAppConfig) -> IndexPaths:
    """
    Build the filesystem layout from the configuration.

    Parameters
    ----------
    config : DatabaseAppConfig
        Deployment settings.

    Returns
    -------
    paths : IndexPaths
        Filesystem layout.
    """
    return IndexPaths(root=config.data_root)
