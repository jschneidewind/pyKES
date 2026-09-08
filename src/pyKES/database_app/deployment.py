"""
What version of the application is running, and against which data.

An update that silently did not take — a container that did not roll, a
service that restarted into the old release, a data root that is not the one
anybody meant — looks exactly like an update that did. Nothing in the
application said which version was live, so answering that needed an SSH
session and a guess. These four facts are cheap to render and make every
other deployment question diagnosable from the browser that is already open.
"""

from typing import Dict

import streamlit as st

from pyKES.database.index_schema import (
    INDEX_SCHEMA_VERSION,
    read_index_schema_version,
)
from pyKES.database_app.config import (
    DEVELOPMENT_ENVIRONMENT,
    PRODUCTION_ENVIRONMENT,
    DatabaseAppConfig,
    setting_from_environment,
)
from pyKES.utilities.version_information import get_pykes_version


# Characters of the commit shown. Enough to identify a build, short enough to
# sit in a caption.
SHA_LENGTH = 8


def deployment_environment() -> str:
    """
    Read which deployment this is.

    Returns
    -------
    environment : str
        Value of PHOTOCAT_ENV, or `DEVELOPMENT_ENVIRONMENT` when unset — so
        an instance that forgot to say it is production is treated as one that
        is not, rather than the other way round.
    """
    return setting_from_environment("ENV", DEVELOPMENT_ENVIRONMENT)


def deployment_summary(connection, config: DatabaseAppConfig) -> Dict[str, str]:
    """
    Collect what identifies this running deployment.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    config : DatabaseAppConfig
        Deployment settings.

    Returns
    -------
    summary : dict
        Version of the running code, the image and commit it was built from
        where those are set, the index schema version recorded on disk against
        the one this code writes, and the data root in use.
    """
    return {
        "pykes_version": get_pykes_version(),
        "image_tag": setting_from_environment("IMAGE_TAG", ""),
        "git_sha": setting_from_environment("GIT_SHA", "")[:SHA_LENGTH],
        "environment": deployment_environment(),
        "recorded_schema": read_index_schema_version(connection) or "none",
        "code_schema": INDEX_SCHEMA_VERSION,
        "data_root": str(config.data_root),
    }


def render_version_caption(connection, config: DatabaseAppConfig) -> None:
    """
    Render the running version as one line of small print.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    config : DatabaseAppConfig
        Deployment settings.

    Returns
    -------
    None : None
    """
    summary = deployment_summary(connection, config)

    parts = [f"pyKES {summary['pykes_version']}"]
    if summary["image_tag"]:
        parts.append(f"image {summary['image_tag']}")
    if summary["git_sha"]:
        parts.append(summary["git_sha"])
    parts.append(f"index schema {summary['recorded_schema']}")

    # They differ only while an index is mid-migration or opened for repair,
    # which is worth saying out loud rather than leaving to be inferred.
    if summary["recorded_schema"] != summary["code_schema"]:
        parts.append(f"code expects {summary['code_schema']}")

    st.caption("  ·  ".join(parts))


def render_environment_banner() -> None:
    """
    Say so, unmissably, when this is not the production deployment.

    The staging instance runs the same image against a copy of the data, and
    the one mistake that costs real work is uploading into it — or reading it
    and believing it. This is also the backstop behind the guards that keep a
    development override off the server: if all of them fail, the page says so.

    Returns
    -------
    None : None
    """
    environment = deployment_environment()

    if environment == PRODUCTION_ENVIRONMENT:
        return

    st.error(
        f"**{environment.upper()}** — this is not the production database. "
        f"Anything added here is not part of the group's record.",
        icon="🧪",
    )
