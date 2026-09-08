"""
The landing page.

The processing app opens on "upload a file", which is right for a personal
dataset. A group archive has to open on "here is everything we have", because
the answer to *where is the data* must be *it is already here*.
"""

import streamlit as st

from pyKES.database.index_query import database_statistics, display_entity_type
from pyKES.database_app.config import DEFAULT_CONFIG, DatabaseAppConfig
from pyKES.database_app.deployment import (
    render_environment_banner,
    render_version_caption,
)
from pyKES.database_app.session import open_shared_index, read_identity


def render_home(config: DatabaseAppConfig = DEFAULT_CONFIG) -> None:
    """
    Render the landing page.

    Parameters
    ----------
    config : DatabaseAppConfig, optional
        Deployment settings.

    Returns
    -------
    None : None
    """
    connection = open_shared_index(str(config.data_root))
    identity = read_identity(config)

    st.title(config.title)
    render_environment_banner()

    if not identity.authenticated:
        st.warning(
            f"Running without the authenticating proxy, so everything is "
            f"attributed to **{identity.name}** and admin rights are assumed. "
            f"In a deployment nginx and Authelia sit in front and this notice "
            f"does not appear."
        )
    else:
        st.caption(f"Signed in as **{identity.name}**"
                   + ("  ·  admin" if identity.is_admin else ""))

    render_version_caption(connection, config)

    statistics = database_statistics(connection)

    if statistics["entities"] == 0:
        st.info("The database is empty. Add a measured batch or a metadata "
                "sheet on the **Contribute** page to get started.")
        return

    columns = st.columns(4)
    columns[0].metric("Entries", statistics["entities"])
    columns[1].metric("With Traces", statistics["with_payload"])
    columns[2].metric("References", statistics["edges"])
    columns[3].metric("Metadata Fields", statistics["metadata_keys"])

    if statistics["last_change"]:
        st.caption(f"Last change {statistics['last_change'][:19]} UTC")

    st.subheader("What Is in the Database")
    for entity_type, count in sorted(statistics["by_type"].items(),
                                     key=lambda item: -item[1]):
        st.write(f"- **{count}** · {display_entity_type(entity_type)}")

    st.markdown(
        "---\n"
        "- **Browse & Search** — filter on any field, including ones inherited "
        "through references. The filters end up in the URL, so a search is a "
        "link you can share.\n"
        "- **Entry** — one record: its metadata, what it references, what "
        "references it, and its traces.\n"
        "- **Property map** — any quantity against any other, across everything.\n"
        "- **Contribute** — add a measured batch or a metadata sheet.\n"
        "- **Admin** — reference health, key drift, the upload log."
    )
