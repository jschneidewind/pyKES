"""
The three lines every page of the application starts with.

Each page is a thin entry script whose work lives in a component, and each of
them needs the same preamble: the page configuration, the stylesheet, and the
banner that says when this is not the production deployment. Written out per
page it was written out five times and forgotten on four of them — the banner
reached Home and Admin only, so the pages where the mistake is actually made
(uploading into staging on Contribute, reading staging and believing it on
Browse and Entity) showed nothing at all. Collected here, a page added later
inherits all three by construction.
"""

import streamlit as st

from pyKES.database_app.config import DatabaseAppConfig
from pyKES.database_app.deployment import render_environment_banner
from pyKES.database_app.styling import apply_theme


# =============================================================================
# Page preamble
# =============================================================================

# Only Home opens with the sidebar out, because it is the page that explains
# where the other five are.
DEFAULT_SIDEBAR_STATE = "auto"


def configure_page(config: DatabaseAppConfig,
                   sidebar_state: str = DEFAULT_SIDEBAR_STATE) -> None:
    """
    Configure the page, apply the stylesheet and warn if this is not production.

    Parameters
    ----------
    config : DatabaseAppConfig
        Deployment configuration, for the page title and icon.
    sidebar_state : str, optional
        Initial sidebar state, as `st.set_page_config` accepts it.

    Returns
    -------
    None : None
    """
    st.set_page_config(page_title=config.title, page_icon=config.icon,
                       layout="wide", initial_sidebar_state=sidebar_state)
    apply_theme()
    render_environment_banner()
