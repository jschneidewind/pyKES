"""Page entry point; the work lives in the component."""

import streamlit as st

from pyKES.database_app.components.admin_component import render_admin
from pyKES.database_app.config import DEFAULT_CONFIG

st.set_page_config(page_title=DEFAULT_CONFIG.title, page_icon=DEFAULT_CONFIG.icon,
                   layout="wide")

render_admin(DEFAULT_CONFIG)
