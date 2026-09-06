"""Page entry point; the work lives in the component."""

import streamlit as st

from pyKES.database_app.components.search_component import render_search
from pyKES.database_app.config import DEFAULT_CONFIG
from pyKES.database_app.styling import apply_theme

st.set_page_config(page_title=DEFAULT_CONFIG.title, page_icon=DEFAULT_CONFIG.icon,
                   layout="wide")
apply_theme()

render_search(DEFAULT_CONFIG)
