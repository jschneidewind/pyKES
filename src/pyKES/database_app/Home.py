"""Entry point of the database application: `streamlit run Home.py`."""

import streamlit as st

from pyKES.database_app.components.home_component import render_home
from pyKES.database_app.config import DEFAULT_CONFIG

st.set_page_config(page_title=DEFAULT_CONFIG.title, page_icon=DEFAULT_CONFIG.icon,
                   layout="wide", initial_sidebar_state="expanded")

render_home(DEFAULT_CONFIG)
