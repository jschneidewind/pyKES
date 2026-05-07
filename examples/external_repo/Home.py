"""Entry point that delegates to the reusable pyKES home component."""

from config import PYKES_CONFIG
from pyKES.streamlit_app.components import render_home


render_home(PYKES_CONFIG.home_config)
