"""Data Upload page — delegates to the reusable pyKES component."""

from config import PYKES_CONFIG
from pyKES.streamlit_app.components import render_data_upload


render_data_upload(PYKES_CONFIG.data_upload_config)
