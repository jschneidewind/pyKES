"""
pyKES Streamlit Components

This package provides reusable Streamlit components for building data analysis
applications. Components can be easily imported and used in external repositories
to quickly set up a pyKES-powered Streamlit application.

Available Components
-------------------
- render_home: Configurable home page with HDF5 loader
- render_data_upload: Configurable data upload page with file handlers
- render_analysis_results: Analysis results visualization page
- render_time_series: Time-series data visualization page

Example Usage
-------------
>>> from pyKES.streamlit_app.components import render_data_upload
>>> from my_app.config import UPLOAD_CONFIG
>>> render_data_upload(UPLOAD_CONFIG)

Author: pyKES Development Team
Date: 2026
"""

from .home_component import render_home
from .data_upload_component import render_data_upload
from .analysis_results_component import render_analysis_results
from .time_series_component import render_time_series

__all__ = [
    'render_home',
    'render_data_upload',
    'render_analysis_results',
    'render_time_series',
]
