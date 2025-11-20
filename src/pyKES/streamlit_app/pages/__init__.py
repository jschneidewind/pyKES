"""
HTE Streamlit App pages package.

This package contains the individual pages of the Streamlit application
for high-throughput experimentation data analysis.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"


# List of all available pages
__all__ = [
    "Plotting_Rates",
    "Data_Upload_and_Download",
    "Data_Visualization"
]

# Common page configuration
PAGE_CONFIG = {
    "page_title": "HTE Data Visualization",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Shared utility functions for pages if needed
def set_page_config():
    """Set the default page configuration for all pages."""
    import streamlit as st
    st.set_page_config(**PAGE_CONFIG)