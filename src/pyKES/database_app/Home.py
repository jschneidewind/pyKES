"""Entry point of the database application: `streamlit run Home.py`."""

from pyKES.database_app.components.home_component import render_home
from pyKES.database_app.config import DEFAULT_CONFIG
from pyKES.database_app.page import configure_page

# The only page that opens with the sidebar out: it is what explains
# where the other five are.
configure_page(DEFAULT_CONFIG, sidebar_state="expanded")

render_home(DEFAULT_CONFIG)
