"""Page entry point; the work lives in the component."""

from pyKES.database_app.components.search_component import render_search
from pyKES.database_app.config import DEFAULT_CONFIG
from pyKES.database_app.page import configure_page

configure_page(DEFAULT_CONFIG)

render_search(DEFAULT_CONFIG)
