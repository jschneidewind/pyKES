"""Page entry point; the work lives in the component."""

from pyKES.database_app.components.admin_component import render_admin
from pyKES.database_app.config import DEFAULT_CONFIG
from pyKES.database_app.page import configure_page

configure_page(DEFAULT_CONFIG)

render_admin(DEFAULT_CONFIG)
