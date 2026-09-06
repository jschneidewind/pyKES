"""Reusable pages of the database application."""

from pyKES.database_app.components.admin_component import render_admin
from pyKES.database_app.components.entity_component import render_entity
from pyKES.database_app.components.property_map_component import render_property_map
from pyKES.database_app.components.search_component import render_search
from pyKES.database_app.components.upload_component import render_upload

__all__ = ["render_admin", "render_entity", "render_property_map",
           "render_search", "render_upload"]
