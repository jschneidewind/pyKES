"""
The property map: any indexed quantity against any other, across everything.

This is the view that makes an archive worth more than the sum of its files.
Because it reads the materialised effective metadata, an axis can be a value
inherited three references away — quantum yield against the synthesis
temperature of the precursor the catalyst was made from — without the plot
knowing anything about the reference structure.
"""

import streamlit as st

from pyKES.database.index_query import (
    RESULT_PREFIX,
    display_entity_type,
    display_key,
    list_axis_options,
    property_map_data,
)
from pyKES.database.index_schema import ENTITY_TYPES
from pyKES.database_app.config import DEFAULT_CONFIG, DatabaseAppConfig
from pyKES.database_app.session import open_shared_index


# =============================================================================
# Display constants
# =============================================================================

# Marker size and opacity chosen so a few thousand overlapping points stay
# readable rather than saturating into a solid block.
MARKER_SIZE = 9
MARKER_OPACITY = 0.75

FIGURE_HEIGHT = 560


def render_property_map(config: DatabaseAppConfig = DEFAULT_CONFIG) -> None:
    """
    Render the property-map page.

    Parameters
    ----------
    config : DatabaseAppConfig, optional
        Deployment settings.

    Returns
    -------
    None : None
    """
    import plotly.express as px

    connection = open_shared_index(str(config.data_root))

    st.title("Property map")
    st.caption("Any indexed quantity against any other, across every entry — "
               "including values inherited through references.")

    entity_type = st.selectbox("Kind of entry", ENTITY_TYPES,
                               index=ENTITY_TYPES.index(config.default_entity_type),
                               format_func=display_entity_type)

    options = list_axis_options(connection, entity_type)
    if len(options) < 2:
        st.info("At least two numeric quantities have to be indexed before a "
                "property map can be drawn. Upload some data first.")
        return

    left, middle, right = st.columns(3)
    x_key = left.selectbox("X Axis", options, index=0,
                           format_func=display_key)
    y_key = middle.selectbox("Y Axis", options,
                             index=1 if len(options) > 1 else 0,
                             format_func=display_key)
    color_key = right.selectbox("Colour By", ["(none)"] + options,
                                format_func=lambda key: (
                                    key if key == "(none)" else display_key(key)))

    frame = property_map_data(connection, x_key, y_key,
                              None if color_key == "(none)" else color_key,
                              entity_type=entity_type)

    if frame.empty:
        st.warning("No entries carry both of these quantities.")
        return

    figure = px.scatter(
        frame, x="x", y="y",
        color="color" if "color" in frame.columns else None,
        hover_name="entity_id",
        labels={"x": display_key(x_key), "y": display_key(y_key),
                "color": display_key(color_key) if color_key != "(none)" else ""},
    )
    figure.update_traces(marker=dict(size=MARKER_SIZE, opacity=MARKER_OPACITY))
    figure.update_layout(height=FIGURE_HEIGHT,
                         margin=dict(l=10, r=10, t=30, b=10))

    st.plotly_chart(figure, width="stretch")
    st.caption(f"{len(frame)} of the "
               f"{display_entity_type(entity_type).lower()} entries carry both "
               f"quantities.")

    st.download_button("Download Points as CSV",
                       data=frame.to_csv(index=False).encode("utf-8"),
                       file_name="photocat_property_map.csv", mime="text/csv")
