"""One warning, shared by every page that colours its curves by experiment.

`pyKES.plotting.plot_colors` draws an unusable colour in blue rather than
raising, which keeps the figure on the screen but would otherwise be silent —
two experiments quietly sharing a colour looks like a plotting bug, not like a
typo in a spreadsheet cell. The fail-fast rule cannot be honoured literally
here (the fallback is the requested behaviour), so the page says what it did.
"""

import streamlit as st

from pyKES.plotting.plot_colors import FALLBACK_PLOT_COLOR, unrecognized_plot_colors


def render_unrecognized_color_warning(experiments: dict) -> None:
    """
    Name the experiments whose declared colour had to be replaced.

    Parameters
    ----------
    experiments : dict
        Mapping of experiment name to experiment object.

    Returns
    -------
    None : None
        Nothing is written when every colour is usable.
    """

    unusable_colors = unrecognized_plot_colors(experiments)

    if not unusable_colors:
        return

    st.warning(
        f"⚠️ Drawn in {FALLBACK_PLOT_COLOR}: these experiments declare a colour no "
        "plotting library recognises. Correct the `color` column of the overview "
        "sheet — an empty cell counts as unrecognised. "
        + ", ".join(f"**{experiment_name}** (`{declared_color}`)"
                    for experiment_name, declared_color in sorted(unusable_colors.items()))
    )
