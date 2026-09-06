"""
Plotting the traces of one entry, or comparing the traces of many.

Comparing the measurements of a subset — every test of one catalyst, every
catalyst descended from one semiconductor under one set of conditions — is the
main thing the database is for. That comparison is the *same* plot the
processing app already draws, so this module builds nothing of its own: it loads
the payloads a search selected and hands them to
`pyKES.streamlit_app.components.time_series_component`, whose
`build_trace_specifications` and `build_figure` are pure functions over
``{name: Experiment}`` and an instruction dictionary.

Payloads carry the instructions that describe their own curves, so which plots
are on offer follows from the data rather than from configuration here.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.streamlit_app.components.time_series_component import (
    INSTRUCTION_KEY,
    build_figure,
    build_trace_specifications,
)


# =============================================================================
# Display constants
# =============================================================================

# Above this many entries a comparison plot stops being readable and the
# payload loading stops being instant, so the panel asks for a narrower search
# instead of drawing it.
MAX_COMPARISON_ENTRIES = 50

# Curves offered first when the user has not chosen any yet.
DEFAULT_CURVE_COUNT = 1


# =============================================================================
# Loading
# =============================================================================

@st.cache_data(show_spinner=False)
def load_payload_experiments(payload_files: Tuple[str, ...]) -> Tuple[Dict, Dict]:
    """
    Load a set of payloads into the shape the plotting functions expect.

    Parameters
    ----------
    payload_files : tuple of str
        Absolute paths of the payloads to load. A tuple rather than a list
        because Streamlit's data cache keys on the arguments.

    Returns
    -------
    experiments : dict
        Mapping of experiment name to `Experiment`, across every payload.
    plotting_instruction : dict
        The ``time_series_instructions`` entry, taken from the payloads. They
        all come from the same processing app, so the first one that declares
        any is used.
    """
    experiments, instruction = {}, {}

    for payload_file in payload_files:
        if not Path(payload_file).exists():
            continue

        dataset = ExperimentalDataset.load_from_hdf5(payload_file)
        experiments.update(dataset.experiments)

        if not instruction:
            instruction = (dataset.plotting_instruction or {}).get(INSTRUCTION_KEY, {})

    return experiments, instruction


def payload_files_for(rows, payload_directory: Path) -> Tuple[str, ...]:
    """
    Collect the payload paths of the entries that have one.

    Parameters
    ----------
    rows : list of sqlite3.Row
        Entity rows from a search.
    payload_directory : Path
        Directory the payloads live in.

    Returns
    -------
    payload_files : tuple of str
        Absolute paths, in the order the rows came in.
    """
    return tuple(str(payload_directory / row["payload_path"])
                 for row in rows if row["payload_path"])


# =============================================================================
# The panel
# =============================================================================

def render_time_series_panel(payload_files: Tuple[str, ...],
                             key_prefix: str,
                             allow_deselection: bool = True) -> None:
    """
    Draw the curve picker and the figure.

    Parameters
    ----------
    payload_files : tuple of str
        Payloads to plot, as chosen by the caller — a filtered search on the
        browse page, a single entry on the detail page.
    key_prefix : str
        Prefix for the widget keys, so two panels can coexist.
    allow_deselection : bool, optional
        Whether to offer a multiselect for narrowing the plotted set. The browse
        page does; the detail page shows one entry and has nothing to narrow.
        Deselecting here never changes the search itself — it only hides curves.

    Returns
    -------
    None : None
    """
    if not payload_files:
        st.info("None of these entries carry measured traces.")
        return

    experiments, instruction = load_payload_experiments(payload_files)

    if not instruction:
        st.info("These payloads declare no curves to plot. The processing app "
                "sets `time_series_instructions` in its plotting instructions.")
        return

    curve_names = list(instruction)
    selected_curves = st.multiselect(
        "Curves", curve_names, default=curve_names[:DEFAULT_CURVE_COUNT],
        key=f"{key_prefix}_curves")

    plotted = sorted(experiments)
    if allow_deselection and len(plotted) > 1:
        plotted = st.multiselect(
            f"Entries plotted ({len(experiments)} in the current selection)",
            sorted(experiments), default=sorted(experiments),
            key=f"{key_prefix}_entries",
            help="Narrows the plot only — the search results are unchanged.")

    if not selected_curves or not plotted:
        st.caption("Choose at least one curve and one entry to plot.")
        return

    specifications = build_trace_specifications(plotted, experiments,
                                                instruction, selected_curves)

    if not specifications:
        st.warning("These entries carry none of the selected curves.")
        return

    figure = build_figure(specifications, selected_curves)

    # The shared component titles its own figure, which is right on a page that
    # has nothing else on it and redundant under a section heading here.
    figure.update_layout(title_text="", margin=dict(l=10, r=10, t=30, b=10))

    st.plotly_chart(figure, width="stretch", key=f"{key_prefix}_figure")
