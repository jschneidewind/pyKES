"""
Numerical Analysis Results Table

This module provides a Streamlit interface for inspecting the numerical analysis
results of individual experiments side by side. Experiments are selected through
group expanders (as in the time-series page) and their results are rendered as a
table whose rows are the experiments and whose columns are the analysis
quantities.

The columns are defined by the external app through the
``results_table_instructions`` entry of ``ExperimentalDataset.plotting_instruction``::

    'results_table_instructions': {
        'Max. rate (mmol/h/g)': {'result': 'processed_data/max_rate_mmol_h_g'},
        'Apparent quantum yield (%)': {'result': 'processed_data/apparent_quantum_yield',
                                       'format': '.2f'},
    }

Author: pyKES Development Team
Date: 27 August 2026
"""

import streamlit as st
import numpy as np
import pandas as pd

from pyKES.utilities.resolve_attributes import resolve_experiment_attributes
from pyKES.utilities.unit_handler import Quantity


# =============================================================================
# Display constants
# =============================================================================

# Key of the plotting_instruction entry that defines the table columns
INSTRUCTION_KEY = 'results_table_instructions'

# Header of the left-most column listing the experiment names
EXPERIMENT_NAME_COLUMN = 'Experiment'

# Shown whenever a result cannot be resolved for a given experiment
MISSING_VALUE_PLACEHOLDER = '—'

# Format spec applied when an instruction does not define its own 'format'
DEFAULT_VALUE_FORMAT = '.4g'

# Prefix of the per-experiment checkbox widget keys (must not collide with
# the checkbox keys of the time-series page, which shares the selection state)
CHECKBOX_KEY_PREFIX = 'results_table_checkbox_'

# Key of the multiselect mirroring the checkbox selection
MULTISELECT_KEY = 'results_table_selected_experiments'


# =============================================================================
# Selection state handling
# =============================================================================

def update_selection(exp_name: str, checkbox_key: str) -> None:
    """
    Synchronise the shared experiment selection with a checkbox toggle.

    Parameters
    ----------
    exp_name : str
        Name of the experiment the checkbox belongs to.
    checkbox_key : str
        Session-state key of the checkbox widget.

    Returns
    -------
    None : None
        ``st.session_state.selected_experiments`` is updated in-place.
    """
    selected = st.session_state[checkbox_key]

    if selected and exp_name not in st.session_state.selected_experiments:
        st.session_state.selected_experiments.append(exp_name)
    elif not selected and exp_name in st.session_state.selected_experiments:
        st.session_state.selected_experiments.remove(exp_name)


def sync_from_multiselect(multiselect_key: str) -> None:
    """
    Adopt the multiselect content as the shared experiment selection.

    Parameters
    ----------
    multiselect_key : str
        Session-state key of the multiselect widget.

    Returns
    -------
    None : None
        ``st.session_state.selected_experiments`` is updated in-place.
    """
    st.session_state.selected_experiments = st.session_state[multiselect_key]


# =============================================================================
# Value resolution and formatting
# =============================================================================

def resolve_result_value(experiment, path: str):
    """
    Resolve a single result path on an experiment.

    Parameters
    ----------
    experiment : Experiment
        Experiment object the path is resolved against.
    path : str
        Slash-separated attribute path, e.g. ``'processed_data/max_rate'``.

    Returns
    -------
    value : Any or None
        Resolved value, or None if the path is absent for this experiment.
    """
    resolved = resolve_experiment_attributes({'value': path}, experiment, mode='permissive')

    return resolved.get('value', None)


def coerce_to_scalar(value):
    """
    Reduce a resolved value to a plain scalar where that is unambiguous.

    Single-element arrays are unwrapped; longer arrays have no meaningful
    tabular representation and are rejected.

    Parameters
    ----------
    value : Any
        Value returned by the attribute resolution.

    Returns
    -------
    scalar : Any or None
        Scalar value, or None if the value is not representable in a cell.
    """
    if isinstance(value, Quantity):
        return value

    array_value = np.asarray(value)

    if array_value.ndim == 0:
        return array_value.item()
    if array_value.size == 1:
        return array_value.reshape(-1)[0].item()

    return None


def convert_quantity(value, unit: str):
    """
    Express a value as a number, converting Quantity objects to `unit`.

    Parameters
    ----------
    value : Quantity or float or str
        Value to express numerically.
    unit : str or None
        Target unit. Only meaningful for Quantity values; a Quantity without
        a target unit falls back to its supplied unit.

    Returns
    -------
    number : float or str
        Numeric value in the requested unit, or the value unchanged if it is
        not a Quantity.
    """
    if not isinstance(value, Quantity):
        return value

    return value.unit[unit] if unit else value.supplied_value


def format_result_value(value, format_spec: str) -> str:
    """
    Render a single value with the requested format spec.

    Parameters
    ----------
    value : float or str
        Value to render.
    format_spec : str
        Python format spec (e.g. ``'.3f'``); ignored for non-numeric values.

    Returns
    -------
    text : str
        Formatted value.
    """
    if isinstance(value, str):
        return value

    return format(value, format_spec)


def build_cell_text(experiment, result_config: dict) -> str:
    """
    Build the table cell for one analysis result of one experiment.

    Parameters
    ----------
    experiment : Experiment
        Experiment the result is read from.
    result_config : dict
        Instruction entry with the required key ``'result'`` and the optional
        keys ``'unit'`` (target unit for Quantity values), ``'format'``
        (Python format spec) and ``'error'`` (path to an uncertainty, rendered
        as ``value ± error``).

    Returns
    -------
    text : str
        Formatted cell content, or the missing-value placeholder.
    """
    value = coerce_to_scalar(resolve_result_value(experiment, result_config['result']))

    if value is None:
        return MISSING_VALUE_PLACEHOLDER

    unit = result_config.get('unit', None)
    format_spec = result_config.get('format', DEFAULT_VALUE_FORMAT)
    cell_text = format_result_value(convert_quantity(value, unit), format_spec)

    error_path = result_config.get('error', None)
    if error_path:
        error_value = coerce_to_scalar(resolve_result_value(experiment, error_path))
        if error_value is not None:
            error_text = format_result_value(convert_quantity(error_value, unit), format_spec)
            cell_text = f"{cell_text} ± {error_text}"

    # A unit is only appended for plain numbers; Quantity values were already
    # converted into that unit above, so the label applies in both cases.
    if unit:
        cell_text = f"{cell_text} {unit}"

    return cell_text


def build_results_table(
    selected_experiments: list,
    experiments: dict,
    results_table_instructions: dict
) -> pd.DataFrame:
    """
    Assemble the results table for the selected experiments.

    Every instruction key becomes a column so that the table keeps its shape
    across experiments; cells that cannot be resolved show a placeholder.

    Parameters
    ----------
    selected_experiments : list of str
        Names of the experiments to show, in display order — one row each.
    experiments : dict
        Mapping of experiment name to experiment object.
    results_table_instructions : dict
        Mapping of display name to instruction entry (see `build_cell_text`).

    Returns
    -------
    table : pandas.DataFrame
        Table indexed by experiment name with one column per analysis result.
    """
    table_rows = {
        exp_name: [
            build_cell_text(experiments[exp_name], result_config)
            for result_config in results_table_instructions.values()
        ]
        for exp_name in selected_experiments
        if exp_name in experiments
    }

    # Building from an explicit column list keeps the column order and the
    # table shape even when no experiment is selected.
    return pd.DataFrame.from_dict(table_rows,
                                  orient='index',
                                  columns=list(results_table_instructions.keys()))


# =============================================================================
# UI rendering
# =============================================================================

def group_experiments(experiments: dict) -> dict:
    """
    Bucket experiments by their group attribute.

    Parameters
    ----------
    experiments : dict
        Mapping of experiment name to experiment object.

    Returns
    -------
    experiments_by_group : dict
        Mapping of group name to list of (experiment name, experiment) tuples.
    """
    experiments_by_group = {}

    for exp_name, exp_data in experiments.items():
        experiments_by_group.setdefault(exp_data.group, []).append((exp_name, exp_data))

    return experiments_by_group


def build_checkbox_label(exp_name: str, exp_data, group_name: str, group_mapping: dict) -> str:
    """
    Build the checkbox label, annotated with the group's metadata value.

    Parameters
    ----------
    exp_name : str
        Name of the experiment.
    exp_data : Experiment
        Experiment object providing the metadata.
    group_name : str
        Group the experiment belongs to.
    group_mapping : dict
        Mapping of group name to the metadata path characterising that group.

    Returns
    -------
    label : str
        Experiment name, followed by the metadata value in parentheses when
        the group defines one.
    """
    metadata_path = group_mapping.get(group_name, None)

    if metadata_path is None:
        return exp_name

    metadata_value = resolve_experiment_attributes(
        {group_name: metadata_path},
        exp_data,
        mode='permissive'
    )

    if group_name not in metadata_value:
        return exp_name

    return f"{exp_name} ({metadata_value[group_name]})"


def render_group_selection(experiments_by_group: dict, group_mapping: dict) -> None:
    """
    Render the group expanders with one checkbox per experiment.

    Parameters
    ----------
    experiments_by_group : dict
        Mapping of group name to list of (experiment name, experiment) tuples.
    group_mapping : dict
        Mapping of group name to the metadata path characterising that group.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """
    st.header("Group Selection")

    for group_name in sorted(experiments_by_group.keys()):
        experiments_in_group = experiments_by_group[group_name]

        with st.expander(f"{group_name} (n={len(experiments_in_group)})", expanded=False):

            for exp_name, exp_data in sorted(experiments_in_group, key=lambda entry: entry[0]):
                checkbox_key = f"{CHECKBOX_KEY_PREFIX}{exp_name}"
                st.session_state[checkbox_key] = exp_name in st.session_state.selected_experiments

                st.checkbox(
                    build_checkbox_label(exp_name, exp_data, group_name, group_mapping),
                    key=checkbox_key,
                    on_change=update_selection,
                    args=(exp_name, checkbox_key)
                )


def render_experiment_multiselect(all_experiment_names: list) -> None:
    """
    Render the multiselect mirroring the checkbox selection.

    Parameters
    ----------
    all_experiment_names : list of str
        Names of all experiments in the dataset.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """
    st.session_state[MULTISELECT_KEY] = list(st.session_state.selected_experiments)

    st.multiselect(
        "Selected Experiments",
        options=all_experiment_names,
        key=MULTISELECT_KEY,
        on_change=sync_from_multiselect,
        args=(MULTISELECT_KEY,)
    )


def render_help_section() -> None:
    """
    Render the explanatory section at the bottom of the page.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """
    st.markdown("---")
    st.header("How to Use This Tool")

    with st.expander("📚 Detailed Explanation", expanded=False):
        st.markdown(f"""
        ### Understanding the Analysis Results Table

        This tool shows the numerical analysis results of individual experiments
        side by side.

        #### 1. Group Selection (Left Panel)
        - Experiments are organized by their group attribute
        - Click on a group to expand it and see all experiments it contains
        - Where a group defines a characteristic metadata value, it is shown in parentheses
        - Select checkboxes to add experiments to the table

        #### 2. Results Table (Right Panel)
        - **Selected Experiments**: shows and allows manual selection/deselection
        - Each selected experiment becomes one row, labelled with its name
        - Each analysis result becomes one column
        - Cells reading `{MISSING_VALUE_PLACEHOLDER}` mean that result is not available
          for that experiment

        #### 3. Which results are shown
        The columns are defined by the upstream app through the
        `'{INSTRUCTION_KEY}'` entry of the dataset's `plotting_instruction`:

        ```python
        '{INSTRUCTION_KEY}': {{
            'Max. rate (mmol/h/g)': {{'result': 'processed_data/max_rate_mmol_h_g'}},
            'Apparent quantum yield (%)': {{'result': 'processed_data/aqy',
                                            'format': '.2f'}},
        }}
        ```

        Besides the required `'result'` path, each entry may define
        `'unit'` (target unit, used to convert `Quantity` values),
        `'format'` (Python format spec, default `'{DEFAULT_VALUE_FORMAT}'`) and
        `'error'` (path to an uncertainty, rendered as `value ± error`).

        #### Tips
        - The experiment selection is shared with the Time-Series page
        - Use the download button to export the table as CSV
        """)


# =============================================================================
# Main Application
# =============================================================================

def render_results_table() -> None:
    """
    Render the Analysis Results Table page.

    Returns
    -------
    None : None
        The page is written to the current Streamlit script run.
    """
    st.set_page_config(
        page_title="Analysis Results Table",
        page_icon="🔢",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    col_title, col_filename = st.columns([3, 1])
    with col_title:
        st.title("Analysis Results Table")
    with col_filename:
        if st.session_state.hdf5_filename:
            st.markdown(f"<p style='text-align: right; font-size: 0.8em; color: gray; margin-top: 1.5em;'>{st.session_state.hdf5_filename}</p>", unsafe_allow_html=True)

    if st.session_state.experimental_dataset is None:
        st.info("Please upload a HDF5 file on the Home page first.")
        return

    experimental_dataset = st.session_state.experimental_dataset
    plotting_instruction = experimental_dataset.plotting_instruction

    if INSTRUCTION_KEY not in plotting_instruction:
        st.error(f"No '{INSTRUCTION_KEY}' found in plotting_instruction")
        return

    results_table_instructions = plotting_instruction[INSTRUCTION_KEY]

    # Selection state is shared with the time-series page
    if 'selected_experiments' not in st.session_state:
        st.session_state.selected_experiments = []

    col1, col2 = st.columns([1, 2])

    with col1:
        render_group_selection(
            group_experiments(experimental_dataset.experiments),
            experimental_dataset.group_mapping
        )

    with col2:
        st.header("Results Table")

        render_experiment_multiselect(sorted(experimental_dataset.experiments.keys()))

        if not st.session_state.selected_experiments:
            st.info("Select one or more experiments to display their analysis results.")
            return

        results_table = build_results_table(
            st.session_state.selected_experiments,
            experimental_dataset.experiments,
            results_table_instructions
        )
        results_table.index.name = EXPERIMENT_NAME_COLUMN

        st.dataframe(results_table, width='stretch')

        st.download_button(
            label="📥 Download Table as CSV",
            data=results_table.to_csv().encode('utf-8'),
            file_name="analysis_results_table.csv",
            mime="text/csv"
        )

    render_help_section()

    st.markdown("---")
    st.caption("pyKES Analysis Results Table | Powered by Streamlit")
