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

from functools import partial

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

# Suffix of the column holding an instruction's uncertainty. A column of its
# own rather than a 'value ± error' string, so both stay sortable numbers.
# Bare rather than parenthesised, because the unit is parenthesised onto the
# header after it and 'rate ± (umol / h)' reads better than 'rate (±) (umol / h)'.
ERROR_COLUMN_SUFFIX = ' ±'

# Format spec applied when an instruction does not define its own 'format'
DEFAULT_VALUE_FORMAT = '.4g'

# Stands in for a result that cannot be resolved. Reached by the formatter,
# and so by the CSV export and the tests, but *not* by the rendered grid: a
# cell whose underlying value is missing is drawn as "None" by Streamlit
# whatever display value a Styler supplies — the same as every other table in
# the app, where a blank overview cell reads "None" too.
MISSING_VALUE_PLACEHOLDER = '—'

# Prefix of the per-experiment checkbox widget keys (must not collide with
# the checkbox keys of the time-series page, which shares the selection state)
CHECKBOX_KEY_PREFIX = 'results_table_checkbox_'

# Key of the multiselect mirroring the checkbox selection
MULTISELECT_KEY = 'results_table_selected_experiments'

# Keys of the two column selectors above the table
RESULTS_SELECTION_KEY = 'results_table_selected_results'
METADATA_SELECTION_KEY = 'results_table_selected_metadata'


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


def resolve_result_number(experiment, path: str, unit=None):
    """
    Resolve one result path on one experiment to a value fit for a table cell.

    Parameters
    ----------
    experiment : Experiment
        Experiment the path is resolved against.
    path : str
        Slash-separated attribute path, e.g. ``'processed_data/max_rate'``.
    unit : str or None, optional
        Target unit, applied to Quantity values.

    Returns
    -------
    value : float or str or None
        The number, the string where the result genuinely is one, or None
        when the path cannot be resolved. Deliberately *not* formatted: the
        DataFrame keeps numbers so the table sorts by magnitude, and the
        formatting happens in the column configuration at render time.
    """
    value = coerce_to_scalar(resolve_result_value(experiment, path))

    if value is None:
        return None

    return convert_quantity(value, unit)


def error_column_name(label: str) -> str:
    """
    Name the column holding an instruction's uncertainty.

    Parameters
    ----------
    label : str
        Display name of the instruction.

    Returns
    -------
    str
        Column name of the matching uncertainty.
    """
    return f"{label}{ERROR_COLUMN_SUFFIX}"


def column_header(label: str, unit) -> str:
    """
    Build a column header, carrying the unit where the instruction names one.

    Parameters
    ----------
    label : str
        Display name of the instruction, or of its uncertainty.
    unit : str or None
        Target unit of the instruction.

    Returns
    -------
    str
        The header shown above the column.

    Notes
    -----
    The unit belongs to the column, not to each of its cells: repeating it in
    every row costs width, makes the numbers harder to compare down the
    column, and puts text in a cell whose value is a number. `unit` still
    decides what `Quantity` values are converted into — that is its other,
    load-bearing job.
    """

    return f"{label} ({unit})" if unit else label


def select_instructions(results_table_instructions: dict, selected_results: list) -> dict:
    """
    Keep the requested instructions, in the order the app declared them.

    Parameters
    ----------
    results_table_instructions : dict
        Mapping of display name to instruction entry.
    selected_results : list of str
        Instruction names the user asked for.

    Returns
    -------
    dict
        The requested subset of the instructions.
    """
    return {label: result_config
            for label, result_config in results_table_instructions.items()
            if label in selected_results}


def result_column_names(results_table_instructions: dict) -> list:
    """
    List the table columns a set of instructions produces.

    Parameters
    ----------
    results_table_instructions : dict
        Mapping of display name to instruction entry.

    Returns
    -------
    list of str
        One column per instruction, each followed by its uncertainty column
        where the instruction defines an ``'error'`` path.
    """
    column_names = []

    for label, result_config in results_table_instructions.items():
        unit = result_config.get('unit', None)
        column_names.append(column_header(label, unit))

        if result_config.get('error', None):
            column_names.append(column_header(error_column_name(label), unit))

    return column_names


def build_row_values(experiment, results_table_instructions: dict) -> list:
    """
    Resolve one experiment's row of the results table.

    Parameters
    ----------
    experiment : Experiment
        Experiment the results are read from.
    results_table_instructions : dict
        Instruction entries with the required key ``'result'`` and the
        optional keys ``'unit'`` (target unit for Quantity values),
        ``'format'`` (Python format spec, applied at render time) and
        ``'error'`` (path to an uncertainty, given a column of its own).

    Returns
    -------
    list
        Values matching `result_column_names`, None where unresolvable.
    """
    row_values = []

    for result_config in results_table_instructions.values():
        unit = result_config.get('unit', None)
        row_values.append(resolve_result_number(experiment, result_config['result'], unit))

        error_path = result_config.get('error', None)
        if error_path:
            row_values.append(resolve_result_number(experiment, error_path, unit))

    return row_values


def build_results_table(
    selected_experiments: list,
    experiments: dict,
    results_table_instructions: dict
) -> pd.DataFrame:
    """
    Assemble the results table for the selected experiments.

    Every instruction becomes a column so that the table keeps its shape
    across experiments; cells that cannot be resolved stay empty.

    Parameters
    ----------
    selected_experiments : list of str
        Names of the experiments to show, in display order — one row each.
    experiments : dict
        Mapping of experiment name to experiment object.
    results_table_instructions : dict
        Mapping of display name to instruction entry (see `build_row_values`).

    Returns
    -------
    table : pandas.DataFrame
        Table indexed by experiment name, holding numbers rather than
        formatted text so that sorting a column compares magnitudes.
    """
    table_rows = {
        exp_name: build_row_values(experiments[exp_name], results_table_instructions)
        for exp_name in selected_experiments
        if exp_name in experiments
    }

    # Building from an explicit column list keeps the column order and the
    # table shape even when no experiment is selected.
    return pd.DataFrame.from_dict(table_rows,
                                  orient='index',
                                  columns=result_column_names(results_table_instructions))


def available_metadata_columns(overview_df: pd.DataFrame) -> list:
    """
    List the overview columns that can be shown beside the results.

    Parameters
    ----------
    overview_df : pandas.DataFrame
        Overview sheet of the dataset.

    Returns
    -------
    list of str
        Every column but the one naming the experiments, which is already the
        table's index. Empty when the sheet is missing that column, since
        there is then nothing to join the metadata on.
    """
    if overview_df.empty or EXPERIMENT_NAME_COLUMN not in overview_df.columns:
        return []

    return [column for column in overview_df.columns if column != EXPERIMENT_NAME_COLUMN]


def join_metadata_columns(table: pd.DataFrame,
                          overview_df: pd.DataFrame,
                          metadata_columns: list) -> pd.DataFrame:
    """
    Put the requested metadata columns to the left of the results.

    Parameters
    ----------
    table : pandas.DataFrame
        Results table, indexed by experiment name.
    overview_df : pandas.DataFrame
        Overview sheet the metadata is taken from.
    metadata_columns : list of str
        Overview columns to show.

    Returns
    -------
    pandas.DataFrame
        The table with the metadata joined on. The columns are taken from
        ``overview_df`` unchanged, so each keeps its own dtype and therefore
        sorts correctly without further handling.
    """
    if not metadata_columns:
        return table

    metadata = overview_df.set_index(EXPERIMENT_NAME_COLUMN)[metadata_columns]

    return metadata.join(table, how='right')


def format_result_cell(value, format_spec: str = DEFAULT_VALUE_FORMAT) -> str:
    """
    Render one result value the way the instruction asks for.

    Parameters
    ----------
    value : float or str or None
        Resolved value of the cell.
    format_spec : str, optional
        Python format spec from the instruction.

    Returns
    -------
    str
        Formatted cell text: the number alone. The unit is in the header, see
        `column_header`.

    Notes
    -----
    Formatting happens here, on the display side of a Styler, rather than
    through ``st.column_config.NumberColumn``: that formats printf-style and
    has no ``g`` conversion, so ``'.4g'`` — the default, and the spec that
    earns its keep across the many orders of magnitude a rate spans — came
    out as ``0.00001234`` where Python writes ``1.234e-05``. A Styler keeps
    the frame numeric, so the header sort still compares magnitudes.
    """

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return MISSING_VALUE_PLACEHOLDER

    if isinstance(value, str):
        return value

    return format(value, format_spec)


def result_cell_formatters(results_table_instructions: dict) -> dict:
    """
    Build the per-column display formatters of the results table.

    Parameters
    ----------
    results_table_instructions : dict
        Instructions the result columns were built from.

    Returns
    -------
    dict
        Mapping of column name to a one-argument formatter, covering the
        result columns and their uncertainty columns. Metadata columns are
        left out so they render as the overview sheet holds them.
    """

    formatters = {}

    for label, result_config in results_table_instructions.items():
        format_spec = result_config.get('format', DEFAULT_VALUE_FORMAT)
        unit = result_config.get('unit', None)

        for name in (label, error_column_name(label)):
            formatters[column_header(name, unit)] = partial(format_result_cell,
                                                            format_spec=format_spec)

    return formatters


def style_results_table(table: pd.DataFrame, results_table_instructions: dict):
    """
    Attach the display formatting to an assembled results table.

    Parameters
    ----------
    table : pandas.DataFrame
        Table of numbers, as `build_results_table` returns it.
    results_table_instructions : dict
        Instructions the result columns were built from.

    Returns
    -------
    pandas.io.formats.style.Styler
        The table with its result columns formatted. Handing this to
        `st.dataframe` shows the formatted text while the header sort still
        works on the numbers underneath.
    """

    formatters = {column: formatter
                  for column, formatter in result_cell_formatters(results_table_instructions).items()
                  if column in table.columns}

    return table.style.format(formatter=formatters)


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


def render_column_selection(results_table_instructions: dict,
                            overview_df: pd.DataFrame) -> dict:
    """
    Render the two column selectors above the table.

    Parameters
    ----------
    results_table_instructions : dict
        Every analysis result the dataset defines.
    overview_df : pandas.DataFrame
        Overview sheet supplying the metadata columns on offer.

    Returns
    -------
    selected_instructions : dict
        The instructions the user asked for.
    selected_metadata : list of str
        The overview columns the user asked for.
    """
    selection_columns = st.columns(2)

    with selection_columns[0]:
        selected_results = st.multiselect(
            "Results to show",
            options=list(results_table_instructions.keys()),
            default=list(results_table_instructions.keys()),
            key=RESULTS_SELECTION_KEY,
        )

    metadata_options = available_metadata_columns(overview_df)

    with selection_columns[1]:
        selected_metadata = st.multiselect(
            "Metadata to show",
            options=metadata_options,
            default=[],
            key=METADATA_SELECTION_KEY,
            help="Overview-sheet columns shown to the left of the results.",
        )

        if not metadata_options:
            st.caption("No metadata on offer: the dataset carries no overview sheet "
                       f"with an '{EXPERIMENT_NAME_COLUMN}' column to join it on.")

    return select_instructions(results_table_instructions, selected_results), selected_metadata


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
        - **Results to show**: which analysis results become columns
        - **Metadata to show**: overview-sheet columns joined to the left of the results
        - Each selected experiment becomes one row, labelled with its name
        - Empty cells mean that result is not available for that experiment
        - Click a column header to sort by it — the cells hold numbers, so the
          order follows the magnitude rather than the leading digit

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
        `'unit'` (target unit: `Quantity` values are converted into it, and it
        names the column — a cell holds the number alone), `'format'` (Python
        format spec, default `'{DEFAULT_VALUE_FORMAT}'`) and `'error'` (path to
        an uncertainty, which becomes its own `label{ERROR_COLUMN_SUFFIX}` column).

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

        selected_instructions, selected_metadata = render_column_selection(
            results_table_instructions, experimental_dataset.overview_df)

        if not st.session_state.selected_experiments:
            st.info("Select one or more experiments to display their analysis results.")
            return

        results_table = build_results_table(
            st.session_state.selected_experiments,
            experimental_dataset.experiments,
            selected_instructions
        )
        results_table.index.name = EXPERIMENT_NAME_COLUMN

        results_table = join_metadata_columns(results_table,
                                              experimental_dataset.overview_df,
                                              selected_metadata)

        st.dataframe(style_results_table(results_table, selected_instructions),
                     width='stretch')

        st.download_button(
            label="📥 Download Table as CSV",
            data=results_table.to_csv().encode('utf-8'),
            file_name="analysis_results_table.csv",
            mime="text/csv"
        )

    render_help_section()

    st.markdown("---")
    st.caption("pyKES Analysis Results Table | Powered by Streamlit")
