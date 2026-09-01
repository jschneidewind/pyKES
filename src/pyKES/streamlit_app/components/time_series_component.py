"""
Time-Series Data Visualization

This module provides a Streamlit interface for plotting the time-series data of
selected experiments. Experiments are picked through group expanders and the
curves to plot are defined by the external app through the
``time_series_instructions`` entry of ``ExperimentalDataset.plotting_instruction``:

    'time_series_instructions': {
        'Reaction (O2)': {
            'x': 'processed_data/time_reaction_s',
            'y': 'processed_data/data_reaction_umol',
            'unit_x': 'processed_data/time_unit',
            'unit_y': 'processed_data/data_unit',
            },
        'Rate (O2)': {
            'x': 'processed_data/time_reaction_s',
            'y': 'processed_data/rate_gaussian_umol_s',
            'x_point': 'processed_data/max_rate_time_s',
            'y_point': 'processed_data/max_rate_umol_s',
            'unit_x': 'processed_data/time_unit',
            'unit_y': 'processed_data/rate_gaussian_unit',
            },
    }

Besides the required ``'x'`` and ``'y'`` paths, an entry may define
``'x_point'`` / ``'y_point'`` (a single highlighted point) and ``'unit_x'`` /
``'unit_y'``. The unit entries are optional: they are either a path into the
experiment holding the unit string, or a literal unit such as ``'umol / s'``.
Where they are absent the plot behaves as before, with generic axis titles.

Curves whose y-unit differs are drawn against their own y axis, added on the
right-hand side, so that e.g. amounts (``umol``) and rates (``umol / s``) can be
displayed in one figure without one of them being squashed flat.

Author: pyKES Development Team
Date: 31 August 2026
"""

import streamlit as st
import plotly.graph_objects as go

from pyKES.utilities.resolve_attributes import resolve_experiment_attributes, resolve_path_slash


# =============================================================================
# Instruction keys
# =============================================================================

# Key of the plotting_instruction entry defining the available curves
INSTRUCTION_KEY = 'time_series_instructions'

# Instruction keys carrying units rather than data paths
X_UNIT_KEY = 'unit_x'
Y_UNIT_KEY = 'unit_y'


# =============================================================================
# Display constants
# =============================================================================

# Axis titles used where an instruction defines no unit
DEFAULT_X_AXIS_TITLE = "Time (s)"
DEFAULT_Y_AXIS_TITLE = "Value"

# Separator between the units of curves that share an axis but disagree on units
UNIT_SEPARATOR = ", "

# Fraction of the figure width freed for each additional right-hand y axis
SECONDARY_AXIS_WIDTH = 0.06

# Grid color of the primary axes; the additional y axes draw no grid, because
# several overlapping grids make the figure unreadable
GRID_COLOR = 'rgba(255,255,255,0.2)'

# Marker / line styling of the curves and of the highlighted single points
MARKER_SIZE = 5
MARKER_OPACITY = 0.7
LINE_WIDTH = 2
POINT_MARKER_SIZE = 12
POINT_MARKER_COLOR = 'red'

FIGURE_HEIGHT = 600

# Session-state keys (the experiment selection is shared with the results table)
CHECKBOX_KEY_PREFIX = 'time_series_checkbox_'
MULTISELECT_KEY = 'time_series_selected_experiments'


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
# Instruction resolution
# =============================================================================

def strip_unit_entries(plotting_instruction: dict) -> dict:
    """
    Drop the unit entries from every instruction, keeping the data paths.

    Units are resolved separately because the data resolution is all-or-nothing
    per curve: a unit path that is missing for one experiment would otherwise
    remove that experiment's curve entirely.

    Parameters
    ----------
    plotting_instruction : dict
        Mapping of curve name to its instruction entry.

    Returns
    -------
    data_instruction : dict
        The same mapping without the ``unit_x`` / ``unit_y`` entries.
    """
    return {
        plot_type: {key: value for key, value in plot_config.items()
                    if key not in (X_UNIT_KEY, Y_UNIT_KEY)}
        for plot_type, plot_config in plotting_instruction.items()
    }


def resolve_unit_label(experiment, unit_specification) -> str:
    """
    Resolve one unit entry of an instruction against an experiment.

    A specification whose first component names an attribute of the experiment
    (``processed_data``, ``metadata``, …) is treated as a path; anything else
    is a literal unit string, which lets an app write ``'umol / s'`` directly
    instead of storing it per experiment.

    Parameters
    ----------
    experiment : Experiment
        Experiment the specification is resolved against.
    unit_specification : str or None
        Path to the unit or a literal unit string.

    Returns
    -------
    unit : str or None
        Unit label, or None where no unit is defined or the declared path is
        absent for this experiment.
    """
    if not unit_specification:
        return None

    root_component = unit_specification.split('/')[0].strip()

    if not hasattr(experiment, root_component):
        return unit_specification

    try:
        return str(resolve_path_slash(unit_specification, experiment))
    except (ValueError, KeyError, AttributeError):
        return None


def build_trace_specifications(selected_experiments: list,
                               experiments: dict,
                               plotting_instruction: dict,
                               selected_plot_types: list) -> list:
    """
    Collect everything needed to draw the selected curves.

    Parameters
    ----------
    selected_experiments : list of str
        Names of the experiments to plot, in display order.
    experiments : dict
        Mapping of experiment name to experiment object.
    plotting_instruction : dict
        Mapping of curve name to its instruction entry.
    selected_plot_types : list of str
        Curve names the user asked for.

    Returns
    -------
    trace_specifications : list of dict
        One entry per drawable curve, holding the experiment name, curve name,
        color, the resolved data, and the resolved x/y units (None where
        undefined).
    """
    data_instruction = strip_unit_entries(plotting_instruction)
    trace_specifications = []

    for exp_name in selected_experiments:
        if exp_name not in experiments:
            continue

        experiment = experiments[exp_name]
        resolved_plots = resolve_experiment_attributes(data_instruction, experiment, mode='permissive')

        for plot_type in selected_plot_types:
            if plot_type not in resolved_plots:
                continue

            plot_config = plotting_instruction[plot_type]

            trace_specifications.append({
                'experiment_name': exp_name,
                'plot_type': plot_type,
                'color': experiment.color,
                'data': resolved_plots[plot_type],
                'unit_x': resolve_unit_label(experiment, plot_config.get(X_UNIT_KEY)),
                'unit_y': resolve_unit_label(experiment, plot_config.get(Y_UNIT_KEY)),
            })

    return trace_specifications


# =============================================================================
# Axis assignment
# =============================================================================

def order_y_units(trace_specifications: list, selected_plot_types: list) -> list:
    """
    List the y units in the order in which their curves were selected.

    The order decides which unit keeps the left-hand axis, so it follows the
    user's selection rather than the order the experiments happen to be in.

    Parameters
    ----------
    trace_specifications : list of dict
        Entries as built by `build_trace_specifications`.
    selected_plot_types : list of str
        Curve names in selection order.

    Returns
    -------
    units : list of str or None
        One unit per selected curve that has traces, in selection order.
    """
    units_by_plot_type = {}

    for specification in trace_specifications:
        # The first experiment plotted may lack the unit while a later one has
        # it, so an entry already present is only overwritten when it is None.
        if units_by_plot_type.get(specification['plot_type']) is None:
            units_by_plot_type[specification['plot_type']] = specification['unit_y']

    return [units_by_plot_type[plot_type] for plot_type in selected_plot_types
            if plot_type in units_by_plot_type]


def assign_y_axes(y_units: list) -> dict:
    """
    Map each distinct y unit onto a plotly y-axis id.

    The first unit keeps the primary (left-hand) axis; every further unit gets
    its own axis, which `build_axis_layout` places on the right. Curves without
    a unit stay on the primary axis, so instructions that define no units keep
    the previous single-axis behavior.

    Parameters
    ----------
    y_units : list of str or None
        Units in the order that decides axis placement.

    Returns
    -------
    axis_ids : dict
        Mapping of unit (including None) to axis id ``'y'``, ``'y2'``, ….
    """
    axis_ids = {}

    for unit in y_units:
        if unit is None or unit in axis_ids:
            continue
        axis_ids[unit] = 'y' if not axis_ids else f'y{len(axis_ids) + 1}'

    axis_ids[None] = 'y'

    return axis_ids


def combine_unit_labels(units: list) -> str:
    """
    Join the distinct units sharing one axis into a single axis title.

    Parameters
    ----------
    units : list of str or None
        Units of the curves drawn against the axis.

    Returns
    -------
    title : str or None
        Single unit, the distinct units joined by `UNIT_SEPARATOR` where curves
        with different units share the axis, or None if none is defined.
    """
    distinct_units = list(dict.fromkeys(unit for unit in units if unit))

    if not distinct_units:
        return None

    return UNIT_SEPARATOR.join(distinct_units)


def build_axis_layout(axis_ids: dict, x_units: list) -> dict:
    """
    Build the plotly layout entries for the x axis and all y axes.

    Additional y axes are stacked at the right-hand edge; the x-axis domain is
    shortened to make room for the second and any further right-hand axis.

    Parameters
    ----------
    axis_ids : dict
        Mapping of unit to axis id, as returned by `assign_y_axes`.
    x_units : list of str or None
        x units of the drawn curves.

    Returns
    -------
    layout : dict
        Keyword arguments for `plotly.graph_objects.Figure.update_layout`.
    """
    axis_titles = {}
    for unit, axis_id in axis_ids.items():
        if unit is not None:
            axis_titles.setdefault(axis_id, unit)

    number_of_axes = len(set(axis_ids.values()))
    right_hand_axes = number_of_axes - 1
    domain_right = 1.0 - SECONDARY_AXIS_WIDTH * max(right_hand_axes - 1, 0)

    layout = {
        'xaxis': dict(title=combine_unit_labels(x_units) or DEFAULT_X_AXIS_TITLE,
                      domain=[0.0, domain_right],
                      gridcolor=GRID_COLOR),
        'yaxis': dict(title=axis_titles.get('y', DEFAULT_Y_AXIS_TITLE),
                      gridcolor=GRID_COLOR),
    }

    for axis_index in range(2, number_of_axes + 1):
        layout[f'yaxis{axis_index}'] = dict(
            title=axis_titles.get(f'y{axis_index}', DEFAULT_Y_AXIS_TITLE),
            overlaying='y',
            side='right',
            anchor='free',
            position=domain_right + SECONDARY_AXIS_WIDTH * (axis_index - 2),
            showgrid=False,
        )

    return layout


# =============================================================================
# Figure construction
# =============================================================================

def build_hover_template(experiment_name: str, plot_type: str,
                         unit_x: str, unit_y: str) -> str:
    """
    Build the hover box of one curve, annotated with its units.

    Parameters
    ----------
    experiment_name, plot_type : str
        Identify the curve in the hover box header.
    unit_x, unit_y : str or None
        Units appended to the hovered coordinates.

    Returns
    -------
    hover_template : str
        Plotly hover template.
    """
    x_suffix = f" {unit_x}" if unit_x else ""
    y_suffix = f" {unit_y}" if unit_y else ""

    return (f"<b>{experiment_name} - {plot_type}</b><br>"
            f"X: %{{x}}{x_suffix}<br>"
            f"Y: %{{y}}{y_suffix}<br>"
            "<extra></extra>")


def uses_marker_style(plot_type: str) -> bool:
    """
    Decide whether a curve is drawn as markers rather than as a line.

    Raw and (unsmoothed) rate data are scattered, so they read better as
    markers; smoothed and fitted curves are drawn as lines.

    Parameters
    ----------
    plot_type : str
        Curve name from the plotting instructions.

    Returns
    -------
    marker_style : bool
        True where the curve is drawn with markers.
    """
    lowered = plot_type.lower()

    return 'raw' in lowered or ('rate' in lowered and 'smoothed' not in lowered)


def add_curve(figure: go.Figure, specification: dict, axis_id: str) -> None:
    """
    Add one curve — and its highlighted point, if any — to the figure.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        Figure the traces are added to.
    specification : dict
        Trace entry as built by `build_trace_specifications`.
    axis_id : str
        Id of the y axis the curve belongs to.

    Returns
    -------
    None : None
        The figure is mutated in place.
    """
    plot_data = specification['data']
    color = specification['color']
    trace_name = f"{specification['experiment_name']} - {specification['plot_type']}"
    hover_template = build_hover_template(specification['experiment_name'],
                                          specification['plot_type'],
                                          specification['unit_x'],
                                          specification['unit_y'])

    marker_style = uses_marker_style(specification['plot_type'])

    figure.add_trace(go.Scatter(
        x=plot_data['x'],
        y=plot_data['y'],
        name=trace_name,
        yaxis=axis_id,
        mode='markers' if marker_style else 'lines',
        marker=dict(color=color, size=MARKER_SIZE, opacity=MARKER_OPACITY) if marker_style else None,
        line=None if marker_style else dict(color=color, width=LINE_WIDTH),
        hovertemplate=hover_template,
        hoverlabel=dict(font_color=color)
    ))

    # Added after the curve so the highlighted point is drawn on top of it.
    if 'x_point' in plot_data and 'y_point' in plot_data:
        figure.add_trace(go.Scatter(
            x=[plot_data['x_point']],
            y=[plot_data['y_point']],
            name=f"{trace_name} Point",
            yaxis=axis_id,
            mode='markers',
            marker=dict(color=POINT_MARKER_COLOR, size=POINT_MARKER_SIZE, symbol='circle'),
            hovertemplate=hover_template,
            hoverlabel=dict(font_color=POINT_MARKER_COLOR)
        ))


def build_figure(trace_specifications: list, selected_plot_types: list) -> go.Figure:
    """
    Build the complete time-series figure.

    Parameters
    ----------
    trace_specifications : list of dict
        Entries as built by `build_trace_specifications`.
    selected_plot_types : list of str
        Curve names in selection order, deciding the y-axis order.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Figure with one trace per curve and one y axis per distinct y unit.
    """
    axis_ids = assign_y_axes(order_y_units(trace_specifications, selected_plot_types))

    figure = go.Figure()

    for specification in trace_specifications:
        add_curve(figure, specification, axis_ids[specification['unit_y']])

    figure.update_layout(
        title="Time-Series Data Visualization",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_color='black'
        ),
        showlegend=True,
        legend=dict(
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        height=FIGURE_HEIGHT,
        **build_axis_layout(axis_ids,
                            [specification['unit_x'] for specification in trace_specifications])
    )

    return figure


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


def render_experiment_metadata(experimental_dataset) -> None:
    """
    Show the overview rows of the currently selected experiments.

    Parameters
    ----------
    experimental_dataset : ExperimentalDataset
        Dataset providing ``overview_df`` and the experiments.

    Returns
    -------
    None : None
        Widgets are written to the current Streamlit container.
    """
    st.markdown("#### Selected Experiments Metadata")

    if experimental_dataset.overview_df.empty:
        st.markdown("**Selected Experiments:**")
        for exp_name in st.session_state.selected_experiments:
            exp_data = experimental_dataset.experiments[exp_name]
            st.markdown(f"- **{exp_name}** (Group: {exp_data.group}, Color: {exp_data.color})")
        return

    mask = experimental_dataset.overview_df['Experiment'].isin(st.session_state.selected_experiments)
    filtered_df = experimental_dataset.overview_df[mask]

    if filtered_df.empty:
        st.info("No overview data available for selected experiments.")
    else:
        st.dataframe(filtered_df, width='stretch')


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
        ### Understanding the Time-Series Data Visualization Tool

        This tool helps visualize time-series data from experiments organized by groups.

        #### 1. Group Selection (Left Panel)
        - Experiments are organized by their group attribute
        - Click on a group to expand and see all experiments in that group
        - For non-reference groups, relevant metadata values are shown in parentheses
        - Select checkboxes to add experiments to the visualization

        #### 2. Visualization Panel (Right Panel)
        - **Selected Experiments**: Shows and allows manual selection/deselection of experiments
        - **Select Data to Display**: Choose which data types to plot (based on available plotting instructions)

        #### 3. Understanding the Plot
        - Each experiment is shown in its designated color
        - Different data types (raw, smoothed, fitted) are shown with different styles
        - Hover over data points to see exact values, including their units
        - Click legend items once to hide/show that trace
        - Double-click legend items to show only that trace

        #### 4. Axes and units
        Curves carry units when the upstream app declares them in the
        `'{INSTRUCTION_KEY}'` entry of `plotting_instruction`:

        ```python
        'Rate (O2)': {{
            'x': 'processed_data/time_reaction_s',
            'y': 'processed_data/rate_gaussian_umol_s',
            'unit_x': 'processed_data/time_unit',
            'unit_y': 'processed_data/rate_gaussian_unit',
        }}
        ```

        `'unit_x'` / `'unit_y'` are optional and are either a path to the unit
        stored with the experiment or a literal unit string. Curves whose y unit
        differs get their own y axis on the right, so amounts and rates can be
        shown together. Without units the axes keep their generic titles.

        #### 5. Experiment Metadata
        The table at the bottom shows detailed metadata for all selected experiments.

        #### Tips
        - Use the multiselect box for quick bulk selection/deselection
        - Compare different processing methods by selecting multiple data types
        - Use the same colors across different views for easy experiment identification
        """)


# =============================================================================
# Main Application
# =============================================================================

def render_time_series() -> None:
    """
    Render the Time-Series Data Visualization page.

    Returns
    -------
    None : None
        The page is written to the current Streamlit script run.
    """
    st.set_page_config(
        page_title="Time-Series Data Visualization",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    col_title, col_filename = st.columns([3, 1])
    with col_title:
        st.title("Time-Series Data Visualization")
    with col_filename:
        if st.session_state.hdf5_filename:
            st.markdown(f"<p style='text-align: right; font-size: 0.8em; color: gray; margin-top: 1.5em;'>{st.session_state.hdf5_filename}</p>", unsafe_allow_html=True)

    if st.session_state.experimental_dataset is None:
        st.info("Please upload a HDF5 file on the Home page first.")
        return

    experimental_dataset = st.session_state.experimental_dataset
    plotting_instruction = experimental_dataset.plotting_instruction[INSTRUCTION_KEY]

    # Selection state is shared with the results-table page
    if 'selected_experiments' not in st.session_state:
        st.session_state.selected_experiments = []
    if 'selected_plot_types' not in st.session_state:
        st.session_state.selected_plot_types = []

    col1, col2 = st.columns([1, 2])

    with col1:
        render_group_selection(
            group_experiments(experimental_dataset.experiments),
            experimental_dataset.group_mapping
        )

    with col2:
        st.header("Visualization Panel")

        ctrl_col1, ctrl_col2 = st.columns(2)

        with ctrl_col1:
            st.session_state[MULTISELECT_KEY] = list(st.session_state.selected_experiments)
            st.multiselect(
                "Selected Experiments",
                options=sorted(experimental_dataset.experiments.keys()),
                key=MULTISELECT_KEY,
                on_change=sync_from_multiselect,
                args=(MULTISELECT_KEY,)
            )

        with ctrl_col2:
            if plotting_instruction:
                st.session_state.selected_plot_types = st.multiselect(
                    "Select Data to Display",
                    list(plotting_instruction.keys()),
                    default=None
                )
            else:
                st.warning("No plotting instructions defined in dataset.")

        trace_specifications = build_trace_specifications(
            st.session_state.selected_experiments,
            experimental_dataset.experiments,
            plotting_instruction,
            st.session_state.selected_plot_types
        )

        st.plotly_chart(build_figure(trace_specifications,
                                     st.session_state.selected_plot_types),
                        width='stretch')

        if st.session_state.selected_experiments:
            render_experiment_metadata(experimental_dataset)

    render_help_section()

    st.markdown("---")
    st.caption("pyKES Time-Series Visualization | Powered by Streamlit")
