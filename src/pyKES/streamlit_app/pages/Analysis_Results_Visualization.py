"""
Analysis Results Visualization - Refactored Version

This module provides a Streamlit interface for visualizing kinetic analysis results
from photochemical experiments. It organizes experiments by groups and metadata,
allowing users to compare analysis results across different experimental conditions.

Author: pyKES Development Team
Date: 24 November 2025
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Tuple, Optional, Any

from pyKES.utilities.resolve_attributes import resolve_experiment_attributes


# =============================================================================
# Data Processing Functions
# =============================================================================

def filter_active_experiments(
    experiments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Filter experiments to include only those marked as active.
    
    Parameters
    ----------
    experiments : Dict[str, Any]
        Dictionary mapping experiment names to experiment data objects.
        Each experiment object should have a metadata/Active attribute.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing only active experiments.
        
    Examples
    --------
    >>> all_experiments = {
    ...     'Exp_001': exp_obj_1,  # metadata/Active = True
    ...     'Exp_002': exp_obj_2,  # metadata/Active = False
    ...     'Exp_003': exp_obj_3   # metadata/Active = True
    ... }
    >>> active = filter_active_experiments(all_experiments)
    >>> len(active)
    2
    >>> 'Exp_002' in active
    False
    """
    active_experiments = {}
    
    for exp_name, exp_data in experiments.items():
        try:
            is_active = resolve_experiment_attributes(
                {'Active': 'metadata/Active'}, 
                exp_data, 
                mode='permissive'
            )
            if is_active.get('Active', False):
                active_experiments[exp_name] = exp_data
        except (ValueError, KeyError, AttributeError):
            # If Active field not found or can't be resolved, skip experiment
            continue
    
    return active_experiments


def create_experiment_subsets(
    active_experiments: Dict[str, Any],
    group_mapping: Dict[str, Optional[str]]
) -> Tuple[Dict[str, List[Tuple[str, List[str]]]], Dict[str, List[str]]]:
    """
    Organize experiments into subsets based on group membership and metadata values.
    
    Parameters
    ----------
    active_experiments : Dict[str, Any]
        Dictionary of active experiments (experiment_name -> experiment_data).
    group_mapping : Dict[str, Optional[str]]
        Mapping of group names to metadata paths. None indicates a reference group.
        Example: {'Reference': None, 'Intensity': 'metadata/Intensity / Irradiance [mW/cm2]'}
    
    Returns
    -------
    subsets_by_group : Dict[str, List[Tuple[str, List[str]]]]
        Dictionary mapping group names to lists of (subset_label, experiment_names).
        Example: {'Intensity': [('10.0 (n=3)', ['Exp_001', 'Exp_002', 'Exp_003'])]}
    all_subsets : Dict[str, List[str]]
        Dictionary mapping subset keys to experiment names.
        Subset keys format: 'GroupName||MetadataValue'
        Example: {'Intensity||10.0': ['Exp_001', 'Exp_002', 'Exp_003']}
        
    Examples
    --------
    >>> active_exps = {
    ...     'Exp_001': exp_with_intensity_10,
    ...     'Exp_002': exp_with_intensity_10,
    ...     'Exp_003': exp_with_intensity_20
    ... }
    >>> group_map = {'Intensity': 'metadata/Intensity / Irradiance [mW/cm2]'}
    >>> subsets_by_group, all_subsets = create_experiment_subsets(active_exps, group_map)
    >>> len(all_subsets)
    2
    >>> 'Intensity||10.0' in all_subsets
    True
    """
    subsets_by_group = defaultdict(list)
    all_subsets = {}
    
    for group_name in group_mapping.keys():
        experiments_in_group = []
        
        # Find all active experiments in this group
        for exp_name, exp_data in active_experiments.items():
            if exp_data.group == group_name:
                experiments_in_group.append((exp_name, exp_data))
        
        if not experiments_in_group:
            continue
        
        # For Reference group, create one subset with all experiments
        if group_name == 'Reference' or group_mapping[group_name] is None:
            subset_label = f"Reference (n={len(experiments_in_group)})"
            subset_key = f"{group_name}||Reference"
            exp_names = [exp_name for exp_name, _ in experiments_in_group]
            subsets_by_group[group_name].append((subset_label, exp_names))
            all_subsets[subset_key] = exp_names
        else:
            # Group by metadata value
            metadata_path = group_mapping[group_name]
            subsets_by_metadata = defaultdict(list)
            
            for exp_name, exp_data in experiments_in_group:
                try:
                    metadata_value = resolve_experiment_attributes(
                        {'value': metadata_path}, 
                        exp_data, 
                        mode='permissive'
                    )
                    if 'value' in metadata_value:
                        value = metadata_value['value']
                        subsets_by_metadata[value].append(exp_name)
                except (ValueError, KeyError, AttributeError):
                    continue
            
            # Create subset labels
            for metadata_value, exp_names in sorted(subsets_by_metadata.items()):
                subset_label = f"{metadata_value} (n={len(exp_names)})"
                subset_key = f"{group_name}||{metadata_value}"
                subsets_by_group[group_name].append((subset_label, exp_names))
                all_subsets[subset_key] = exp_names

    
    return dict(subsets_by_group), all_subsets


def extract_x_value_for_experiment(
    exp_data: Any,
    x_axis_mode: str,
    exp_name: str,
    x_axis_group_mapping: Dict[str, str],
    group_mapping: Dict[str, Optional[str]]
) -> Optional[Any]:
    """
    Extract the x-axis value for an experiment based on the selected x-axis mode.
    
    Parameters
    ----------
    exp_data : Any
        The experiment data object.
    x_axis_mode : str
        The x-axis display mode ('Experiments' or a metadata column name).
    exp_name : str
        The experiment name (used when x_axis_mode is 'Experiments').
    x_axis_group_mapping : Dict[str, str]
        Mapping from display names to group names.
    group_mapping : Dict[str, Optional[str]]
        Mapping from group names to metadata paths.
    
    Returns
    -------
    Optional[Any]
        The x-axis value, or None if it cannot be resolved.
        
    Examples
    --------
    >>> exp_data = experiment_with_intensity_10
    >>> x_value = extract_x_value_for_experiment(
    ...     exp_data, 
    ...     'Intensity / Irradiance [mW/cm2]',
    ...     'Exp_001',
    ...     {'Intensity / Irradiance [mW/cm2]': 'Intensity'},
    ...     {'Intensity': 'metadata/Intensity / Irradiance [mW/cm2]'}
    ... )
    >>> x_value
    10.0
    """
    # Determine x-value
    if x_axis_mode == 'Experiments':
        return exp_name
    else:
        # Extract group name from x-axis selection using mapping
        x_group_name = x_axis_group_mapping.get(x_axis_mode)
        if x_group_name and x_group_name in group_mapping and group_mapping[x_group_name] is not None:
            metadata_path = group_mapping[x_group_name]
            try:
                metadata_value = resolve_experiment_attributes(
                    {'value': metadata_path}, 
                    exp_data, 
                    mode='permissive'
                )
                return metadata_value.get('value', None)
            except (ValueError, KeyError, AttributeError):
                return None
        else:
            return None


def collect_plot_data(
    selected_subsets: Dict[str, List[str]],
    active_experiments: Dict[str, Any],
    selected_analysis_results: List[str],
    kinetic_results_instructions: Dict[str, Dict[str, str]],
    x_axis_mode: str,
    x_axis_group_mapping: Dict[str, str],
    group_mapping: Dict[str, Optional[str]]
) -> List[Tuple[str, str, str, Any, Any, str, Any]]:
    """
    Collect data points for plotting from selected experiments and analysis results.
    
    Parameters
    ----------
    selected_subsets : Dict[str, List[str]]
        Dictionary mapping subset keys to lists of experiment names.
    active_experiments : Dict[str, Any]
        Dictionary of active experiments.
    selected_analysis_results : List[str]
        List of analysis result names to extract (e.g., ['k1', 'k2']).
    kinetic_results_instructions : Dict[str, Dict[str, str]]
        Instructions for accessing each analysis result.
        Example: {'k1': {'Value': 'analysis_results/rates/k1', 'Unit': 's⁻¹'}}
    x_axis_mode : str
        Selected x-axis mode ('Experiments' or metadata column name).
    x_axis_group_mapping : Dict[str, str]
        Mapping from display names to group names.
    group_mapping : Dict[str, Optional[str]]
        Mapping from group names to metadata paths.
    
    Returns
    -------
    List[Tuple[str, str, str, Any, Any, str, Any]]
        List of tuples containing:
        (subset_key, subset_value, exp_name, exp_data, x_value, result_name, y_value)
        
    Examples
    --------
    >>> selected_subsets = {'Intensity||10.0': ['Exp_001', 'Exp_002']}
    >>> selected_results = ['k1', 'k2']
    >>> plot_data = collect_plot_data(
    ...     selected_subsets, active_exps, selected_results,
    ...     kinetic_instructions, 'Experiments', {}, {}
    ... )
    >>> len(plot_data)
    4  # 2 experiments × 2 results
    """
    plot_data = []
    
    for subset_key, exp_names in selected_subsets.items():
        group_name, subset_value = subset_key.split('||', 1)
        
        for exp_name in exp_names:
            if exp_name not in active_experiments:
                continue
            
            exp_data = active_experiments[exp_name]
            
            # Determine x-value
            x_value = extract_x_value_for_experiment(
                exp_data, x_axis_mode, exp_name, 
                x_axis_group_mapping, group_mapping
            )
            
            if x_value is None:
                continue
            
            # Resolve analysis results
            for result_name in selected_analysis_results:
                if result_name not in kinetic_results_instructions:
                    continue
                
                result_config = kinetic_results_instructions[result_name]
                value_path = result_config.get('Value', None)
                
                if value_path is None:
                    continue
                
                try:
                    result_value = resolve_experiment_attributes(
                        {'value': value_path}, 
                        exp_data, 
                        mode='permissive'
                    )
                    y_value = result_value.get('value', None)
                    
                    if y_value is not None:
                        plot_data.append((
                            subset_key, subset_value, exp_name, 
                            exp_data, x_value, result_name, y_value
                        ))
                except (ValueError, KeyError, AttributeError):
                    continue
    
    return plot_data


def sort_plot_data(
    plot_data: List[Tuple[str, str, str, Any, Any, str, Any]],
    x_axis_mode: str
) -> List[Tuple[str, str, str, Any, Any, str, Any]]:
    """
    Sort plot data appropriately based on the x-axis mode.
    
    Parameters
    ----------
    plot_data : List[Tuple[str, str, str, Any, Any, str, Any]]
        Unsorted plot data.
    x_axis_mode : str
        The x-axis display mode ('Experiments' or metadata column name).
    
    Returns
    -------
    List[Tuple[str, str, str, Any, Any, str, Any]]
        Sorted plot data.
        
    Examples
    --------
    >>> plot_data = [
    ...     ('subset1', 'val1', 'Exp_003', ..., 'Exp_003', 'k1', 0.5),
    ...     ('subset1', 'val1', 'Exp_001', ..., 'Exp_001', 'k1', 0.3),
    ... ]
    >>> sorted_data = sort_plot_data(plot_data, 'Experiments')
    >>> sorted_data[0][2]  # First experiment name
    'Exp_001'
    """
    if x_axis_mode == 'Experiments':
        # For experiment mode, sort by experiment name
        return sorted(plot_data, key=lambda x: x[2])
    else:
        # For metadata mode, sort by subset_key, result_name, x_value
        return sorted(plot_data, key=lambda x: (x[0], x[5], x[4]))


# =============================================================================
# UI State Management Functions
# =============================================================================

def initialize_session_state() -> None:
    """
    Initialize all required session state variables.
    
    This function ensures all necessary session state keys exist with
    appropriate default values. Should be called at the start of the app.
    
    Examples
    --------
    >>> # In Streamlit app
    >>> initialize_session_state()
    >>> 'selected_subsets' in st.session_state
    True
    """
    if 'selected_subsets' not in st.session_state:
        st.session_state.selected_subsets = {}
    if 'all_subsets' not in st.session_state:
        st.session_state.all_subsets = {}
    if 'selected_analysis_results' not in st.session_state:
        st.session_state.selected_analysis_results = []
    if 'selected_x_axis' not in st.session_state:
        st.session_state.selected_x_axis = 'Experiments'


def update_subset_selection(
    subset_key: str, 
    experiments_in_subset: List[str], 
    value: bool
) -> None:
    """
    Handle checkbox state changes for experiment subsets.
    
    Parameters
    ----------
    subset_key : str
        The unique key for the subset (format: 'GroupName||MetadataValue').
    experiments_in_subset : List[str]
        List of experiment names in this subset.
    value : bool
        New checkbox state (True = selected, False = deselected).
        
    Examples
    --------
    >>> # When user checks a subset checkbox
    >>> update_subset_selection(
    ...     'Intensity||10.0',
    ...     ['Exp_001', 'Exp_002', 'Exp_003'],
    ...     True
    ... )
    >>> 'Intensity||10.0' in st.session_state.selected_subsets
    True
    """
    if value:
        # Add all experiments in subset
        st.session_state.selected_subsets[subset_key] = experiments_in_subset
    else:
        # Remove subset
        if subset_key in st.session_state.selected_subsets:
            del st.session_state.selected_subsets[subset_key]


def sync_from_multiselect() -> None:
    """
    Synchronize selected_subsets state from the multiselect widget.
    
    This callback ensures consistency between the multiselect display
    and the internal selected_subsets dictionary.
    
    Examples
    --------
    >>> # Called automatically when multiselect widget changes
    >>> st.session_state.multiselect_widget = ['Intensity||10.0', 'Intensity||20.0']
    >>> sync_from_multiselect()
    >>> len(st.session_state.selected_subsets)
    2
    """
    selected_subset_keys = st.session_state.multiselect_widget
    # Rebuild selected_subsets dict based on selection
    new_selected = {}
    for subset_key in selected_subset_keys:
        if subset_key in st.session_state.all_subsets:
            new_selected[subset_key] = st.session_state.all_subsets[subset_key]
    st.session_state.selected_subsets = new_selected


# =============================================================================
# UI Rendering Functions
# =============================================================================

def render_group_selection_sidebar(
    subsets_by_group: Dict[str, List[Tuple[str, List[str]]]],
    group_mapping: Dict[str, Optional[str]]
) -> None:
    """
    Render the left sidebar for selecting experiment groups and subsets.
    
    Parameters
    ----------
    subsets_by_group : Dict[str, List[Tuple[str, List[str]]]]
        Dictionary mapping group names to lists of (subset_label, experiment_names).
    group_mapping : Dict[str, Optional[str]]
        Mapping of group names to metadata paths.
        
    Examples
    --------
    >>> subsets_by_group = {
    ...     'Intensity': [
    ...         ('10.0 (n=3)', ['Exp_001', 'Exp_002', 'Exp_003']),
    ...         ('20.0 (n=2)', ['Exp_004', 'Exp_005'])
    ...     ]
    ... }
    >>> render_group_selection_sidebar(subsets_by_group, group_mapping)
    """
    st.header("Group Selection")
    
    for group_name in sorted(subsets_by_group.keys()):
        subsets = subsets_by_group[group_name]
        
        # Create group title
        if group_name == 'Reference' or group_mapping[group_name] is None:
            group_title = f"Reference (n={sum(len(exps) for _, exps in subsets)})"
        else:
            # Extract column name from metadata path
            metadata_path = group_mapping[group_name]
            column_name = metadata_path.split('/', 1)[-1] if '/' in metadata_path else metadata_path
            total_count = sum(len(exps) for _, exps in subsets)
            group_title = f"{column_name} (n={total_count})"
        
        with st.expander(group_title, expanded=False):
            # Display each subset in the group
            for subset_label, exp_names in sorted(subsets, key=lambda x: x[0]):
                subset_key = _find_subset_key(exp_names)
                
                if subset_key is None:
                    continue
                
                is_selected = subset_key in st.session_state.selected_subsets
                
                # Create checkbox for subset
                st.checkbox(
                    subset_label,
                    value=is_selected,
                    key=f"checkbox_{subset_key}",
                    on_change=update_subset_selection,
                    args=(subset_key, exp_names, not is_selected)
                )


def _find_subset_key(exp_names: List[str]) -> Optional[str]:
    """
    Find the subset key for a given list of experiment names.
    
    Parameters
    ----------
    exp_names : List[str]
        List of experiment names to find.
    
    Returns
    -------
    Optional[str]
        The subset key if found, None otherwise.
    """
    for key, exps in st.session_state.all_subsets.items():
        if exps == exp_names:
            return key
    return None


def create_subset_display_labels(
    all_subset_keys: List[str]
) -> Dict[str, str]:
    """
    Create human-readable display labels for subset keys.
    
    Parameters
    ----------
    all_subset_keys : List[str]
        List of subset keys (format: 'GroupName||MetadataValue').
    
    Returns
    -------
    Dict[str, str]
        Mapping from subset keys to display labels.
        
    Examples
    --------
    >>> keys = ['Intensity||10.0', 'Reference||Reference']
    >>> labels = create_subset_display_labels(keys)
    >>> labels['Intensity||10.0']
    'Intensity: 10.0 (n=3)'
    >>> labels['Reference||Reference']
    'Reference (n=5)'
    """
    subset_display_labels = {}
    
    for key in all_subset_keys:
        group_name, value = key.split('||', 1)
        exp_names = st.session_state.all_subsets[key]
        
        if value == 'Reference':
            display_label = f"Reference (n={len(exp_names)})"
        else:
            display_label = f"{group_name}: {value} (n={len(exp_names)})"
        
        subset_display_labels[key] = display_label
    
    return subset_display_labels


def create_x_axis_options(
    group_mapping: Dict[str, Optional[str]]
) -> Tuple[List[str], Dict[str, str]]:
    """
    Create x-axis selection options from group mapping.
    
    Parameters
    ----------
    group_mapping : Dict[str, Optional[str]]
        Mapping of group names to metadata paths.
    
    Returns
    -------
    x_axis_options : List[str]
        List of display names for x-axis selection.
    x_axis_group_mapping : Dict[str, str]
        Mapping from display names to group names.
        
    Examples
    --------
    >>> group_map = {
    ...     'Reference': None,
    ...     'Intensity': 'metadata/Intensity / Irradiance [mW/cm2]'
    ... }
    >>> options, mapping = create_x_axis_options(group_map)
    >>> 'Experiments' in options
    True
    >>> 'Intensity / Irradiance [mW/cm2]' in options
    True
    """
    x_axis_options = ['Experiments']
    x_axis_group_mapping = {}
    
    for group_name, metadata_path in group_mapping.items():
        if metadata_path is not None:
            column_name = metadata_path.split('/', 1)[-1] if '/' in metadata_path else metadata_path
            x_axis_options.append(column_name)
            x_axis_group_mapping[column_name] = group_name
    
    return x_axis_options, x_axis_group_mapping


def render_control_panel(
    group_mapping: Dict[str, Optional[str]],
    kinetic_results_instructions: Dict[str, Dict[str, str]]
) -> Tuple[str, Dict[str, str]]:
    """
    Render the control panel with multiselect, x-axis, and analysis result selectors.
    
    Parameters
    ----------
    group_mapping : Dict[str, Optional[str]]
        Mapping of group names to metadata paths.
    kinetic_results_instructions : Dict[str, Dict[str, str]]
        Instructions for accessing each analysis result.
    
    Returns
    -------
    x_axis_mode : str
        Selected x-axis mode.
    x_axis_group_mapping : Dict[str, str]
        Mapping from display names to group names.
        
    Examples
    --------
    >>> group_map = {'Intensity': 'metadata/Intensity / Irradiance [mW/cm2]'}
    >>> kinetic_instr = {'k1': {'Value': 'rates/k1', 'Unit': 's⁻¹'}}
    >>> x_mode, x_mapping = render_control_panel(group_map, kinetic_instr)
    """
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    
    with ctrl_col1:
        # Multiselect for selected subsets overview
        all_subset_keys = list(st.session_state.all_subsets.keys())
        subset_display_labels = create_subset_display_labels(all_subset_keys)
        
        st.multiselect(
            "Selected Experiments",
            options=all_subset_keys,
            default=list(st.session_state.selected_subsets.keys()),
            format_func=lambda x: subset_display_labels.get(x, x),
            key="multiselect_widget",
            on_change=sync_from_multiselect
        )
    
    with ctrl_col2:
        # X-axis selection
        x_axis_options, x_axis_group_mapping = create_x_axis_options(group_mapping)
        
        x_axis_mode = st.selectbox(
            'Select X-Axis',
            x_axis_options,
            index=0
        )
        st.session_state.selected_x_axis = x_axis_mode
    
    with ctrl_col3:
        # Analysis results selection
        available_results = list(kinetic_results_instructions.keys())
        
        # Initialize with first result if empty
        if not st.session_state.selected_analysis_results and available_results:
            st.session_state.selected_analysis_results = [available_results[0]]
        
        selected_results = st.multiselect(
            "Select Analysis Results",
            options=available_results,
            default=None,
            key="analysis_results_multiselect"
        )
        
        # Update session state with the current selection
        st.session_state.selected_analysis_results = selected_results
    
    return x_axis_mode, x_axis_group_mapping


def determine_axis_labels(
    selected_analysis_results: List[str],
    kinetic_results_instructions: Dict[str, Dict[str, str]],
    x_axis_mode: str
) -> Tuple[str, str]:
    """
    Determine appropriate labels for x and y axes.
    
    Parameters
    ----------
    selected_analysis_results : List[str]
        List of selected analysis result names.
    kinetic_results_instructions : Dict[str, Dict[str, str]]
        Instructions for accessing each analysis result.
    x_axis_mode : str
        Selected x-axis mode.
    
    Returns
    -------
    x_axis_label : str
        Label for x-axis.
    y_axis_label : str
        Label for y-axis (includes unit from last selected result).
        
    Examples
    --------
    >>> results = ['k1', 'k2']
    >>> instructions = {'k2': {'Unit': 's⁻¹', 'Value': 'rates/k2'}}
    >>> x_label, y_label = determine_axis_labels(results, instructions, 'Experiments')
    >>> y_label
    's⁻¹'
    """
    # Determine y-axis label (use last selected result's unit)
    y_axis_label = "Value"
    if selected_analysis_results:
        last_result = selected_analysis_results[-1]
        if last_result in kinetic_results_instructions:
            y_axis_label = kinetic_results_instructions[last_result].get('Unit', 'Value')
    
    # Determine x-axis label
    x_axis_label = x_axis_mode
    
    return x_axis_label, y_axis_label


def create_plotly_figure(
    plot_data: List[Tuple[str, str, str, Any, Any, str, Any]],
    x_axis_label: str,
    y_axis_label: str,
    x_axis_mode: str
) -> go.Figure:
    """
    Create a Plotly figure from collected plot data.
    
    Parameters
    ----------
    plot_data : List[Tuple[str, str, str, Any, Any, str, Any]]
        Sorted plot data containing all points to visualize.
    x_axis_label : str
        Label for x-axis.
    y_axis_label : str
        Label for y-axis.
    x_axis_mode : str
        Selected x-axis mode ('Experiments' or metadata column name).
    
    Returns
    -------
    go.Figure
        Configured Plotly figure ready for display.
        
    Examples
    --------
    >>> plot_data = [
    ...     ('subset1', 'val1', 'Exp_001', exp_obj, 'Exp_001', 'k1', 0.5),
    ...     ('subset1', 'val1', 'Exp_002', exp_obj, 'Exp_002', 'k1', 0.6)
    ... ]
    >>> fig = create_plotly_figure(plot_data, 'Experiments', 's⁻¹', 'Experiments')
    >>> fig.layout.xaxis.title.text
    'Experiments'
    """
    fig = go.Figure()
    
    # Group by subset and result for plotting
    for (subset_key, result_name), group in groupby(plot_data, key=lambda x: (x[0], x[5])):
        group_data = list(group)
        
        if not group_data:
            continue
        
        # Get subset info
        _, subset_value, _, first_exp_data, _, _, _ = group_data[0]
        exp_color = first_exp_data.color
        
        # Extract data for plotting
        x_values = []
        y_values = []
        exp_names = []
        
        for _, _, exp_name, exp_data, x_value, _, y_value in group_data:
            x_values.append(x_value)
            y_values.append(y_value)
            exp_names.append(exp_name)
        
        # Create hover template
        hover_template = (
            "<b>%{text}</b><br>"
            f"{x_axis_label}: %{{x}}<br>"
            f"{result_name}: %{{y:.2e}}<br>"
            "<extra></extra>"
        )
        
        # Create trace name
        group_name = subset_key.split('||')[0]
        if subset_value == 'Reference':
            trace_name = f"Reference - {result_name}"
        else:
            trace_name = f"{group_name}: {subset_value} - {result_name}"
        
        # Add trace (dots + lines)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            name=trace_name,
            mode='lines+markers',
            marker=dict(color=exp_color, size=8),
            line=dict(color=exp_color, width=2),
            text=exp_names,
            hovertemplate=hover_template,
            hoverlabel=dict(font_color=exp_color)
        ))
    
    # Update layout
    fig.update_layout(
        title="Analysis Results",
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
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
        height=600
    )
    
    # For "Experiments" mode, set categorical x-axis
    if x_axis_mode == 'Experiments':
        # Get unique experiment names in order
        unique_exp_names = []
        seen = set()
        for item in plot_data:
            exp_name = item[2]
            if exp_name not in seen:
                unique_exp_names.append(exp_name)
                seen.add(exp_name)
        
        fig.update_xaxes(
            type='category',
            categoryorder='array',
            categoryarray=unique_exp_names
        )
    
    # Update axis colors and format y-axis in scientific notation
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.2)',
        tickformat='.2e'
    )
    
    return fig


def render_metadata_table(
    experimental_dataset: Any,
    active_experiments: Dict[str, Any]
) -> None:
    """
    Render a table showing metadata for selected experiments.
    
    Parameters
    ----------
    experimental_dataset : Any
        The experimental dataset object containing overview_df.
    active_experiments : Dict[str, Any]
        Dictionary of active experiments.
        
    Examples
    --------
    >>> render_metadata_table(dataset, active_exps)
    # Displays a Streamlit dataframe with experiment metadata
    """
    if not st.session_state.selected_subsets:
        return
    
    st.markdown("#### Selected Experiments Metadata")
    
    # Collect all selected experiment names
    all_selected_exp_names = []
    for exp_names in st.session_state.selected_subsets.values():
        all_selected_exp_names.extend(exp_names)
    
    if not experimental_dataset.overview_df.empty:
        mask = experimental_dataset.overview_df['Experiment'].isin(all_selected_exp_names)
        filtered_df = experimental_dataset.overview_df[mask]
        
        if not filtered_df.empty:
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.info("No overview data available for selected experiments.")
    else:
        # Display basic info if no overview_df
        st.markdown("**Selected Experiments:**")
        for exp_name in all_selected_exp_names:
            if exp_name in active_experiments:
                exp_data = active_experiments[exp_name]
                st.markdown(f"- **{exp_name}** (Group: {exp_data.group}, Color: {exp_data.color})")


def create_json_export_data(
    plot_data: List[Tuple[str, str, str, Any, Any, str, Any]],
    x_axis_mode: str,
    selected_analysis_results: List[str],
    kinetic_results_instructions: Dict[str, Dict[str, str]],
    experimental_dataset: Any,
    active_experiments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a JSON-serializable dictionary containing plot data and metadata.
    
    Parameters
    ----------
    plot_data : List[Tuple[str, str, str, Any, Any, str, Any]]
        The plot data containing (subset_key, subset_value, exp_name, exp_data, x_value, result_name, y_value).
    x_axis_mode : str
        The selected x-axis mode ('Experiments' or metadata column name).
    selected_analysis_results : List[str]
        List of selected analysis result names.
    kinetic_results_instructions : Dict[str, Dict[str, str]]
        Instructions for accessing each analysis result (includes units).
    experimental_dataset : Any
        The experimental dataset object.
    active_experiments : Dict[str, Any]
        Dictionary of active experiments.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with structure:
        {
            'plotting_data': {
                'analysis_result_name': {
                    'unit': str,
                    'data': {
                        'x_value': {
                            'min': float,
                            'max': float,
                            'mean': float,
                            'median': float,
                            'std': float,
                            'count': int,
                            'experiments_at_this_x_value': [
                                {
                                    'experiment_name': str,
                                    'y_value': float,
                                    'subset_key': str,
                                    'subset_value': str,
                                    'group_name': str,
                                    'color': str
                                }
                            ]
                        }
                    }
                }
            },
            'metadata': {...}
        }
    
    Examples
    --------
    >>> json_data = create_json_export_data(
    ...     plot_data, 'Experiments', ['k1', 'k2'],
    ...     kinetic_instructions, dataset, active_exps
    ... )
    >>> 'plotting_data' in json_data
    True
    >>> 'metadata' in json_data
    True
    >>> json_data['plotting_data']['k1']['unit']
    's⁻¹'
    """
    # Organize data by analysis result and x-value
    data_by_result_and_x = defaultdict(lambda: defaultdict(list))
    
    for subset_key, subset_value, exp_name, exp_data, x_value, result_name, y_value in plot_data:
        group_name = subset_key.split('||')[0]
        
        # Convert x_value to string for categorical mode, keep as-is for numeric
        if x_axis_mode == 'Experiments':
            x_key = str(x_value)
        else:
            # Try to keep numeric types for proper JSON serialization
            try:
                x_key = float(x_value) if not isinstance(x_value, str) else x_value
            except (ValueError, TypeError):
                x_key = str(x_value)
        
        data_by_result_and_x[result_name][x_key].append({
            'experiment_name': exp_name,
            'y_value': float(y_value) if not isinstance(y_value, str) else y_value,
            'subset_key': subset_key,
            'subset_value': subset_value,
            'group_name': group_name,
            'color': exp_data.color
        })
    
    # Calculate statistics for each result and x-value
    plotting_data = {}
    for result_name, x_values_dict in data_by_result_and_x.items():
        # Sort x_values if they are numeric
        try:
            # Try to sort numerically
            sorted_x_values = sorted(x_values_dict.items(), key=lambda item: float(item[0]))
        except (ValueError, TypeError):
            # If conversion fails, sort as strings (categorical)
            sorted_x_values = sorted(x_values_dict.items(), key=lambda item: str(item[0]))
        
        # Get unit for this analysis result
        result_unit = kinetic_results_instructions.get(result_name, {}).get('Unit', 'Unknown')
        
        plotting_data[result_name] = {
            'unit': result_unit,
            'data': {}
        }
        
        for x_value, experiments_list in sorted_x_values:
            # Extract y-values for statistics
            y_values = [exp['y_value'] for exp in experiments_list if isinstance(exp['y_value'], (int, float))]
            
            if y_values:
                plotting_data[result_name]['data'][x_value] = {
                    'min': float(np.min(y_values)),
                    'max': float(np.max(y_values)),
                    'mean': float(np.mean(y_values)),
                    'median': float(np.median(y_values)),
                    'std': float(np.std(y_values)),
                    'count': len(y_values),
                    'experiments_at_this_x_value': experiments_list
                }
            else:
                # Handle case where no numeric y-values exist
                plotting_data[result_name]['data'][x_value] = {
                    'min': None,
                    'max': None,
                    'mean': None,
                    'median': None,
                    'std': None,
                    'count': len(experiments_list),
                    'experiments_at_this_x_value': experiments_list
                }
    
    # Collect all selected experiment names
    all_selected_exp_names = []
    for exp_names in st.session_state.selected_subsets.values():
        all_selected_exp_names.extend(exp_names)
    
    # Determine x-axis label and type
    x_axis_label = x_axis_mode
    x_axis_type = 'categorical' if x_axis_mode == 'Experiments' or 'D2O' else 'numerical'
    
    # Build metadata section
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'h5_filename': st.session_state.hdf5_filename,
        'x_axis': {
            'mode': x_axis_mode,
            'label': x_axis_label,
            'type': x_axis_type
        },
        'selected_analysis_results': selected_analysis_results,
        'selected_subsets': {
            key: {
                'experiments': exp_names,
                'count': len(exp_names)
            }
            for key, exp_names in st.session_state.selected_subsets.items()
        },
        'total_experiments_displayed': len(set(all_selected_exp_names)),
        'total_data_points': len(plot_data)
    }
    
    return {
        'plotting_data': plotting_data,
        'metadata': metadata
    }


def render_help_section() -> None:
    """
    Render the help/explanation section at the bottom of the page.
    
    Examples
    --------
    >>> render_help_section()
    # Displays an expandable section with usage instructions
    """
    st.markdown("---")
    st.header("How to Use This Tool")
    
    with st.expander("📚 Detailed Explanation", expanded=False):
        st.markdown("""
        ### Understanding the Analysis Results Visualization Tool
        
        This tool helps visualize kinetic analysis results from experiments organized by groups.
        
        #### 1. Group Selection (Left Panel)
        - Experiments are organized by their group attribute
        - Within each group, experiments with the same metadata values are grouped into subsets
        - Only experiments marked as Active (metadata/Active = True) are included
        - Select checkboxes to add entire subsets to the visualization
        - The number in parentheses (n=X) shows how many experiments are in each subset
        
        #### 2. Visualization Panel (Right Panel)
        - **Selected Experiments**: Shows selected subsets (not individual experiments)
        - **Select X-Axis**: Choose what to plot on the x-axis
          - "Experiments": Show individual experiments side-by-side
          - Group options: Plot against metadata values (e.g., "Intensity / Irradiance [mW/cm2]")
        - **Select Analysis Results**: Choose which kinetic parameters to visualize
        
        #### 3. Understanding the Plot
        - Each subset is shown in its designated color
        - Individual experiments appear as dots
        - Experiments within the same subset are connected by lines
        - Hover over dots to see experiment name and values
        - Multiple analysis results can be compared simultaneously
        - Y-axis unit is determined by the last selected analysis result
        
        #### 4. Experiment Metadata
        The table at the bottom shows detailed metadata for all experiments in selected subsets.
        
        #### Tips
        - Use subset selection for quick comparison of experimental conditions
        - Switch between "Experiments" and metadata x-axes to view different perspectives
        - Compare multiple kinetic parameters by selecting several analysis results
        - Colors are consistent across different groups for easy identification
        """)


# =============================================================================
# Main Application Logic
# =============================================================================

def main() -> None:
    """
    Main application entry point for the Analysis Results Visualization page.
    
    This function orchestrates the entire Streamlit application flow:
    1. Configure page settings
    2. Check for loaded dataset
    3. Initialize session state
    4. Process experiment data
    5. Render UI components
    6. Generate visualizations
    """
    # Configure page
    st.set_page_config(
        page_title="Analysis Results Visualization", 
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and filename display
    col_title, col_filename = st.columns([3, 1])
    with col_title:
        st.title("Analysis Results Visualization")
    with col_filename:
        if st.session_state.hdf5_filename:
            st.markdown(f"<p style='text-align: right; font-size: 0.8em; color: gray; margin-top: 1.5em;'>{st.session_state.hdf5_filename}</p>", unsafe_allow_html=True)
    
    # Check if dataset is loaded
    if st.session_state.experimental_dataset is None:
        st.info("Please upload a HDF5 file on the Home page first.")
        return
    
    # Get dataset and configuration
    experimental_dataset = st.session_state.experimental_dataset
    group_mapping = experimental_dataset.group_mapping
    plotting_instruction = experimental_dataset.plotting_instruction
    
    # Validate kinetic results instructions
    if 'kinetic_results_instructions' not in plotting_instruction:
        st.error("No 'kinetic_results_instructions' found in plotting_instruction")
        return
    
    kinetic_results_instructions = plotting_instruction['kinetic_results_instructions']
    
    # Initialize session state
    initialize_session_state()
    
    # Process experiment data
    active_experiments = filter_active_experiments(experimental_dataset.experiments)
    subsets_by_group, all_subsets = create_experiment_subsets(active_experiments, group_mapping)
    st.session_state.all_subsets = all_subsets
    
    # Create two-column layout
    col1, col2 = st.columns([1, 2])
    
    # Render left sidebar
    with col1:
        render_group_selection_sidebar(subsets_by_group, group_mapping)
    
    # Render visualization panel
    with col2:
        st.header("Visualization Panel")
        
        # Render control panel
        x_axis_mode, x_axis_group_mapping = render_control_panel(
            group_mapping, 
            kinetic_results_instructions
        )
        
        # Determine axis labels
        x_axis_label, y_axis_label = determine_axis_labels(
            st.session_state.selected_analysis_results,
            kinetic_results_instructions,
            x_axis_mode
        )
        
        # Collect and process plot data
        plot_data = collect_plot_data(
            st.session_state.selected_subsets,
            active_experiments,
            st.session_state.selected_analysis_results,
            kinetic_results_instructions,
            x_axis_mode,
            x_axis_group_mapping,
            group_mapping
        )
        
        # Sort plot data
        plot_data = sort_plot_data(plot_data, x_axis_mode)
        
        # Create and display figure
        fig = create_plotly_figure(plot_data, x_axis_label, y_axis_label, x_axis_mode)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add download button for JSON export
        if plot_data:  # Only show if there's data to export
            json_data = create_json_export_data(
                plot_data,
                x_axis_mode,
                st.session_state.selected_analysis_results,
                kinetic_results_instructions,
                experimental_dataset,
                active_experiments
            )
            
            json_str = json.dumps(json_data, indent=2)
            
            # Create filename with timestamp and x_axis_mode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Simplify x_axis_mode for filename (remove special characters, limit length)
            x_axis_simple = x_axis_mode.replace(' ', '_').replace('/', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            # Limit to 30 characters to keep filename reasonable
            x_axis_simple = x_axis_simple[:4]
            filename = f"{x_axis_simple}_analysis_results_{timestamp}.json"
            
            st.download_button(
                label="📥 Download Data as JSON",
                data=json_str,
                file_name=filename,
                mime="application/json",
                help="Download the currently displayed analysis results and metadata as a JSON file"
            )
        
        # Display metadata table
        render_metadata_table(experimental_dataset, active_experiments)
    
    # Render help section
    render_help_section()
    
    # Footer
    st.markdown("---")
    st.caption("pyKES Analysis Results Visualization | Powered by Streamlit")


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
