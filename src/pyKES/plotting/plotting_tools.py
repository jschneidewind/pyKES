"""
Plotting tools for visualizing analysis results from JSON files.

This module provides functions to create plots from JSON files containing
analysis results with statistical data (mean, min, max, std) at different
x-values.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from matplotlib import rcParams

# Set matplotlib style
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Arial'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.bf'] = 'Arial:bold'

def load_json_data(json_path: str) -> Dict[str, Any]:
    """
    Load JSON data from file.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    
    Returns
    -------
    dict
        Loaded JSON data.
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def is_x_axis_numerical(data: Dict[str, Any]) -> bool:
    """
    Determine if x-axis data is numerical or categorical.
    
    Parameters
    ----------
    data : dict
        Full JSON data dictionary.
    
    Returns
    -------
    bool
        True if x-axis is numerical, False if categorical.
    """
    x_axis_type = data['metadata']['x_axis'].get('type', 'numerical')
    return x_axis_type == 'numerical'


def extract_plotting_data(
    data: Dict[str, Any],
    analysis_result_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, Optional[List[str]]]:
    """
    Extract x, y, and error bar data for a specific analysis result.
    
    Parameters
    ----------
    data : dict
        Full JSON data dictionary.
    analysis_result_name : str
        Name of the analysis result to extract.
    
    Returns
    -------
    x_values : np.ndarray
        Array of x values (numerical indices for categorical, floats for numerical).
    y_values : np.ndarray
        Array of mean y values.
    yerr_lower : np.ndarray
        Lower error bar values (mean - min).
    yerr_upper : np.ndarray
        Upper error bar values (max - mean).
    unit : str
        Unit string for the y-axis.
    x_labels : list of str or None
        List of x-axis labels for categorical data. None for numerical data.
    """
    analysis_data = data['plotting_data'][analysis_result_name]
    unit = analysis_data['unit']
    
    # Get x-axis type
    is_numerical = is_x_axis_numerical(data)
    
    # Extract x values
    x_str_values = list(analysis_data['data'].keys())
    
    if is_numerical:
        # Convert to float and sort numerically
        x_values = np.array([float(x) for x in x_str_values])
        sort_indices = np.argsort(x_values)
        x_values = x_values[sort_indices]
        x_labels = None
    else:
        # Categorical data - use sequential indices and preserve labels
        x_labels = x_str_values
        x_values = np.arange(len(x_str_values))
        sort_indices = np.arange(len(x_str_values))  # Keep original order
    
    # Extract y values and errors in sorted order
    y_values = []
    yerr_lower = []
    yerr_upper = []
    
    for idx in sort_indices:
        x_str = x_str_values[idx]
        point_data = analysis_data['data'][x_str]
        mean = point_data['mean']
        y_values.append(mean)
        yerr_lower.append(mean - point_data['min'])
        yerr_upper.append(point_data['max'] - mean)
    
    return (
        x_values,
        np.array(y_values),
        np.array(yerr_lower),
        np.array(yerr_upper),
        unit,
        x_labels
    )


def get_x_axis_label(data: Dict[str, Any]) -> str:
    """
    Extract x-axis label from metadata.
    
    Parameters
    ----------
    data : dict
        Full JSON data dictionary.
    
    Returns
    -------
    str
        X-axis label.
    """
    return data['metadata']['x_axis'].get('label', 'X-axis')


def check_units_compatibility(units: List[str]) -> bool:
    """
    Check if all units are the same.
    
    Parameters
    ----------
    units : list of str
        List of unit strings.
    
    Returns
    -------
    bool
        True if all units are identical, False otherwise.
    """
    return len(set(units)) == 1


def plot_single_analysis_result(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    yerr_lower: np.ndarray,
    yerr_upper: np.ndarray,
    label: str,
    color: str = 'blue',
    linestyle: Optional[str] = None,
    marker: str = 'o',
    markersize: float = 6,
    capsize: float = 3,
    **kwargs
) -> None:
    """
    Plot a single analysis result on the given axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x_values : np.ndarray
        X coordinates.
    y_values : np.ndarray
        Y coordinates (mean values).
    yerr_lower : np.ndarray
        Lower error bar values.
    yerr_upper : np.ndarray
        Upper error bar values.
    label : str
        Label for the plot legend.
    color : str, optional
        Color for the plot (default: 'blue').
    linestyle : str or None, optional
        Line style ('-', '--', '-.', ':', etc.). If None, no line is drawn.
    marker : str, optional
        Marker style (default: 'o').
    markersize : float, optional
        Size of markers (default: 6).
    capsize : float, optional
        Size of error bar caps (default: 3).
    **kwargs
        Additional keyword arguments passed to errorbar.
    """
    # Create asymmetric error bars
    yerr = np.array([yerr_lower, yerr_upper])
    
    # Plot with or without line
    if linestyle is not None:
        ax.errorbar(
            x_values, y_values, yerr=yerr,
            label=label, color=color, linestyle=linestyle,
            marker=marker, markersize=markersize, capsize=capsize,
            **kwargs
        )
    else:
        ax.errorbar(
            x_values, y_values, yerr=yerr,
            label=label, color=color, linestyle='',
            marker=marker, markersize=markersize, capsize=capsize,
            **kwargs
        )


def plot_analysis_results(
    json_path: str,
    analysis_results: Optional[Union[str, List[str]]] = None,
    ax: Optional[Axes] = None,
    colors: Optional[Union[str, Dict[str, str]]] = None,
    linestyles: Optional[Union[str, Dict[str, str]]] = None,
    marker: str = 'o',
    markersize: float = 6,
    capsize: float = 3,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    legend: bool = True,
    legend_loc: Optional[Union[str, int]] = None,
    grid: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ylim2: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylabel2: Optional[str] = None,
    result_labels: Optional[Dict[str, str]] = None,
    **kwargs
) -> Tuple[plt.Figure, Union[Axes, Tuple[Axes, Axes]]]:
    """
    Create plots from JSON file containing analysis results.
    
    This function automatically handles multiple analysis results with different
    units by creating dual y-axes when needed. If all analysis results share
    the same unit, they are plotted on a single axis.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    analysis_results : str, list of str, or None, optional
        Name(s) of analysis result(s) to plot. If None, plots all available
        analysis results.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, creates new figure and axes.
    colors : str, dict, or None, optional
        Color specification. Can be:
        - Single color string: used for all analysis results
        - Dict mapping analysis result names to colors
        - None: uses default color cycle
    linestyles : str, dict, or None, optional
        Line style specification. Can be:
        - Single style string: used for all analysis results
        - Dict mapping analysis result names to line styles
        - None: no lines drawn (markers only)
    marker : str, optional
        Marker style (default: 'o').
    markersize : float, optional
        Size of markers (default: 6).
    capsize : float, optional
        Size of error bar caps (default: 3).
    figsize : tuple of float, optional
        Figure size as (width, height) in inches (default: (8, 6)).
        Only used if ax is None.
    title : str or None, optional
        Plot title. If None, no title is set.
    legend : bool, optional
        Whether to show legend (default: True).
    legend_loc : str, int, or None, optional
        Legend location. Can be a string ('best', 'upper right', 'upper left',
        'lower left', 'lower right', 'right', 'center left', 'center right',
        'lower center', 'upper center', 'center') or an integer code (0-10).
        If None, uses 'best' for single axis and 'best' for dual axis.
    grid : bool, optional
        Whether to show grid (default: False).
    xlim : tuple of float or None, optional
        X-axis limits as (xmin, xmax). If None, uses automatic scaling.
    ylim : tuple of float or None, optional
        Y-axis limits as (ymin, ymax). If None, uses automatic scaling.
    ylim2 : tuple of float or None, optional
        Secondary y-axis limits as (ymin, ymax) for dual y-axis plots.
        If None, uses automatic scaling. Only used when plotting results
        with different units.
    xlabel : str or None, optional
        Custom x-axis label. If None, uses label from JSON metadata.
    ylabel : str or None, optional
        Custom y-axis label. If None, uses unit from first analysis result.
    ylabel2 : str or None, optional
        Custom secondary y-axis label for dual y-axis plots.
        If None, uses unit from second analysis result.
    result_labels : dict or None, optional
        Dictionary mapping analysis result names to custom display labels.
        If None or if a result name is not in the dict, uses the original
        analysis result name.
        Example: {'k1': 'Rate constant k₁', 'k2': 'Rate constant k₂'}
    **kwargs
        Additional keyword arguments passed to errorbar.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes.Axes or tuple of Axes
        Axes object(s). Returns tuple (ax1, ax2) if dual y-axes are created.
    
    Examples
    --------
    Plot all analysis results with default settings:
    
    >>> fig, ax = plot_analysis_results('results.json')
    >>> plt.show()
    
    Plot specific results with custom colors and line styles:
    
    >>> fig, ax = plot_analysis_results(
    ...     'results.json',
    ...     analysis_results=['H2 max rate', 'O2 max rate'],
    ...     colors={'H2 max rate': 'blue', 'O2 max rate': 'red'},
    ...     linestyles={'H2 max rate': '-', 'O2 max rate': '--'}
    ... )
    
    Plot on existing axes:
    
    >>> fig, ax = plt.subplots()
    >>> fig, ax = plot_analysis_results('results.json', ax=ax)
    """
    # Load data
    data = load_json_data(json_path)
    
    # Determine which analysis results to plot
    if analysis_results is None:
        analysis_results = list(data['plotting_data'].keys())
    elif isinstance(analysis_results, str):
        analysis_results = [analysis_results]
    
    # Extract data for all analysis results
    plot_data = {}
    units = []
    x_labels = None
    for result_name in analysis_results:
        x, y, yerr_lower, yerr_upper, unit, x_lab = extract_plotting_data(data, result_name)
        plot_data[result_name] = (x, y, yerr_lower, yerr_upper)
        units.append(unit)
        if x_lab is not None:
            x_labels = x_lab  # Use categorical labels if available
    
    # Check if units are compatible
    same_units = check_units_compatibility(units)
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        ax1 = ax
    
    # Prepare color mapping
    if colors is None:
        color_map = {}
    elif isinstance(colors, str):
        color_map = {name: colors for name in analysis_results}
    else:
        color_map = colors
    
    # Prepare linestyle mapping
    if linestyles is None:
        linestyle_map = {name: None for name in analysis_results}
    elif isinstance(linestyles, str):
        linestyle_map = {name: linestyles for name in analysis_results}
    else:
        linestyle_map = linestyles
    
    # Get x-axis label (use custom if provided)
    x_label = xlabel if xlabel is not None else get_x_axis_label(data)
    
    # Prepare result label mapping
    if result_labels is None:
        label_map = {name: name for name in analysis_results}
    else:
        label_map = {name: result_labels.get(name, name) for name in analysis_results}
    
    # Set default legend location
    if legend_loc is None:
        legend_loc = 'best'
    
    # Plot based on unit compatibility
    if same_units or len(analysis_results) == 1:
        # All on one axis
        for result_name in analysis_results:

            x, y, yerr_lower, yerr_upper = plot_data[result_name]
            color = color_map.get(result_name, None)
            linestyle = linestyle_map.get(result_name, None)
            display_label = label_map[result_name]
            
            plot_single_analysis_result(
                ax1, x, y, yerr_lower, yerr_upper,
                label=display_label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
                capsize=capsize,
                **kwargs
            )
        
        ax1.set_xlabel(x_label)
        y_label = ylabel if ylabel is not None else units[0]
        ax1.set_ylabel(y_label)
        
        # Set x-tick labels for categorical data
        if x_labels is not None:
            ax1.set_xticks(range(len(x_labels)))
            ax1.set_xticklabels(x_labels)
        
        # Apply axis limits
        if xlim is not None:
            ax1.set_xlim(xlim)
        if ylim is not None:
            ax1.set_ylim(ylim)
        
        if grid:
            ax1.grid(True, alpha=0.3)
        if legend:
            ax1.legend(loc=legend_loc)
        if title:
            ax1.set_title(title)
        
        return fig, ax1
    
    else:
        # Multiple y-axes needed (currently supports up to 2)
        if len(analysis_results) > 2:
            raise ValueError(
                "Plotting more than 2 analysis results with different units "
                "is not currently supported. Please plot them separately or "
                "ensure they share the same units."
            )
        
        # Create twin axis
        ax2 = ax1.twinx()
        
        # Plot first result on left axis
        result_name = analysis_results[0]
        x, y, yerr_lower, yerr_upper = plot_data[result_name]
        color = color_map.get(result_name, 'blue')
        linestyle = linestyle_map.get(result_name, None)
        display_label = label_map[result_name]
        
        plot_single_analysis_result(
            ax1, x, y, yerr_lower, yerr_upper,
            label=display_label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            capsize=capsize,
            **kwargs
        )
        
        ax1.set_xlabel(x_label)
        y_label = ylabel if ylabel is not None else units[0]
        ax1.set_ylabel(y_label, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Set x-tick labels for categorical data
        if x_labels is not None:
            ax1.set_xticks(range(len(x_labels)))
            ax1.set_xticklabels(x_labels)
        
        # Plot second result on right axis
        result_name = analysis_results[1]
        x, y, yerr_lower, yerr_upper = plot_data[result_name]
        color = color_map.get(result_name, 'red')
        linestyle = linestyle_map.get(result_name, None)
        display_label = label_map[result_name]
        
        plot_single_analysis_result(
            ax2, x, y, yerr_lower, yerr_upper,
            label=display_label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            capsize=capsize,
            **kwargs
        )
        
        y_label2 = ylabel2 if ylabel2 is not None else units[1]
        ax2.set_ylabel(y_label2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Apply axis limits
        if xlim is not None:
            ax1.set_xlim(xlim)
        if ylim is not None:
            ax1.set_ylim(ylim)
        if ylim2 is not None:
            ax2.set_ylim(ylim2)
        
        if grid:
            ax1.grid(True, alpha=0.3)
        
        if legend:
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc)
        
        if title:
            ax1.set_title(title)
        
        return fig, (ax1, ax2)
