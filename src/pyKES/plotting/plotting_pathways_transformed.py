import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from matplotlib import rcParams
#
# ----------------------------------------
# Font settings of Water Splitting Group
# ----------------------------------------
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12 #important: for presentations change the font.size to 18
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Arial'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.bf'] = 'Arial:bold'

def plot_elliptical_band(data, 
                         ax, 
                         curvature=0.8, 
                         bulge=1.6, 
                         bar_offset=0.05,
                         color='#0F202E', 
                         alpha=0.6,
                         linewidth=2,
                         bar_linewidth=2):
    """
    Plots vertical bars at offsets from A/B and C/D, connected by a U-shaped curve.
    
    Parameters
    ----------
    data : dict
        Dictionary with 'source' and 'target' keys containing coordinates.
        - source: {'x': float, 'y': (y_bottom, y_top)}
        - target: {'x': float, 'y': (y_bottom, y_top)}
    ax : matplotlib.axes.Axes
        Axes to plot on.
    curvature : float
        Vertical depth of the loop relative to width.
    bulge : float
        Horizontal outward curve magnitude relative to width.
    bar_offset : float
        Horizontal offset for the vertical bars from the original x positions.
    color : str
        Color for both bars and curve.
    alpha : float
        Transparency.
    linewidth : float
        Width of the curved line.
    bar_linewidth : float
        Width of the vertical bars.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted elements.
    """

    x1 = data['target']['x']
    x2 = data['source']['x']
    yA = data['target']['y'][1]
    yB = data['target']['y'][0]
    yC = data['source']['y'][1]
    yD = data['source']['y'][0]

    # Calculate bar positions with offsets
    x_bar_left = x1 - bar_offset   # Bar to the left of A/B
    x_bar_right = x2 + bar_offset  # Bar to the right of C/D

    # Plot vertical bar at left (A/B side)
    ax.plot([x_bar_left, x_bar_left], [yB, yA], 
            color=color, linewidth=bar_linewidth, solid_capstyle='butt')
    
    # Plot vertical bar at right (C/D side)
    ax.plot([x_bar_right, x_bar_right], [yD, yC], 
            color=color, linewidth=bar_linewidth, solid_capstyle='butt')

    # Calculate midpoints of the bars
    mid_AB = (yA + yB) / 2
    mid_CD = (yC + yD) / 2

    # Width is now between the two bars
    width = x_bar_right - x_bar_left
    min_y = min(mid_AB, mid_CD)
    
    # Control points for the U-shaped curve
    ctrl_x_left = x_bar_left - (bulge * width)
    ctrl_x_right = x_bar_right + (bulge * width)
    ctrl_y = min_y - (curvature * width)

    # Construct the path as a single curved line
    verts = [
        (x_bar_left, mid_AB),       # Start at midpoint of left bar
        (ctrl_x_left, ctrl_y),      # Control Point 1
        (ctrl_x_right, ctrl_y),     # Control Point 2
        (x_bar_right, mid_CD),      # End at midpoint of right bar
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, alpha=alpha, linewidth=linewidth)
    ax.add_patch(patch)
    
    return ax

def plot_curved_band(data, 
                     ax=None, 
                     color='steelblue', 
                     alpha=0.6,
                     bar_offset=0.05,
                     bar_linewidth=2):
    """
    Plot a smooth curved band connecting source and target points with vertical bars.
    
    Parameters
    ----------
    data : dict
        Dictionary with 'source' and 'target' keys, each containing
        'x' (float) and 'y' (tuple of two floats) values.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    color : str, optional
        Fill color for the band.
    alpha : float, optional
        Transparency of the band.
    bar_offset : float, optional
        Horizontal offset for the vertical bars from the original x positions.
    bar_linewidth : float, optional
        Width of the vertical bars.
    
    Returns
    -------
    PathPatch
        The matplotlib patch object.
    """
    if ax is None:
        ax = plt.gca()
    
    # Extract coordinates
    x0 = data['source']['x']
    x1 = data['target']['x']
    y0_bottom, y0_top = data['source']['y']
    y1_bottom, y1_top = data['target']['y']
    
    # Calculate bar positions with offsets
    x_bar_source = x0 + bar_offset   # Bar to the right of source
    x_bar_target = x1 - bar_offset   # Bar to the left of target
    
    # Plot vertical bar at source side
    ax.plot([x_bar_source, x_bar_source], [y0_bottom, y0_top], 
            color=color, linewidth=bar_linewidth, solid_capstyle='butt')
    
    # Plot vertical bar at target side
    ax.plot([x_bar_target, x_bar_target], [y1_bottom, y1_top], 
            color=color, linewidth=bar_linewidth, solid_capstyle='butt')
    
    # Control point offset for smooth curves (based on distance between bars)
    ctrl_offset = (x_bar_target - x_bar_source) / 2
    
    # Define the path using cubic Bézier curves
    # Top edge: source top -> target top
    # Bottom edge: target bottom -> source bottom (reversed to close the shape)
    
    vertices = [
        # Start at source bar top
        (x_bar_source, y0_top),
        # Cubic Bézier to target bar top
        (x_bar_source + ctrl_offset, y0_top),  # control point 1
        (x_bar_target - ctrl_offset, y1_top),  # control point 2
        (x_bar_target, y1_top),                # end point
        # Line to target bar bottom
        (x_bar_target, y1_bottom),
        # Cubic Bézier back to source bar bottom
        (x_bar_target - ctrl_offset, y1_bottom),  # control point 1
        (x_bar_source + ctrl_offset, y0_bottom),  # control point 2
        (x_bar_source, y0_bottom),                # end point
        # Close path
        (x_bar_source, y0_top),
    ]
    
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,  # top curve
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,  # bottom curve
        Path.CLOSEPOLY,
    ]
    
    path = Path(vertices, codes)
    patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha)
    ax.add_patch(patch)
    
    return patch

def plot_pathway_bars(data, figsize=(10, 8), 
                      colormap = 'tab20', 
                      ax = None,
                      excluded_nodes = [],
                      excluded_links = [],
                      node_linewidth = 5,
                      label_offset = 0.15,
                      label_fontsize = 12,
                      save_figure = False,
                      file_name = 'pathway_plot',
                      forward_link_kwargs = {'alpha': 0.6},
                      backward_link_kwargs = {'alpha': 0.8},
                      ):
    """
    Plot nodes as vertical bars from y_min to y_max at their level.
    
    Parameters
    ----------
    data : dict
        Pathways data with 'nodes' containing node information.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Collect unique species names and assign colors
    species_names = set()
    for node in data['nodes'].values():
        if 'y_min' in node and 'y_max' in node:
            species_names.add(node['name'])
    
    cmap = plt.colormaps[colormap]
    color_map = {name: cmap(i % 10) for i, name in enumerate(sorted(species_names))}
    
    for node_id, node in data['nodes'].items():
        # Only plot nodes that have y_min and y_max
        if 'y_min' not in node or 'y_max' not in node:
            continue

        if node_id in excluded_nodes:
            continue
        
        level = node['level']
        y_min = node['y_min']
        y_max = node['y_max']
        name = node['name']
        color = color_map[name]
        
        # Plot vertical bar
        ax.plot([level, level], [y_min, y_max], linewidth=node_linewidth, 
                solid_capstyle='butt', color=color)
        
        # Add label at the center of the bar
        ax.text(level, y_max + label_offset, name, fontsize=label_fontsize, 
                va='center', ha = 'center', fontweight='bold', color = color)
    
    for link in data['links']:

        if link['source'] in excluded_nodes or link['target'] in excluded_nodes:
            continue

        # Check if link should be excluded by (source, target) tuple
        if (link['source'], link['target']) in excluded_links:
            continue

        target = data['nodes'][link['target']]
        color_of_target = color_map[target['name']]

        if link['looping'] == False:
            plot_curved_band(link['coordinates'], color=color_of_target,ax=ax, **forward_link_kwargs) 
        else:
            plot_elliptical_band(link['coordinates'], color=color_of_target, ax=ax, **backward_link_kwargs)

    # hide axes
    ax.axis('off')
    
    plt.tight_layout()

    if save_figure:
        fig.savefig(f"{file_name}.pdf")



