import matplotlib.colors as mcolors
import numpy as np

def lighten_color(color, amount=0.3):
    """
    Lightens the given color by mixing it with white.
    
    Parameters
    ----------
    color : color specification
        Color to lighten (can be name, hex, RGB tuple, etc.)
    amount : float
        Amount to lighten (0 = original color, 1 = white)
        
    Returns
    -------
    tuple
        RGB tuple of the lightened color
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = color
    c = np.array(c)
    white = np.array([1, 1, 1])
    return tuple(c + (white - c) * amount)