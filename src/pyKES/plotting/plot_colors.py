"""Turn a user-supplied colour into one a plotting library will accept.

``Experiment.color`` comes straight out of a metadata column, so it is
whatever someone typed into a spreadsheet cell. Handed to plotly unchecked, a
typo raises ``ValueError`` and the page shows a traceback instead of the plot
— one bad cell in a 44-experiment sheet costs the whole figure.

Two things go wrong in practice:

* a **misspelt name** (``'ligthblue'``), which no library recognises;
* an **empty cell**, which is subtler. ``metadata_dict.get('color', 'black')``
  finds the key present and returns the float ``nan`` pandas put there, so the
  documented ``'black'`` default never applies.

Both land on `FALLBACK_PLOT_COLOR` here. Colours are resolved where they are
*read for plotting*, not where they are stored, so ``overview_df``,
``Experiment.color`` and the metadata listings keep showing what the user
actually wrote — which is what makes the mistake findable.
"""

import matplotlib.colors as mcolors


# Used for every colour no plotting library can make sense of.
FALLBACK_PLOT_COLOR = 'blue'

# CSS functional notation. Valid for plotly, rejected by matplotlib's
# `is_color_like`, so it has to bypass that check rather than fail it.
CSS_FUNCTIONAL_PREFIXES = ('rgb', 'hsl', 'hsv')


def is_css_functional_color(color) -> bool:
    """
    Report whether a colour is written in CSS functional notation.

    Parameters
    ----------
    color : Any
        Colour specification to inspect.

    Returns
    -------
    bool
        True for ``'rgb(...)'``, ``'rgba(...)'``, ``'hsl(...)'`` and friends.
    """

    return (isinstance(color, str)
            and color.strip().lower().startswith(CSS_FUNCTIONAL_PREFIXES))


def plot_color_recognized(color) -> bool:
    """
    Report whether a colour can be drawn as it stands.

    Parameters
    ----------
    color : Any
        Colour specification, typically ``Experiment.color``.

    Returns
    -------
    bool
        True when a plotting library will accept the value. False for a
        misspelt name, an empty cell (``nan``) and anything that is not a
        colour at all.
    """

    return is_css_functional_color(color) or mcolors.is_color_like(color)


def resolve_plot_color(color):
    """
    Express a colour in a form both plotly and matplotlib accept.

    Parameters
    ----------
    color : Any
        Colour specification, typically ``Experiment.color``.

    Returns
    -------
    str
        The colour as hex, unchanged where it is CSS functional notation, or
        `FALLBACK_PLOT_COLOR` where it is unusable.

    Notes
    -----
    Recognised colours are converted to hex rather than passed through: the
    matplotlib-only names ``'tab:blue'`` and ``'C0'`` pass `is_color_like` and
    would then be rejected by plotly.
    """

    if is_css_functional_color(color):
        return color

    if mcolors.is_color_like(color):
        return mcolors.to_hex(color)

    return FALLBACK_PLOT_COLOR


def unrecognized_plot_colors(experiments: dict) -> dict:
    """
    Collect the experiments whose declared colour cannot be drawn.

    Parameters
    ----------
    experiments : dict
        Mapping of experiment name to experiment object.

    Returns
    -------
    dict
        Mapping of experiment name to the unusable value it declares, so a
        page can name both rather than silently drawing everything blue.
    """

    return {experiment_name: experiment.color
            for experiment_name, experiment in experiments.items()
            if not plot_color_recognized(experiment.color)}
