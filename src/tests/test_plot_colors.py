"""
Tests for the plot-colour guard.

The values covered are the ones a metadata column actually produces: a name
somebody typed, a hex code, a matplotlib-only name that plotly would reject,
CSS functional notation, and the two failure modes — a typo and an empty cell,
which pandas hands over as ``nan``.
"""

import pytest

from pyKES.plotting.plot_colors import (
    FALLBACK_PLOT_COLOR,
    plot_color_recognized,
    resolve_plot_color,
    unrecognized_plot_colors,
)


class SyntheticExperiment:
    """Minimal stand-in for `Experiment` exposing only its colour."""

    def __init__(self, color):
        self.color = color


@pytest.mark.parametrize('color, expected', [
    ('red', '#ff0000'),
    ('#1f77b4', '#1f77b4'),
    ((0.0, 0.0, 1.0), '#0000ff'),
])
def test_usable_colors_become_hex(color, expected):
    assert resolve_plot_color(color) == expected


def test_matplotlib_only_names_are_converted():
    # 'tab:blue' and 'C0' pass matplotlib's check and are then rejected by
    # plotly, so passing them through unchanged would not be enough.
    assert resolve_plot_color('tab:blue') == '#1f77b4'
    assert resolve_plot_color('C0') == '#1f77b4'


def test_css_functional_notation_is_passed_through():
    # Valid for plotly, rejected by matplotlib's is_color_like
    assert resolve_plot_color('rgb(31, 119, 180)') == 'rgb(31, 119, 180)'
    assert resolve_plot_color('rgba(31, 119, 180, 0.5)') == 'rgba(31, 119, 180, 0.5)'


@pytest.mark.parametrize('color', ['ligthblue', '', '   ', float('nan'), None, 42])
def test_unusable_colors_fall_back(color):
    assert resolve_plot_color(color) == FALLBACK_PLOT_COLOR
    assert not plot_color_recognized(color)


def test_unrecognized_colors_are_reported_with_their_value():
    experiments = {
        'Exp_001': SyntheticExperiment('red'),
        'Exp_002': SyntheticExperiment('ligthblue'),
        'Exp_003': SyntheticExperiment(float('nan')),
    }

    unusable = unrecognized_plot_colors(experiments)

    assert sorted(unusable) == ['Exp_002', 'Exp_003']
    assert unusable['Exp_002'] == 'ligthblue'


def test_nothing_is_reported_when_every_color_is_usable():
    experiments = {'Exp_001': SyntheticExperiment('red'),
                   'Exp_002': SyntheticExperiment('rgb(1, 2, 3)')}

    assert unrecognized_plot_colors(experiments) == {}
