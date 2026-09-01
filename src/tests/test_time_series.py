"""
Tests for the unit handling and axis assignment of the time-series component.

Only the figure construction is exercised; the Streamlit rendering itself
requires a script run context and is not tested here.
"""

import numpy as np
import pytest

from pyKES.streamlit_app.components.time_series_component import (
    DEFAULT_X_AXIS_TITLE,
    DEFAULT_Y_AXIS_TITLE,
    SECONDARY_AXIS_WIDTH,
    assign_y_axes,
    build_figure,
    build_trace_specifications,
    combine_unit_labels,
    resolve_unit_label,
)


class SyntheticExperiment:
    """Minimal stand-in for `Experiment` exposing the plotted dictionaries."""

    def __init__(self, color, raw_data, processed_data):
        self.color = color
        self.raw_data = raw_data
        self.processed_data = processed_data


INSTRUCTIONS = {
    'Raw (O2)': {'x': 'raw_data/time_s',
                 'y': 'raw_data/oxygen'},
    'Reaction (O2)': {'x': 'processed_data/time_s',
                      'y': 'processed_data/amount',
                      'unit_x': 'processed_data/time_unit',
                      'unit_y': 'processed_data/amount_unit'},
    'Rate (O2)': {'x': 'processed_data/time_s',
                  'y': 'processed_data/rate',
                  'x_point': 'processed_data/max_rate_time',
                  'y_point': 'processed_data/max_rate',
                  'unit_x': 'processed_data/time_unit',
                  'unit_y': 'processed_data/rate_unit'},
    'Literal unit': {'x': 'processed_data/time_s',
                     'y': 'processed_data/amount',
                     'unit_x': 's',
                     'unit_y': 'umol / L'},
    'Undeclared unit': {'x': 'processed_data/time_s',
                        'y': 'processed_data/amount',
                        'unit_y': 'processed_data/absent_unit'},
}

TIME = np.arange(5.0)


@pytest.fixture
def experiments():
    return {
        # Exp_001 carries every unit, Exp_002 only the amount ones
        'Exp_001': SyntheticExperiment(
            'blue',
            {'time_s': TIME, 'oxygen': TIME},
            {'time_s': TIME, 'amount': TIME * 2, 'rate': TIME * 0.1,
             'max_rate_time': 3.0, 'max_rate': 0.3,
             'time_unit': 's', 'amount_unit': 'umol', 'rate_unit': 'umol / s'},
        ),
        'Exp_002': SyntheticExperiment(
            'red',
            {'time_s': TIME, 'oxygen': TIME},
            {'time_s': TIME, 'amount': TIME * 3,
             'time_unit': 's', 'amount_unit': 'umol'},
        ),
    }


# ---------------------------------------------------------------------------
# Unit resolution
# ---------------------------------------------------------------------------

def test_unit_paths_and_literals_are_both_resolved(experiments):
    experiment = experiments['Exp_001']

    assert resolve_unit_label(experiment, 'processed_data/rate_unit') == 'umol / s'
    assert resolve_unit_label(experiment, 'umol / s') == 'umol / s'
    assert resolve_unit_label(experiment, None) is None


def test_a_declared_but_absent_unit_path_yields_no_unit(experiments):
    assert resolve_unit_label(experiments['Exp_002'], 'processed_data/rate_unit') is None


def test_missing_units_do_not_drop_the_curve(experiments):
    specifications = build_trace_specifications(
        ['Exp_002'], experiments, INSTRUCTIONS, ['Undeclared unit'])

    assert len(specifications) == 1
    assert specifications[0]['unit_y'] is None


def test_combined_unit_labels_are_distinct():
    assert combine_unit_labels(['umol', 'umol', None]) == 'umol'
    assert combine_unit_labels(['umol', 'mmol']) == 'umol, mmol'
    assert combine_unit_labels([None, None]) is None


# ---------------------------------------------------------------------------
# Axis assignment
# ---------------------------------------------------------------------------

def test_first_unit_keeps_the_primary_axis():
    axis_ids = assign_y_axes(['umol', 'umol / s', 'umol'])

    assert axis_ids == {'umol': 'y', 'umol / s': 'y2', None: 'y'}


def test_curves_without_units_share_the_primary_axis():
    assert assign_y_axes([None, None]) == {None: 'y'}


def test_plot_without_units_keeps_the_generic_titles(experiments):
    plot_types = ['Raw (O2)']
    figure = build_figure(
        build_trace_specifications(['Exp_001'], experiments, INSTRUCTIONS, plot_types),
        plot_types)

    assert figure.layout.xaxis.title.text == DEFAULT_X_AXIS_TITLE
    assert figure.layout.yaxis.title.text == DEFAULT_Y_AXIS_TITLE
    assert 'yaxis2' not in figure.layout


def test_units_become_axis_titles(experiments):
    plot_types = ['Reaction (O2)']
    figure = build_figure(
        build_trace_specifications(['Exp_001'], experiments, INSTRUCTIONS, plot_types),
        plot_types)

    assert figure.layout.xaxis.title.text == 's'
    assert figure.layout.yaxis.title.text == 'umol'


def test_second_y_unit_gets_its_own_right_hand_axis(experiments):
    plot_types = ['Reaction (O2)', 'Rate (O2)']
    specifications = build_trace_specifications(
        ['Exp_001', 'Exp_002'], experiments, INSTRUCTIONS, plot_types)
    figure = build_figure(specifications, plot_types)

    assert figure.layout.yaxis.title.text == 'umol'
    assert figure.layout.yaxis2.title.text == 'umol / s'
    assert figure.layout.yaxis2.side == 'right'
    assert figure.layout.yaxis2.overlaying == 'y'

    # The rate curve and its highlighted point share the secondary axis
    axes_by_trace = {trace.name: trace.yaxis for trace in figure.data}
    assert axes_by_trace['Exp_001 - Reaction (O2)'] == 'y'
    assert axes_by_trace['Exp_001 - Rate (O2)'] == 'y2'
    assert axes_by_trace['Exp_001 - Rate (O2) Point'] == 'y2'
    assert axes_by_trace['Exp_002 - Reaction (O2)'] == 'y'


def test_third_axis_shortens_the_plotting_area(experiments):
    plot_types = ['Reaction (O2)', 'Rate (O2)', 'Literal unit']
    figure = build_figure(
        build_trace_specifications(['Exp_001'], experiments, INSTRUCTIONS, plot_types),
        plot_types)

    assert figure.layout.xaxis.domain == (0.0, 1.0 - SECONDARY_AXIS_WIDTH)
    assert figure.layout.yaxis2.position == pytest.approx(1.0 - SECONDARY_AXIS_WIDTH)
    assert figure.layout.yaxis3.position == pytest.approx(1.0)
    assert figure.layout.yaxis3.title.text == 'umol / L'


def test_selection_order_decides_the_primary_axis(experiments):
    plot_types = ['Rate (O2)', 'Reaction (O2)']
    figure = build_figure(
        build_trace_specifications(['Exp_001'], experiments, INSTRUCTIONS, plot_types),
        plot_types)

    assert figure.layout.yaxis.title.text == 'umol / s'
    assert figure.layout.yaxis2.title.text == 'umol'


def test_hover_template_carries_the_units(experiments):
    plot_types = ['Reaction (O2)']
    figure = build_figure(
        build_trace_specifications(['Exp_001'], experiments, INSTRUCTIONS, plot_types),
        plot_types)

    assert 'X: %{x} s' in figure.data[0].hovertemplate
    assert 'Y: %{y} umol' in figure.data[0].hovertemplate
