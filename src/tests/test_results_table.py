"""
Tests for the analysis results table component.

Only the value resolution and table assembly are exercised; the Streamlit
rendering itself requires a script run context and is not tested here.
"""

import numpy as np
import pytest

from pyKES.streamlit_app.components.results_table_component import (
    MISSING_VALUE_PLACEHOLDER,
    build_cell_text,
    build_results_table,
    coerce_to_scalar,
    group_experiments,
)
from pyKES.utilities.unit_handler import Quantity


class SyntheticExperiment:
    """Minimal stand-in for `Experiment` exposing `processed_data` and `group`."""

    def __init__(self, group, processed_data):
        self.group = group
        self.processed_data = processed_data


INSTRUCTIONS = {
    'Max. rate (mmol/h/g)': {'result': 'processed_data/max_rate',
                             'format': '.2f',
                             'error': 'processed_data/max_rate_error'},
    'Apparent quantum yield (%)': {'result': 'processed_data/apparent_quantum_yield'},
}


@pytest.fixture
def experiments():
    return {
        'Exp_001': SyntheticExperiment('Intensity', {
            'max_rate': 12.345,
            'max_rate_error': 0.678,
            'apparent_quantum_yield': 3.21,
        }),
        'Exp_002': SyntheticExperiment('Intensity', {
            'max_rate': np.array([18.0]),
        }),
        'Exp_003': SyntheticExperiment('Reference', {}),
    }


def test_table_shape_and_headers(experiments):
    table = build_results_table(list(experiments.keys()), experiments, INSTRUCTIONS)

    assert list(table.index) == ['Exp_001', 'Exp_002', 'Exp_003']
    assert list(table.columns) == list(INSTRUCTIONS.keys())


def test_values_are_formatted_with_uncertainty(experiments):
    table = build_results_table(['Exp_001'], experiments, INSTRUCTIONS)

    assert table.loc['Exp_001', 'Max. rate (mmol/h/g)'] == '12.35 ± 0.68'
    assert table.loc['Exp_001', 'Apparent quantum yield (%)'] == '3.21'


def test_missing_values_become_placeholders(experiments):
    table = build_results_table(['Exp_002', 'Exp_003'], experiments, INSTRUCTIONS)

    # Single-element arrays are unwrapped, absent paths are placeholders
    assert table.loc['Exp_002', 'Max. rate (mmol/h/g)'] == '18.00'
    assert table.loc['Exp_002', 'Apparent quantum yield (%)'] == MISSING_VALUE_PLACEHOLDER
    assert (table.loc['Exp_003'] == MISSING_VALUE_PLACEHOLDER).all()


def test_unknown_experiments_are_skipped(experiments):
    table = build_results_table(['Exp_001', 'Deleted_experiment'], experiments, INSTRUCTIONS)

    assert list(table.index) == ['Exp_001']


def test_empty_selection_keeps_the_columns(experiments):
    table = build_results_table([], experiments, INSTRUCTIONS)

    assert list(table.columns) == list(INSTRUCTIONS.keys())
    assert table.empty


def test_quantity_is_converted_to_requested_unit():
    experiment = SyntheticExperiment('Intensity', {'rate': Quantity(3600.0, 'mmol / s')})
    cell = build_cell_text(experiment, {'result': 'processed_data/rate',
                                        'unit': 'mmol / h',
                                        'format': '.0f'})

    assert cell == '12960000 mmol / h'


def test_multi_element_arrays_are_not_representable():
    assert coerce_to_scalar(np.array([1.0, 2.0])) is None


def test_experiments_are_bucketed_by_group(experiments):
    experiments_by_group = group_experiments(experiments)

    assert sorted(experiments_by_group.keys()) == ['Intensity', 'Reference']
    assert [name for name, _ in experiments_by_group['Intensity']] == ['Exp_001', 'Exp_002']
