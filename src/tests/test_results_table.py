"""
Tests for the analysis results table component.

Only the value resolution and table assembly are exercised; the Streamlit
rendering itself requires a script run context and is not tested here.

The table holds numbers rather than formatted text, which is what makes the
header sort compare magnitudes; `test_columns_sort_by_magnitude` is the
regression that pins it.
"""

import numpy as np
import pandas as pd
import pytest

from pyKES.streamlit_app.components.results_table_component import (
    build_column_config,
    build_number_format,
    build_results_table,
    coerce_to_scalar,
    error_column_name,
    group_experiments,
    join_metadata_columns,
    resolve_result_number,
    select_instructions,
)
from pyKES.utilities.unit_handler import Quantity


class SyntheticExperiment:
    """Minimal stand-in for `Experiment` exposing `processed_data` and `group`."""

    def __init__(self, group, processed_data):
        self.group = group
        self.processed_data = processed_data


MAX_RATE_COLUMN = 'Max. rate (mmol/h/g)'
QUANTUM_YIELD_COLUMN = 'Apparent quantum yield (%)'

INSTRUCTIONS = {
    MAX_RATE_COLUMN: {'result': 'processed_data/max_rate',
                      'format': '.2f',
                      'error': 'processed_data/max_rate_error'},
    QUANTUM_YIELD_COLUMN: {'result': 'processed_data/apparent_quantum_yield'},
}


@pytest.fixture
def experiments():
    return {
        'Exp_001': SyntheticExperiment('Intensity', {
            'max_rate': 12.345,
            'max_rate_error': 0.678,
            'apparent_quantum_yield': 9.0,
        }),
        'Exp_002': SyntheticExperiment('Intensity', {
            'max_rate': np.array([18.0]),
            'apparent_quantum_yield': 18.0,
        }),
        'Exp_003': SyntheticExperiment('Reference', {}),
    }


def test_table_shape_and_headers(experiments):
    table = build_results_table(list(experiments.keys()), experiments, INSTRUCTIONS)

    assert list(table.index) == ['Exp_001', 'Exp_002', 'Exp_003']
    # The uncertainty is a column of its own, so both halves stay sortable
    assert list(table.columns) == [MAX_RATE_COLUMN,
                                   error_column_name(MAX_RATE_COLUMN),
                                   QUANTUM_YIELD_COLUMN]


def test_values_stay_numbers(experiments):
    table = build_results_table(['Exp_001'], experiments, INSTRUCTIONS)

    assert table.loc['Exp_001', MAX_RATE_COLUMN] == pytest.approx(12.345)
    assert table.loc['Exp_001', error_column_name(MAX_RATE_COLUMN)] == pytest.approx(0.678)
    assert table.loc['Exp_001', QUANTUM_YIELD_COLUMN] == pytest.approx(9.0)


def test_columns_sort_by_magnitude(experiments):
    table = build_results_table(['Exp_002', 'Exp_001'], experiments, INSTRUCTIONS)

    # Formatted as text, '18.00' sorted before '9.00'; as numbers it does not.
    assert list(table.sort_values(QUANTUM_YIELD_COLUMN).index) == ['Exp_001', 'Exp_002']
    assert list(table.sort_values(MAX_RATE_COLUMN).index) == ['Exp_001', 'Exp_002']


def test_missing_values_are_empty_cells(experiments):
    table = build_results_table(['Exp_002', 'Exp_003'], experiments, INSTRUCTIONS)

    # Single-element arrays are unwrapped, absent paths leave the cell empty
    assert table.loc['Exp_002', MAX_RATE_COLUMN] == pytest.approx(18.0)
    assert pd.isna(table.loc['Exp_002', error_column_name(MAX_RATE_COLUMN)])
    assert table.loc['Exp_003'].isna().all()


def test_unknown_experiments_are_skipped(experiments):
    table = build_results_table(['Exp_001', 'Deleted_experiment'], experiments, INSTRUCTIONS)

    assert list(table.index) == ['Exp_001']


def test_empty_selection_keeps_the_columns(experiments):
    table = build_results_table([], experiments, INSTRUCTIONS)

    assert list(table.columns) == [MAX_RATE_COLUMN,
                                   error_column_name(MAX_RATE_COLUMN),
                                   QUANTUM_YIELD_COLUMN]
    assert table.empty


def test_only_selected_results_become_columns(experiments):
    selected = select_instructions(INSTRUCTIONS, [QUANTUM_YIELD_COLUMN])
    table = build_results_table(list(experiments), experiments, selected)

    assert list(table.columns) == [QUANTUM_YIELD_COLUMN]


def test_quantity_is_converted_to_requested_unit():
    experiment = SyntheticExperiment('Intensity', {'rate': Quantity(3600.0, 'mmol / s')})

    assert resolve_result_number(experiment, 'processed_data/rate',
                                 'mmol / h') == pytest.approx(12960000.0)


def test_multi_element_arrays_are_not_representable():
    assert coerce_to_scalar(np.array([1.0, 2.0])) is None


def test_number_format_translates_the_python_spec():
    assert build_number_format('.2f', None) == '%.2f'
    assert build_number_format('.4g', 'mmol / h') == '%.4g mmol / h'
    # A literal '%' would otherwise start a conversion of its own
    assert build_number_format('.1f', '%') == '%.1f %%'


def test_only_numeric_columns_are_given_a_number_format(experiments):
    experiments['Exp_004'] = SyntheticExperiment('Reference',
                                                 {'apparent_quantum_yield': 'not measured'})
    table = build_results_table(list(experiments), experiments, INSTRUCTIONS)

    column_config = build_column_config(table, INSTRUCTIONS)

    assert QUANTUM_YIELD_COLUMN not in column_config
    assert MAX_RATE_COLUMN in column_config


def test_metadata_columns_join_to_the_left(experiments):
    table = build_results_table(['Exp_001', 'Exp_002'], experiments, INSTRUCTIONS)
    table.index.name = 'Experiment'
    overview_df = pd.DataFrame({'Experiment': ['Exp_001', 'Exp_002'],
                                'Irradiance [mW/cm2]': [40.0, 80.0],
                                'Comment': ['first', 'second']})

    joined = join_metadata_columns(table, overview_df, ['Irradiance [mW/cm2]'])

    assert list(joined.columns)[0] == 'Irradiance [mW/cm2]'
    assert list(joined.index) == ['Exp_001', 'Exp_002']
    # Taken from overview_df unchanged, so the column keeps its own dtype
    assert joined['Irradiance [mW/cm2]'].tolist() == [40.0, 80.0]
    assert 'Comment' not in joined.columns


def test_experiments_are_bucketed_by_group(experiments):
    experiments_by_group = group_experiments(experiments)

    assert sorted(experiments_by_group.keys()) == ['Intensity', 'Reference']
    assert [name for name, _ in experiments_by_group['Intensity']] == ['Exp_001', 'Exp_002']
