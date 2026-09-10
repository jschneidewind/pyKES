"""
Tests for the analysis results table component.

Only the value resolution and table assembly are exercised; the Streamlit
rendering itself requires a script run context and is not tested here.

The table holds numbers rather than formatted text, which is what makes the
header sort compare magnitudes; `test_columns_sort_by_magnitude` is the
regression that pins it. The formatting happens on the display side of a
Styler, so `test_cells_are_formatted_as_python_does` pins the digits that
reach the screen.
"""

import numpy as np
import pandas as pd
import pytest

from pyKES.streamlit_app.components.results_table_component import (
    MISSING_VALUE_PLACEHOLDER,
    build_results_table,
    coerce_to_scalar,
    error_column_name,
    format_result_cell,
    group_experiments,
    join_metadata_columns,
    resolve_result_number,
    result_cell_formatters,
    select_instructions,
    style_results_table,
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


def test_cells_are_formatted_as_python_does():
    # st.column_config.NumberColumn has no 'g' conversion and rendered these
    # as '0.00001234' and '12960000'; the default spec exists for exactly the
    # range of magnitudes a rate spans.
    assert format_result_cell(1.2345e-5) == '1.234e-05'
    assert format_result_cell(12960000.0) == '1.296e+07'
    assert format_result_cell(3.14159, '.2f') == '3.14'
    assert format_result_cell(3.14159, '.2f', 'mmol / h') == '3.14 mmol / h'


def test_unresolvable_cells_get_the_placeholder():
    assert format_result_cell(None) == MISSING_VALUE_PLACEHOLDER
    assert format_result_cell(np.nan) == MISSING_VALUE_PLACEHOLDER


def test_string_results_pass_through_unformatted():
    assert format_result_cell('not measured', '.2f') == 'not measured'


def test_every_result_column_gets_a_formatter():
    formatters = result_cell_formatters(INSTRUCTIONS)

    assert set(formatters) == {MAX_RATE_COLUMN, error_column_name(MAX_RATE_COLUMN),
                               QUANTUM_YIELD_COLUMN, error_column_name(QUANTUM_YIELD_COLUMN)}


def test_styling_keeps_the_numbers_and_formats_the_display(experiments):
    table = build_results_table(['Exp_001'], experiments, INSTRUCTIONS)

    styled = style_results_table(table, INSTRUCTIONS)

    # The frame the sort runs against is untouched...
    assert styled.data.loc['Exp_001', MAX_RATE_COLUMN] == 12.345
    # ...while the display carries the instruction's own format spec.
    assert '12.35' in styled.to_html()


def test_styling_ignores_columns_the_table_does_not_have(experiments):
    table = build_results_table(['Exp_001'], experiments,
                                select_instructions(INSTRUCTIONS, [QUANTUM_YIELD_COLUMN]))

    assert '9' in style_results_table(table, INSTRUCTIONS).to_html()


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
