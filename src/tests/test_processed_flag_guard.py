"""
Tests for refusing to write a file whose results no longer follow from it.

Editing metadata without reprocessing left a dataset holding ``processed_data``
derived from values the file no longer contained — a contradiction nothing
downstream can detect, because the numbers look like every other result. A
save now refuses, and the Data Upload page withholds the download.

The other half of the guarantee is that the flag is actually cleared whenever
metadata changes under stored results. There are three such paths — the editor
grid, an uploaded workbook, and a repair of a file that was written before the
overview sheet owned the metadata — and this module pins all three, since the
refusal is only as good as the flag it reads.
"""

import numpy as np
import pandas as pd
import pytest

from pyKES.database.data_processing import mark_experiment_processed
from pyKES.database.database_experiments import (METADATA_LOADING_KEY, METADATA_PROCESSING_KEY,
                                                 Experiment, ExperimentalDataset,
                                                 describe_experiment_names)
from .test_metadata_consistency import corrupt_stored_metadata
from .test_metadata_editor_page import (SHEET, build_workbook, edit_cell, run_page,
                                        upload_workbook)

IRRADIANCE = 'Irradiance [mW/cm2]'


def build_experiment(name, irradiance=40.0):
    """An experiment whose stored result was computed from its irradiance."""
    return Experiment(experiment_name=name,
                      raw_data_file=f'{name}.csv',
                      color='blue',
                      group='Intensity',
                      metadata={'Experiment': name, IRRADIANCE: irradiance},
                      raw_data={'signal': np.array([1.0, 2.0])},
                      processed_data={'rate_per_irradiance': 2.0 / irradiance})


@pytest.fixture
def dataset():
    """One processed experiment, and one row that has not been ingested."""
    dataset = ExperimentalDataset(
        overview_df=pd.DataFrame({'Experiment': ['Exp_001', 'Exp_002'],
                                  IRRADIANCE: [40.0, 80.0],
                                  'Comment': ['first', 'second'],
                                  'Processed': ['True', 'False']}),
        processing_parameters={METADATA_LOADING_KEY: ['Experiment'],
                               METADATA_PROCESSING_KEY: [IRRADIANCE]})
    dataset.add_experiment(build_experiment('Exp_001'))

    return dataset


def edit_overview(dataset, column, value, experiment_name='Exp_001'):
    """Change one overview cell and push it into the experiment, as the editor does."""
    rows = dataset.overview_df['Experiment'].eq(experiment_name)
    dataset.overview_df.loc[rows, column] = value
    dataset.synchronize_experiment_metadata()


# =============================================================================
# What a save refuses
# =============================================================================

def test_a_dataset_whose_results_are_current_saves(dataset, tmp_path):
    dataset.save_to_hdf5(str(tmp_path / 'current.h5'), verbose=False)

    assert (tmp_path / 'current.h5').exists()


def test_metadata_changed_without_reprocessing_refuses_to_save(dataset, tmp_path):
    edit_overview(dataset, IRRADIANCE, 99.0)
    dataset.overview_df.loc[0, 'Processed'] = 'False'

    with pytest.raises(ValueError, match="no longer follow from their metadata"):
        dataset.save_to_hdf5(str(tmp_path / 'stale.h5'), verbose=False)

    assert not (tmp_path / 'stale.h5').exists()


def test_reprocessing_clears_the_refusal(dataset, tmp_path):
    edit_overview(dataset, IRRADIANCE, 99.0)
    dataset.overview_df.loc[0, 'Processed'] = 'False'

    mark_experiment_processed(dataset, 'Exp_001')
    dataset.save_to_hdf5(str(tmp_path / 'reprocessed.h5'), verbose=False)

    assert (tmp_path / 'reprocessed.h5').exists()


def test_a_row_that_was_never_ingested_does_not_block_a_save(dataset, tmp_path):
    # Processing a sheet a few experiments at a time is a normal way to work,
    # and a row with no experiment behind it has no results to be stale.
    assert dataset.overview_df.loc[1, 'Processed'] == 'False'

    dataset.save_to_hdf5(str(tmp_path / 'partial.h5'), verbose=False)

    assert dataset.stale_processed_experiments() == []


def test_a_sheet_that_does_not_track_the_flag_makes_no_claim(tmp_path):
    # Datasets built without the pipeline — `usage_example`, a notebook — carry
    # no flag column, and nothing has been tracking their processing state.
    untracked = ExperimentalDataset(
        overview_df=pd.DataFrame({'Experiment': ['Exp_001'], IRRADIANCE: [40.0]}))
    untracked.add_experiment(build_experiment('Exp_001'))

    untracked.save_to_hdf5(str(tmp_path / 'untracked.h5'), verbose=False)

    assert untracked.stale_processed_experiments() == []


def test_the_refusal_can_be_overridden_for_a_file_that_is_not_delivered(dataset, tmp_path):
    # The page stages the loaded dataset in a temporary file to merge it;
    # refusing that would make a pending reprocessing run block merging.
    edit_overview(dataset, IRRADIANCE, 99.0)
    dataset.overview_df.loc[0, 'Processed'] = 'False'

    dataset.save_to_hdf5(str(tmp_path / 'staged.h5'), verbose=False,
                         allow_stale_processed_data=True)

    assert (tmp_path / 'staged.h5').exists()


def test_the_refusal_names_the_experiments(dataset, tmp_path):
    dataset.add_experiment(build_experiment('Exp_002', irradiance=80.0))
    dataset.overview_df['Processed'] = ['False', 'False']

    with pytest.raises(ValueError, match="Exp_001, Exp_002"):
        dataset.save_to_hdf5(str(tmp_path / 'stale.h5'), verbose=False)


def test_a_long_list_of_names_is_summarised():
    described = describe_experiment_names([f'Exp_{index:03d}' for index in range(9)])

    assert described == 'Exp_000, Exp_001, Exp_002, Exp_003, Exp_004, and 4 more'


# =============================================================================
# A repaired file: metadata changed with nothing editing anything
# =============================================================================

def test_a_repaired_file_is_flagged_for_reprocessing(dataset, tmp_path):
    path = str(tmp_path / 'diverged.h5')
    dataset.save_to_hdf5(path, verbose=False)
    corrupt_stored_metadata(path, 'Exp_001', IRRADIANCE, 999.0)

    repaired = ExperimentalDataset.load_from_hdf5(path)

    # Its result was computed from 999.0, which the load has just replaced.
    assert repaired.stale_processed_experiments() == ['Exp_001']

    with pytest.raises(ValueError, match="no longer follow from their metadata"):
        repaired.save_to_hdf5(str(tmp_path / 'roundtrip.h5'), verbose=False)


def test_a_repair_of_a_column_processing_ignores_costs_nothing(dataset, tmp_path):
    path = str(tmp_path / 'diverged_comment.h5')
    dataset.save_to_hdf5(path, verbose=False)
    corrupt_stored_metadata(path, 'Exp_001', 'Comment', 'drifted')

    repaired = ExperimentalDataset.load_from_hdf5(path)

    assert repaired.metadata_repair_report != {}
    assert repaired.stale_processed_experiments() == []
    repaired.save_to_hdf5(str(tmp_path / 'roundtrip.h5'), verbose=False)


def test_a_dataset_declaring_nothing_treats_every_repair_as_invalidating(tmp_path):
    # How datasets behaved before the declarations existed: without them the
    # dataset cannot say which columns its results rest on.
    undeclared = ExperimentalDataset(
        overview_df=pd.DataFrame({'Experiment': ['Exp_001'], 'Comment': ['first'],
                                  'Processed': ['True']}))
    experiment = build_experiment('Exp_001')
    experiment.metadata['Comment'] = 'first'
    undeclared.add_experiment(experiment)

    path = str(tmp_path / 'undeclared.h5')
    undeclared.save_to_hdf5(path, verbose=False)
    corrupt_stored_metadata(path, 'Exp_001', 'Comment', 'drifted')

    assert ExperimentalDataset.load_from_hdf5(path).stale_processed_experiments() == ['Exp_001']


# =============================================================================
# What clears the flag: the editor, and an uploaded workbook
# =============================================================================

def flags(app):
    """The processed flag of every row, by experiment."""
    overview_df = app.session_state['experimental_dataset'].overview_df

    return overview_df.set_index('Experiment')['Processed'].to_dict()


def test_editing_a_processing_column_clears_the_flag():
    assert flags(edit_cell(run_page(), 0, IRRADIANCE, 55.0)) == {'Exp_001': 'False',
                                                                 'Exp_002': 'True'}


def test_editing_a_column_processing_ignores_leaves_the_flag_alone():
    assert flags(edit_cell(run_page(), 0, 'Comment', 'revised')) == {'Exp_001': 'True',
                                                                     'Exp_002': 'True'}


def test_an_uploaded_sheet_clears_the_flag_for_a_changed_processing_column():
    app = upload_workbook(run_page(), SHEET.assign(**{IRRADIANCE: [55, 80]}))

    assert flags(app) == {'Exp_001': 'False', 'Exp_002': 'True'}


def test_an_uploaded_sheet_clears_the_flag_for_a_changed_filename():
    # A filename cannot be edited in the grid — changing it means the raw data
    # behind the results is a different measurement — but a workbook can carry
    # a corrected one, and then the results do not follow from it either.
    app = upload_workbook(run_page(), SHEET.assign(**{'File name O2': ['other.csv', 'two.csv']}))

    assert flags(app) == {'Exp_001': 'False', 'Exp_002': 'True'}


def test_an_uploaded_sheet_leaves_the_flag_alone_for_a_column_processing_ignores():
    app = upload_workbook(run_page(), SHEET.assign(Comment=['revised', 'second']))

    assert flags(app) == {'Exp_001': 'True', 'Exp_002': 'True'}


def test_uploading_an_unchanged_sheet_costs_no_reprocessing():
    assert flags(upload_workbook(run_page())) == {'Exp_001': 'True', 'Exp_002': 'True'}


def test_a_dataset_declaring_nothing_is_invalidated_by_any_sheet_change():
    app = upload_workbook(run_page(declarations='{}'),
                          SHEET.assign(Comment=['revised', 'second']))

    assert flags(app) == {'Exp_001': 'False', 'Exp_002': 'True'}


def test_a_row_the_sheet_adds_is_flagged_unprocessed_not_blank():
    # Left to the concat the flag came back as NaN, which the selectors read as
    # unprocessed only by accident and a save has to interpret.
    grown = pd.concat([SHEET, pd.DataFrame([{'Experiment': 'Exp_003',
                                             'File name O2': 'three.csv',
                                             IRRADIANCE: 120, 'Comment': 'third'}])],
                      ignore_index=True)

    assert flags(upload_workbook(run_page(), grown)) == {'Exp_001': 'True', 'Exp_002': 'True',
                                                         'Exp_003': 'False'}


# =============================================================================
# The page withholds the download
# =============================================================================

def download_button(app):
    """The download button, or None when the page is not offering one."""
    return next((element for element in app.get('download_button')
                 if 'Download HDF5' in element.label), None)


def test_the_page_offers_the_download_while_the_results_are_current():
    assert download_button(run_page()) is not None


def test_the_page_withholds_the_download_after_a_metadata_edit():
    # Serialized while the page renders, since `st.download_button` needs the
    # bytes up front — so the refusal has to happen here rather than reaching
    # the user as a traceback over the rest of the page.
    app = edit_cell(run_page(), 0, IRRADIANCE, 55.0)

    assert [element.value for element in app.exception] == []
    assert download_button(app) is None
    assert any('Download withheld' in element.value and 'Exp_001' in element.value
               for element in app.warning)


def test_the_page_offers_the_download_again_once_reprocessed():
    app = edit_cell(run_page(), 0, IRRADIANCE, 55.0)

    mark_experiment_processed(app.session_state['experimental_dataset'], 'Exp_001')
    app.run(timeout=60)

    assert download_button(app) is not None
