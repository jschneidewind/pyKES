"""
Build the finished HDF5 file shipped with the example data.

Runs the example app's own pipeline over the example metadata sheet and raw
files, outside Streamlit, and writes the result next to them. Re-run it after
changing the processing functions so the shipped dataset keeps matching the
code that produced it::

    python examples/example_data/build_example_dataset.py

The app's own upload page does the same work one experiment per rerun; this
is the batch path, and `read_in_experiments_single_threaded` is what both
share.
"""

import sys
from pathlib import Path

import pandas as pd

EXAMPLE_DATA = Path(__file__).resolve().parent
APP = EXAMPLE_DATA.parent / 'external_repo'

# The app's modules import each other by bare name, the way Streamlit runs
# them with the entry script's directory on the path.
sys.path.insert(0, str(APP))

from pyKES.database.data_processing import read_in_experiments_single_threaded  # noqa: E402
from pyKES.database.database_experiments import (ExperimentalDataset,  # noqa: E402
                                                 import_overview_excel)

from config import DATA_UPLOAD_CONFIG, EXTERNAL_VERSION  # noqa: E402

OVERVIEW_SHEET = EXAMPLE_DATA / 'metadata' / 'example_overview.xlsx'
RAW_DATA = EXAMPLE_DATA / 'raw_data'
OUTPUT = EXAMPLE_DATA / 'example_dataset.h5'


def report_progress(completed, total, experiment_name):
    """Print one line per finished experiment."""
    if experiment_name is not None:
        print(f"  [{completed}/{total}] {experiment_name}")


def build_dataset() -> ExperimentalDataset:
    """
    Ingest every experiment of the example sheet.

    Returns
    -------
    ExperimentalDataset
        Dataset holding all six experiments, processed.
    """

    handler = DATA_UPLOAD_CONFIG.file_handlers[0]

    dataset = ExperimentalDataset(
        group_mapping=DATA_UPLOAD_CONFIG.group_mapping,
        plotting_instruction=DATA_UPLOAD_CONFIG.plotting_instruction,
        processing_parameters=DATA_UPLOAD_CONFIG.processing_parameters,
    )

    # 'Processed' is compared as text, so it has to be read as text.
    dataset.update_overview_df(
        import_overview_excel(OVERVIEW_SHEET, 'Sheet1', dtype={'Processed': str}),
        DATA_UPLOAD_CONFIG.metadata_excel_experiment_column)

    dataset.set_external_version(EXTERNAL_VERSION)

    results = read_in_experiments_single_threaded(
        database=dataset,
        metadata_retrival_function=handler.metadata_retrival_function,
        raw_data_reading_function=handler.raw_data_reading_function,
        processing_function=handler.processing_function,
        overview_df_experiment_column=handler.overview_df_experiment_column,
        directory=RAW_DATA,
        progress_callback=report_progress,
    )

    failures = [result for result in results if not result['success']]

    if failures:
        raise RuntimeError("\n".join(f"{failure['file']}: {failure['error']}"
                                     for failure in failures))

    return dataset


if __name__ == '__main__':
    print(f"Processing {OVERVIEW_SHEET.name} ...")
    dataset = build_dataset()

    dataset.save_to_hdf5(str(OUTPUT), compression='gzip', verbose=False)

    print(f"\nWrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
    print(dataset.describe_version())
    print(pd.DataFrame({
        'H2 max rate / umol/h': {name: experiment.processed_data['H2_max_rate_umol_h']
                                 for name, experiment in dataset.experiments.items()},
        'O2 max rate / umol/h': {name: experiment.processed_data['O2_max_rate_umol_h']
                                 for name, experiment in dataset.experiments.items()},
        'H2 AQY / %': {name: experiment.processed_data['H2_apparent_quantum_yield_percent']
                       for name, experiment in dataset.experiments.items()},
    }))
