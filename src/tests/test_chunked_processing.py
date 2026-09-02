"""
Tests for the rerun-driven processing of `pyKES.streamlit_app.chunked_processing`.

The driver exists so the progress bar stays visible in the stlite browser
build, where a blocking loop occupies the only event loop and lets nothing
reach the screen until it has finished. What has to hold is that the run is
spread over several script runs, that exactly one experiment is processed per
pair of them, and that the dataset ends up as it would after a single
`read_in_experiments_single_threaded` call.

`AppTest` runs a Streamlit script in-process and follows `st.rerun`, so the
whole sequence can be driven without a browser.
"""

from pathlib import Path

import pandas as pd
import streamlit
from streamlit.testing.v1 import AppTest

from pyKES.database.data_processing import read_in_experiments_single_threaded
from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.streamlit_app.chunked_processing import paint_job_progress


EXPERIMENT_NAMES = ['Exp_001', 'Exp_002']

# One paint run and one work run per experiment, plus the run that retires the
# job and the final run that renders the finished page.
EXPECTED_SCRIPT_RUNS = 2 * len(EXPERIMENT_NAMES) + 2

# Streamlit script exercising one ingestion job from start to finish. The
# processing callables are defined inside it because AppTest executes the
# string as the app's main script.
CHUNKED_INGESTION_APP = '''
import pandas as pd
import streamlit as st

from pyKES.database.data_processing import ingest_experiment, select_unprocessed_experiments
from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.streamlit_app.chunked_processing import (any_active_job, collect_job_results,
                                                    render_chunked_job, start_chunked_job)

JOB_KEY = "test_ingestion_job"
EXPERIMENT_NAMES = ["Exp_001", "Exp_002"]


def retrieve_metadata(experiment_name, overview_df):
    return {"experiment_name": experiment_name}


def read_raw_data(directory, metadata_dict):
    return {"signal": [1.0, 2.0, 3.0]}


def process_raw_data(raw_data_dict, metadata_dict):
    st.session_state.setdefault("processing_runs", [])
    st.session_state.processing_runs.append(st.session_state.script_runs)

    return {"maximum": max(raw_data_dict["signal"])}


st.session_state.setdefault("script_runs", 0)
st.session_state.script_runs += 1

if "dataset" not in st.session_state:
    dataset = ExperimentalDataset(overview_df=pd.DataFrame({"Experiment": EXPERIMENT_NAMES}))
    st.session_state.dataset = dataset

    start_chunked_job(
        job_key=JOB_KEY,
        experiment_names=select_unprocessed_experiments(dataset),
        context={
            "database": dataset,
            "metadata_retrival_function": retrieve_metadata,
            "raw_data_reading_function": read_raw_data,
            "processing_function": process_raw_data,
        },
    )

render_chunked_job(JOB_KEY, ingest_experiment)

results = collect_job_results(JOB_KEY)

if results is not None:
    st.session_state.results = results

# Stands in for the expensive page sections that must not be re-rendered once
# per experiment.
if not any_active_job([JOB_KEY]):
    st.session_state.setdefault("expensive_renders", 0)
    st.session_state.expensive_renders += 1
    st.write("expensive section")
'''


# Streamlit script exercising the staging of uploaded files. The raw-data
# reader reads the staged file back, so the run only succeeds if the staging
# directory outlives the script run that created it — which is the point of
# not using a TemporaryDirectory context there.
STAGED_UPLOAD_APP = '''
import pandas as pd
import streamlit as st

from pyKES.database.data_processing import ingest_experiment, select_unprocessed_experiments
from pyKES.database.database_experiments import ExperimentalDataset
from pyKES.streamlit_app.chunked_processing import (active_job, collect_job_results,
                                                    render_chunked_job)
from pyKES.streamlit_app.components.data_upload_component import (INGESTION_JOB_KEY_TEMPLATE,
                                                                  _start_ingestion_job)
from pyKES.streamlit_app.config_interface import FileUploadHandler

EXPERIMENT_NAMES = ["Exp_001", "Exp_002"]
JOB_KEY = INGESTION_JOB_KEY_TEMPLATE.format(file_storage_key="raw_data")


class FakeUploadedFile:
    """Stand-in for a Streamlit UploadedFile."""

    def __init__(self, name, content):
        self.name = name
        self.content = content

    def getbuffer(self):
        return memoryview(self.content)


def retrieve_metadata(experiment_name, overview_df):
    return {"experiment_name": experiment_name}


def read_raw_data(directory, metadata_dict):
    staged_file = directory / (metadata_dict["experiment_name"] + ".csv")

    return {"signal": [float(value) for value in staged_file.read_text().split(",")]}


def process_raw_data(raw_data_dict, metadata_dict):
    return {"maximum": max(raw_data_dict["signal"])}


HANDLER = FileUploadHandler(
    label="Raw Data",
    file_type="csv",
    file_storage_key="raw_data",
    overview_df_experiment_column="Experiment",
    metadata_retrival_function=retrieve_metadata,
    raw_data_reading_function=read_raw_data,
    processing_function=process_raw_data,
)

if "dataset" not in st.session_state:
    dataset = ExperimentalDataset(overview_df=pd.DataFrame({"Experiment": EXPERIMENT_NAMES}))
    st.session_state.dataset = dataset

    _start_ingestion_job(
        job_key=JOB_KEY,
        config=HANDLER,
        dataset=dataset,
        uploaded_files=[FakeUploadedFile(name + ".csv", b"1.0,2.0,3.0")
                        for name in EXPERIMENT_NAMES],
        external_version={"app": "test"},
    )

job = active_job(JOB_KEY)

if job is not None:
    st.session_state.staging_directory = job["staging_directory"]

render_chunked_job(JOB_KEY, ingest_experiment)

results = collect_job_results(JOB_KEY)

if results is not None:
    st.session_state.results = results
'''


# Streamlit script rendering the real data-upload page. Reprocessing is the
# flow that needs no file upload, so it can be driven all the way through the
# component by clicking its submit button.
DATA_UPLOAD_PAGE_APP = '''
import pandas as pd
import streamlit as st

from pyKES.database.database_experiments import Experiment, ExperimentalDataset
from pyKES.streamlit_app.components.data_upload_component import render_data_upload
from pyKES.streamlit_app.config_interface import DataUploadConfig, FileUploadHandler

EXPERIMENT_NAMES = ["Exp_001", "Exp_002"]


def retrieve_metadata(experiment_name, overview_df):
    return {"experiment_name": experiment_name}


def read_raw_data(directory, metadata_dict):
    return {"signal": [1.0, 2.0]}


def process_raw_data(raw_data_dict, metadata_dict):
    st.session_state.setdefault("processing_runs", [])
    st.session_state.processing_runs.append(st.session_state.script_runs)

    return {"maximum": max(raw_data_dict["signal"])}


CONFIG = DataUploadConfig(file_handlers=[FileUploadHandler(
    label="Raw Data",
    file_type="csv",
    overview_df_experiment_column="Experiment",
    metadata_retrival_function=retrieve_metadata,
    raw_data_reading_function=read_raw_data,
    processing_function=process_raw_data,
)])

st.session_state.setdefault("script_runs", 0)
st.session_state.script_runs += 1
st.session_state.setdefault("hdf5_filename", None)

if "experimental_dataset" not in st.session_state:
    dataset = ExperimentalDataset(overview_df=pd.DataFrame(
        {"Experiment": EXPERIMENT_NAMES, "Processed": ["True", "True"]}))

    for experiment_name in EXPERIMENT_NAMES:
        dataset.add_experiment(Experiment(
            experiment_name=experiment_name,
            raw_data_file=experiment_name,
            color="black",
            group="default",
            metadata={"experiment_name": experiment_name},
            raw_data={"signal": [1.0, 2.0]},
            processed_data={"maximum": 2.0},
        ))

    st.session_state.experimental_dataset = dataset

render_data_upload(CONFIG)
'''


def run_chunked_ingestion_app():
    """
    Run the ingestion app to completion.

    Returns
    -------
    AppTest
        The finished app, with the dataset and the job bookkeeping in its
        session state.
    """

    app = AppTest.from_string(CHUNKED_INGESTION_APP)
    app.run(timeout=30)

    assert [element.value for element in app.exception] == []

    return app


def test_ingestion_is_spread_over_one_experiment_per_pair_of_runs():
    app = run_chunked_ingestion_app()

    assert app.session_state["script_runs"] == EXPECTED_SCRIPT_RUNS
    assert sorted(app.session_state["dataset"].experiments) == EXPERIMENT_NAMES
    assert [result['success'] for result in app.session_state["results"]] == [True, True]


def test_expensive_sections_are_skipped_while_the_job_runs():
    app = run_chunked_ingestion_app()

    # Only the final run, once the job has been retired.
    assert app.session_state["expensive_renders"] == 1
    assert [element.value for element in app.markdown] == ["expensive section"]


def test_each_experiment_is_announced_before_it_is_processed():
    app = run_chunked_ingestion_app()

    # Experiment N is processed on script run 2N, so the run before it painted
    # the bar and did nothing else. That is what puts the announcement on
    # screen before the experiment occupies the event loop; painting and
    # processing in the same run would leave the bar one experiment behind.
    assert app.session_state["processing_runs"] == [2, 4]


def test_progress_announces_the_experiment_about_to_run(monkeypatch):
    progress_calls = []

    monkeypatch.setattr(streamlit, "progress",
                        lambda value, text: progress_calls.append((value, text)))

    job = {'experiment_names': EXPERIMENT_NAMES, 'completed': 1, 'painted': False}
    paint_job_progress(job)

    fraction, text = progress_calls[0]

    assert fraction == 0.5
    assert 'Exp_002' in text
    assert '2/2' in text
    assert job['painted'] is True


def test_staged_uploads_survive_until_the_job_finishes():
    app = AppTest.from_string(STAGED_UPLOAD_APP)
    app.run(timeout=30)

    assert [element.value for element in app.exception] == []

    # The reader read each staged file back on a later script run than the one
    # that wrote it, which a TemporaryDirectory context would not have allowed.
    assert [result['success'] for result in app.session_state["results"]] == [True, True]
    assert sorted(app.session_state["dataset"].experiments) == EXPERIMENT_NAMES

    # The flag stays text across the whole chunked run. A bool column here
    # would raise `TypeError: Invalid value 'True' for dtype 'bool'` on the
    # first success, the bug fixed in 0.2.0 for the loop function.
    processed_flags = app.session_state["dataset"].overview_df['Processed']
    assert processed_flags.tolist() == ['True', 'True']

    assert not Path(app.session_state["staging_directory"]).exists()


def test_reprocessing_runs_chunked_through_the_page():
    app = AppTest.from_string(DATA_UPLOAD_PAGE_APP)
    app.run(timeout=60)

    assert [element.value for element in app.exception] == []

    reprocess_button = next(button for button in app.button if "Reprocess" in button.label)
    reprocess_button.click()
    app.run(timeout=60)

    assert [element.value for element in app.exception] == []

    # Each experiment reprocessed on its own run, one run after the paint run
    # that announced it.
    processing_runs = app.session_state["processing_runs"]
    assert len(processing_runs) == len(EXPERIMENT_NAMES)
    assert processing_runs == sorted(set(processing_runs))
    assert all(later - earlier == 2
               for earlier, later in zip(processing_runs, processing_runs[1:]))

    assert [element.value for element in app.success] == [
        "✓ Reprocessed 2 experiment(s) successfully"]

    # The sections skipped during the run are back on the finished page.
    assert any("Download Dataset" in element.value for element in app.subheader)


def test_chunked_run_matches_the_loop_function():
    app = run_chunked_ingestion_app()

    looped_dataset = ExperimentalDataset(
        overview_df = pd.DataFrame({'Experiment': EXPERIMENT_NAMES}))

    read_in_experiments_single_threaded(
        database = looped_dataset,
        metadata_retrival_function = lambda name, overview_df: {'experiment_name': name},
        raw_data_reading_function = lambda directory, metadata: {'signal': [1.0, 2.0, 3.0]},
        processing_function = lambda raw_data, metadata: {'maximum': max(raw_data['signal'])},
    )

    chunked_dataset = app.session_state["dataset"]

    assert sorted(chunked_dataset.experiments) == sorted(looped_dataset.experiments)
    assert (chunked_dataset.overview_df['Processed'].tolist()
            == looped_dataset.overview_df['Processed'].tolist())
    assert chunked_dataset.version['last_processed'] is not None
