"""
Drive a long processing run one experiment per Streamlit rerun.

The pyKES app is also deployed as a static browser page via stlite, which runs
Streamlit inside Pyodide on the browser's *single* event loop — there is no
separate script-runner thread. A loop that processes every experiment inside
one script run therefore never lets Streamlit deliver anything: the progress
messages it writes are queued and only flushed once the run yields to the
event loop, which happens after the loop has already finished. The progress
bar was invisible in the browser for exactly this reason.

A job started here is instead advanced by one experiment per rerun. Each rerun
ends, the event loop gets control, and the bar painted by that rerun reaches
the screen.

The reruns are app-scoped, so the page that hosts a job has to skip its
expensive sections while `any_active_job` is true — the data-upload page
rewrites the whole HDF5 file in its download section, which must not happen
once per experiment. Fragment-scoped reruns would avoid that, but they are not
usable here: ``st.rerun(scope="fragment")`` is rejected outside a fragment
rerun, and the only way into one is a widget interaction or a ``run_every``
timer that would then poll for the entire duration of the job.

The same code path is used when running under ``streamlit run``; nothing here
is browser-specific.

See [docs/browser_deployment.md](../../../docs/browser_deployment.md).
"""

import shutil
from typing import Callable, Optional

import streamlit as st

from pyKES.database.data_processing import finalize_processing_run


# Session-state key holding the results of the most recently finished job,
# derived from the job's own key. The results outlive the job itself so the page
# can report them after the final rerun has torn the job down.
JOB_RESULTS_KEY_TEMPLATE = "{job_key}_results"


def job_results_key(job_key: str) -> str:
    """
    Session-state key under which a finished job parks its results.

    Parameters
    ----------
    job_key : str
        Key identifying the job.

    Returns
    -------
    str
        Session-state key of the results list.
    """

    return JOB_RESULTS_KEY_TEMPLATE.format(job_key=job_key)


def start_chunked_job(job_key: str,
                      experiment_names: list,
                      context: dict,
                      staging_directory: Optional[str] = None) -> None:
    """
    Register a processing run to be advanced one experiment per rerun.

    The caller is expected to trigger a full ``st.rerun`` afterwards, so the
    page re-renders with the job active.

    Parameters
    ----------
    job_key : str
        Session-state key identifying this job. One per page section.
    experiment_names : list of str
        Experiments to process, in order.
    context : dict
        Keyword arguments handed to the step function alongside the experiment
        name — the processing callables, the staging directory and the
        already-resolved external version. Must include the
        `ExperimentalDataset` under ``'database'``: both step functions take it,
        and the job is stamped against it when it finishes.
    staging_directory : str, optional
        Directory holding uploaded files for the duration of the job. Removed
        when the job finishes.

    Returns
    -------
    None : None
    """

    st.session_state.pop(job_results_key(job_key), None)

    st.session_state[job_key] = {
        'experiment_names': experiment_names,
        'completed': 0,
        'results': [],
        'painted': False,
        'staging_directory': staging_directory,
        'context': context,
    }


def active_job(job_key: str) -> Optional[dict]:
    """
    Return the job registered under a key, if there is one.

    Parameters
    ----------
    job_key : str
        Session-state key identifying the job.

    Returns
    -------
    dict or None
        The job state, or None when no job is running.
    """

    return st.session_state.get(job_key)


def collect_job_results(job_key: str) -> Optional[list]:
    """
    Take the results parked by the most recently finished job.

    Parameters
    ----------
    job_key : str
        Session-state key identifying the job.

    Returns
    -------
    list of dict or None
        The results, or None when no job has finished since the last call.
    """

    return st.session_state.pop(job_results_key(job_key), None)


def paint_job_progress(job: dict) -> None:
    """
    Draw the progress bar for the experiment that is about to be processed.

    Painting is a rerun of its own, doing no work, so that the bar reaches the
    screen *before* the next experiment occupies the event loop. Painting and
    processing in the same rerun would leave the bar one experiment behind and
    never show the first one — which is the whole bug this module exists for.

    Parameters
    ----------
    job : dict
        Job state; ``painted`` is set in place.

    Returns
    -------
    None : None
    """

    completed = job['completed']
    total = len(job['experiment_names'])

    st.progress(
        completed / total,
        text = f"Processing experiment {completed + 1}/{total}: {job['experiment_names'][completed]}",
    )

    job['painted'] = True


def run_job_step(job: dict, step_function: Callable[..., dict]) -> None:
    """
    Process the next experiment of a job.

    Parameters
    ----------
    job : dict
        Job state; ``results``, ``completed`` and ``painted`` are updated in
        place.
    step_function : callable
        ``(experiment_name, **context) -> result_dict``, e.g.
        `pyKES.database.data_processing.ingest_experiment`.

    Returns
    -------
    None : None
    """

    experiment_name = job['experiment_names'][job['completed']]

    job['results'].append(step_function(experiment_name, **job['context']))
    job['completed'] += 1
    job['painted'] = False


def finish_chunked_job(job_key: str, job: dict) -> None:
    """
    Stamp the dataset, clean up the staging directory and retire the job.

    Parameters
    ----------
    job_key : str
        Session-state key identifying the job.
    job : dict
        Job state to retire.

    Returns
    -------
    None : None
    """

    # Stamped once for the whole run, as the non-chunked loop functions do,
    # rather than after every experiment.
    finalize_processing_run(job['context']['database'],
                            job['results'],
                            job['context'].get('external_version'))

    if job['staging_directory'] is not None:
        shutil.rmtree(job['staging_directory'])

    st.session_state[job_results_key(job_key)] = job['results']
    del st.session_state[job_key]


def render_chunked_job(job_key: str, step_function: Callable[..., dict]) -> None:
    """
    Advance a job by one phase and rerun for the next.

    Alternates painting and processing, so every experiment is announced
    before it runs and the announcement reaches the screen while it runs.

    Parameters
    ----------
    job_key : str
        Session-state key identifying the job.
    step_function : callable
        ``(experiment_name, **context) -> result_dict``.

    Returns
    -------
    None : None
    """

    job = active_job(job_key)

    if job is None:
        return

    # st.rerun raises, so each branch below ends the script run.
    if job['completed'] == len(job['experiment_names']):
        finish_chunked_job(job_key, job)
        st.rerun()

    if job['painted']:
        run_job_step(job, step_function)
    else:
        paint_job_progress(job)

    st.rerun()


def any_active_job(job_keys: list) -> bool:
    """
    Report whether any of the given jobs is currently running.

    Pages use this to skip sections that are too expensive to re-render on
    every step of a job.

    Parameters
    ----------
    job_keys : list of str
        Session-state keys of the jobs the page can start.

    Returns
    -------
    bool
        True while at least one of them is active.
    """

    return any(active_job(job_key) is not None for job_key in job_keys)
