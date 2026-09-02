"""
Drive a long processing run one experiment per Streamlit rerun.

The pyKES app is also deployed as a static browser page via stlite, which runs
Streamlit inside Pyodide on the browser's *single* event loop — there is no
separate script-runner thread. A loop that processes every experiment inside
one script run therefore never lets Streamlit deliver anything: the progress
messages it writes are queued and only flushed once the run yields to the
event loop, which happens after the loop has already finished. The progress
bar was invisible in the browser for exactly this reason.

A job started here is instead advanced by one experiment per run of an
``st.fragment(run_every=...)``. The timer that drives those runs lives in the
*frontend*: Streamlit sends it as an auto-rerun instruction, and the browser
asks for the next run. Each run therefore **ends normally** and its elements
are delivered before the next one starts.

Ending normally is the part that matters, and it is why `st.rerun` cannot be
used to drive this instead. A script run that ends in `st.rerun` is followed
immediately by the next one, and `AppSession` clears the browser queue on every
``SCRIPT_STARTED`` — so a progress bar drawn by a run that then reruns is
discarded before it is ever sent, unless a flush happens to fall between the
two. On a server it usually does, because the flush runs on another thread. In
the browser it does not, and the bar is never seen at all. Measured in headless
Chrome against stlite 1.7.3: with `st.rerun` the page went from blank straight
to the finished state; with the timer the bar advanced step by step.

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

# How often the frontend asks for the next step of a running job. It only paces
# the hand-over between steps — the step itself takes as long as it takes — so
# it is set well below the duration of any real experiment and costs one round
# trip per step.
JOB_STEP_INTERVAL_SECONDS = 0.1


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


def job_is_complete(job: dict) -> bool:
    """
    Report whether every experiment of a job has been processed.

    Parameters
    ----------
    job : dict
        Job state.

    Returns
    -------
    bool
        True once no experiment is left.
    """

    return job['completed'] == len(job['experiment_names'])


def paint_job_progress(job: dict) -> None:
    """
    Draw the progress bar for the experiment that is about to be processed.

    Called at the end of every step, so the run that delivers this bar is over
    before the experiment it names occupies the worker. The bar therefore
    stands on screen for exactly as long as that experiment takes.

    Parameters
    ----------
    job : dict
        Job state. Must not be complete — the caller finishes the job first, so
        ``completed`` always indexes an experiment here.

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


def run_job_step(job: dict, step_function: Callable[..., dict]) -> None:
    """
    Process the next experiment of a job.

    Parameters
    ----------
    job : dict
        Job state; ``results`` and ``completed`` are updated in place.
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


@st.fragment(run_every=JOB_STEP_INTERVAL_SECONDS)
def render_job_progress(job_key: str) -> None:
    """
    Draw the progress bar of a running job. Draws nothing else, ever.

    Deliberately a separate fragment from `advance_job`: a fragment rerun
    clears the deltas of the fragments *in that run* and preserves everyone
    else's, so a bar drawn here survives every step the worker takes. Drawn
    from inside the worker instead, it was wiped by the worker's own next tick
    before it was ever sent, and the browser showed the first experiment for
    the whole run.

    Parameters
    ----------
    job_key : str
        Session-state key identifying the job.

    Returns
    -------
    None : None
    """

    job = active_job(job_key)

    if job is None or job_is_complete(job):
        return

    paint_job_progress(job)


@st.fragment(run_every=JOB_STEP_INTERVAL_SECONDS)
def advance_job(job_key: str, step_function: Callable[..., dict]) -> None:
    """
    Process one experiment per timer tick. Writes nothing to the page.

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

    if job_is_complete(job):
        finish_chunked_job(job_key, job)

        # App-scoped: the page re-renders once without either fragment, so the
        # timers stop and the sections skipped during the job come back.
        st.rerun()

    run_job_step(job, step_function)


def render_chunked_job(job_key: str, step_function: Callable[..., dict]) -> None:
    """
    Run a job to completion, one experiment per timer tick, showing progress.

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

    render_job_progress(job_key)
    advance_job(job_key, step_function)


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
