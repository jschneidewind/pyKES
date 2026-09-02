# Running the Streamlit pages in the browser

The pyKES Streamlit pages are used in two quite different runtimes:

1. **A normal server**, started with `streamlit run`. Python runs on a machine,
   the browser only draws what the server sends it.
2. **The browser itself**, via [stlite](https://github.com/whitphx/stlite),
   which embeds Python (Pyodide) in the page. External apps deploy this way to
   get a static site with no server behind it — pyKES-Well-App, for instance,
   ships one on GitHub Pages.

The second runtime has one constraint that shapes how a long-running page has
to be written, and getting it wrong produces a page that *looks* frozen. This
document explains the constraint, the pattern the data-upload page uses to work
within it, and what to check when adding a new long-running section.

---

## 1. The constraint: one event loop, no second thread

Under `streamlit run`, your script runs on its own thread. While it works, the
server's event loop keeps running on another thread, delivering the elements
your script has produced so far. A progress bar updated inside a loop therefore
reaches the screen while the loop is still going.

In the browser there is no second thread. Python, Streamlit's runtime and the
page all share the browser's single event loop. Anything a script writes is put
on a queue, and that queue is only drained when the script **yields control**
back to the event loop — which a plain Python loop never does until it is
finished.

So this, which works perfectly on a server:

```python
progress = st.empty()

for completed, experiment_name in enumerate(experiment_names, start=1):
    process(experiment_name)                                  # seconds of work
    progress.progress(completed / total, text=experiment_name)
```

shows the user *nothing at all* in the browser. Every `progress()` call is
queued; the queue is drained only once the loop has ended, at which point the
final state is drawn — and if the code then clears the placeholder, as the
data-upload page used to, not even that is seen. The page appears to do nothing
for minutes, and then the results simply appear.

The same constraint is why stlite makes `time.sleep()` a no-op and why
`st.spinner()` cannot show a spinner around a blocking call: nothing can be
drawn while the only event loop is occupied.

## 2. The pattern: one experiment per rerun

A script run ends at every rerun, and *that* is a yield point: the queue drains,
the browser paints, and the next run starts. So instead of looping inside one
run, a long job is spread across many runs, one experiment each.

`pyKES.streamlit_app.chunked_processing` implements this. The job lives in
`st.session_state` — the experiment names, how many are done, the results so
far, and the context the processing needs:

```python
start_chunked_job(
    job_key = "ingestion_job_raw_data",
    experiment_names = select_unprocessed_experiments(dataset),
    context = {'database': dataset, 'processing_function': ..., ...},
    staging_directory = staging_directory,
)
st.rerun()
```

From then on, every run calls `render_chunked_job`, which advances the job by
one phase and reruns:

| Phase | What the run does |
| --- | --- |
| paint | Draws the progress bar for the experiment *about to* run. No work. |
| work | Processes that one experiment. |
| finish | Stamps the dataset version, deletes the staging directory, parks the results. |

### Why painting is a phase of its own

Painting and processing in the same run would not help: the bar drawn at the
start of a run still only reaches the screen when that run ends — by which time
the experiment it announced is already done. The display would lag one
experiment behind and the first one would never be announced at all, which is
exactly the symptom this pattern was written to fix. A paint run does no work,
so it ends immediately and the announcement is on screen for the whole time the
following work run is busy.

The cost is two runs per experiment. Both are cheap next to the processing
itself.

### Why the reruns are app-scoped

`st.fragment` exists to rerun one part of a page in isolation, which would be a
natural fit. It cannot be used here: `st.rerun(scope="fragment")` is rejected
unless it is called *during a fragment rerun*, and the only ways into one are a
widget interaction or a `run_every` timer that would then poll for the whole
duration of the job. So the reruns are ordinary, app-scoped ones.

That means the whole page script re-runs once per phase, and any section that
is expensive to render would run with it. The data-upload page therefore skips
its tail — the HDF5 merge, the download button (which serializes the entire
dataset), the overview table and the statistics — while a job is running:

```python
if any_active_job(_page_job_keys(config)):
    return
```

The finish phase reruns app-scoped one last time, so those sections come back
and render once against the finished dataset.

### Staging uploaded files

Raw-data files arrive as uploads and have to be on disk for the reader to find.
A `tempfile.TemporaryDirectory()` context cannot be used, because it would
delete the files at the end of the run that created them — long before the last
experiment is read. The page uses `tempfile.mkdtemp()` instead, and
`finish_chunked_job` removes the directory when the job is done. A user who
navigates away mid-job leaves one directory behind; in the browser that is
Pyodide's in-memory filesystem, which goes away with the tab.

## 3. Adding a new long-running section

Anything that takes noticeable time and is made of repeatable units should use
the same pattern:

1. Expose the unit of work as a module-level function taking
   `(experiment_name, **context)` — as `ingest_experiment` and
   `reprocess_experiment_by_name` do in `pyKES.database.data_processing`.
2. Start a job with `start_chunked_job` and `st.rerun()`.
3. Call `render_chunked_job(job_key, step_function)` on every run while the job
   is active, and `collect_job_results(job_key)` once it is gone.
4. Add the job's key to the page's `any_active_job` guard.

Work that is *not* made of repeatable units — writing one large HDF5 file, for
instance — cannot be chunked this way, and will still block the browser page
with no feedback. Keep such operations off the path that runs on every rerun.

## 4. Checking it

The unit-level behaviour is covered by `src/tests/test_chunked_processing.py`,
which drives the whole sequence with `streamlit.testing.v1.AppTest` — it runs a
script in-process and follows `st.rerun`, so the run count, the order of the
phases and the survival of the staging directory can all be asserted without a
browser.

What those tests cannot show is the thing that started all this: whether the
bar is actually *painted* while the job runs. That needs the real runtime:

```bash
# server runtime
streamlit run src/pyKES/streamlit_app/Home.py

# browser runtime, against a locally built wheel
uv build
python -m http.server 8000   # in the external app's deploy/ directory
```

In both, the bar must advance experiment by experiment, naming the one in
flight, rather than appearing only when everything is finished.
