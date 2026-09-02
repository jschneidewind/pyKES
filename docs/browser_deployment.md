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

## 2. The pattern: one experiment per timer tick

A script run has to **end** for what it drew to reach the browser. So instead of
looping inside one run, a long job is spread across many runs, one experiment
each.

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

From then on the page calls `render_chunked_job`, which is two `st.fragment`s
with a `run_every` timer:

| Fragment | What each tick does |
| --- | --- |
| `render_job_progress` | Draws the progress bar. Nothing else, ever. |
| `advance_job` | Processes one experiment, or retires a finished job. Draws nothing. |

Two details of that split are not obvious, and both were established by
measuring the real page in headless Chrome rather than by reading the API.

### Why `st.rerun` cannot drive this

The obvious implementation steps the job with `st.rerun()` at the end of each
run. It does not work in the browser at all, and the reason is in
`AppSession._on_scriptrunner_event`: every `SCRIPT_STARTED` calls
`_clear_queue()`, which throws away queued messages that have not been flushed.
A run that ends in `st.rerun` is followed immediately by the next one, so the
bar it drew is discarded before it is ever sent — unless a flush happens to fall
between the two.

On a server one usually does, because the flush runs on another thread. In the
browser it does not. Measured over an eight-experiment run: the page went from
blank straight to the finished state, with the progress bar appearing in **zero**
intermediate DOM states.

A `run_every` fragment does not have this problem: the timer lives in the
frontend, so the browser only asks for the next tick after it has received and
drawn the previous one.

### Why the bar is drawn by a different fragment than the work

`ForwardMsgQueue.clear` takes the ids of the fragments running this time and
preserves every delta that belongs to *other* fragments. A fragment therefore
wipes its own output on its own next tick, but never anyone else's.

So a bar drawn inside the worker fragment is cleared by the worker's next tick
before it is flushed. Measured: the browser showed
`Processing experiment 1/8` for the entire run and never advanced — the only
paint that survived was the very first one, which came from the initial page run
and so had no fragment id. Splitting the painter out fixed it: the same run then
advanced through several experiments.

### What this actually looks like

Honest expectation: the bar appears immediately and advances, but it does **not**
tick once per experiment. Python, Streamlit and the UI share one worker thread,
and the processing keeps it busy, so the painter only gets a turn between
experiments and only some of those paints reach the browser before the next one
replaces them. On an eight-experiment probe the user saw three or four distinct
states rather than eight.

That is a limit of the runtime, not a tuning problem — raising or lowering the
timer interval did not change the number of updates. What matters is that the
page is visibly working and the count climbs, instead of sitting blank for
minutes.

### The page still skips its expensive sections

`advance_job` retires a finished job with an app-scoped `st.rerun()`, so the
whole page re-renders once against the finished dataset. While the job runs, the
page skips its tail — the HDF5 merge, the download button (which serializes the
entire dataset), the overview table and the statistics:

```python
if any_active_job(_page_job_keys(config)):
    return
```

### Staging uploaded files

Raw-data files arrive as uploads and have to be on disk for the reader to find.
A `tempfile.TemporaryDirectory()` context cannot be used, because it would
delete the files at the end of the run that created them — long before the last
experiment is read. The page uses `tempfile.mkdtemp()` instead, and
`finish_chunked_job` removes the directory when the job is done. A user who
navigates away mid-job leaves one directory behind; in the browser that is
Pyodide's in-memory filesystem, which goes away with the tab.

## 3. The other browser constraint: the filesystem is the bundle

Under `streamlit run`, an app's own repository is right there on disk. In the
browser there is no repository — Pyodide's filesystem holds exactly the files
the deployment bundles into it, plus the packages micropip installs. Anything
pyKES reads from disk at runtime therefore has to be part of the bundle.

The one thing that is read this way is the app's version.
`get_project_version(__file__)` searches upwards from the calling module for
the nearest `pyproject.toml` and returns the version declared in it (see
[versioning_and_reprocessing.md](versioning_and_reprocessing.md)). A bundle of
`.py` files alone has no such file, the search reaches the filesystem root, and
every dataset created in the browser is stamped `'version': None` — silently,
since a missing provenance stamp is not an error the page can raise.

So the build script must bundle `pyproject.toml` alongside the app package. In
pyKES-Well-App's `deploy/build.py`:

```python
files["Home.py"] = (STREAMLIT_APP / "Home.py").read_text()
files["pyproject.toml"] = PYPROJECT.read_text()      # provenance of the app
```

The bundled layout then mirrors the repository closely enough for the search:
`pyproject.toml` sits beside the `pykes_well_app/` package, which is one level
above the `config.py` that calls `get_project_version(__file__)`.

To check it without a browser, build the bundle to a directory and read the
version back out of that layout:

```python
version = get_project_version(str(bundle / 'pykes_well_app' / 'config.py'))
```

It must equal the version declared in the repository's `pyproject.toml`, not
`None`.

## 4. Adding a new long-running section

Anything that takes noticeable time and is made of repeatable units should use
the same pattern:

1. Expose the unit of work as a module-level function taking
   `(experiment_name, **context)` — as `ingest_experiment` and
   `reprocess_experiment_by_name` do in `pyKES.database.data_processing`.
2. Start a job with `start_chunked_job` and `st.rerun()`.
3. Call `render_chunked_job(job_key, step_function)` on every run while the job
   is active, and `collect_job_results(job_key)` once it is gone.
4. Add the job's key to the page's `any_active_job` guard.

Never draw progress from inside the fragment that does the work, and never step
a job with `st.rerun()` — the two failure modes measured above.

Work that is *not* made of repeatable units — writing one large HDF5 file, for
instance — cannot be chunked this way, and will still block the browser page
with no feedback. Keep such operations off the path that runs on every rerun.

## 5. Checking it

`src/tests/test_chunked_processing.py` drives the whole sequence with
`streamlit.testing.v1.AppTest`, which runs a script in-process: one `run` per
timer tick, since `AppTest` does not fire the frontend timer. It asserts the
tick count, that the work and the painting live in different fragments, that
staged files survive, and that the finished dataset matches a plain
`read_in_experiments_single_threaded` call.

**None of that can tell you whether the bar is actually painted**, which is the
whole point — every failed design above passed its in-process tests. That needs
a browser, and it is worth re-checking after any change to this module.

The pyKES repository has no deployment of its own, so use an external app's
bundle, pointed at a locally built wheel rather than the released one:

```bash
uv build                                   # in the pyKES repo
cp dist/pykes-*.whl <bundle>/              # next to the app's index.html
# in the bundle's files.js, replace the "pykes==X.Y.Z" requirement with
#   "http://localhost:8000/pykes-X.Y.Z-py3-none-any.whl"
python -m http.server 8000                 # in the bundle directory
```

Then open it and watch. Polling the DOM from a driver script is not enough on
its own — install a `MutationObserver` before the app loads and read its log
afterwards, so that what the browser was shown is recorded rather than sampled:

```js
window.__log = [];
new MutationObserver(function () {
  window.__log.push({t: performance.now(), text: document.body.innerText});
}).observe(document, {childList: true, subtree: true, characterData: true});
```

The bar must appear while the job is running and the count must climb. A single
state that never changes means the painter is being cleared by whatever is doing
the work — see section 2.
