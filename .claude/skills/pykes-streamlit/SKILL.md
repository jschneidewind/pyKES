---
name: pykes-streamlit
description: "Streamlit work in the pyKES repository: the reusable component library that external repos embed and configure, and the stlite browser runtime those repos deploy it into. Use for ANY edit under src/pyKES/streamlit_app/, and before adding a page, component, or long-running operation to a pyKES-derived app. Triggers: streamlit, st., streamlit_app, component, page, config_interface, chunked_processing, stlite, Pyodide, progress bar, rerun, session state, fragment, data upload, time series."
---

# Streamlit in pyKES

Two constraints in this repository are not obvious from the Streamlit API, and both
have already caused bugs that looked fine under `streamlit run` and failed in
production. Read this before touching `src/pyKES/streamlit_app/`.

Streamlit ships its own general skill, `developing-with-streamlit`, inside the
installed package (`streamlit/.agents/skills/`). Use that for the API itself —
widgets, layout, styling, caching. This skill covers only what is specific to pyKES,
and it overrides the general skill wherever they disagree.

## 1. This package ships components, not an application

`src/pyKES/streamlit_app/` is a **library**. External repositories supply their own
`Home.py` and `pages/`, and configure the components through dataclasses in
[config_interface.py](../../../src/pyKES/streamlit_app/config_interface.py) —
`FileUploadHandler`, `DataUploadConfig`, `HomeConfig`, `PyKESStreamlitConfig`. They do
not fork the UI.

**Adding behaviour should mean adding a field to a config dataclass, not editing a
component.** If a change would force every embedding repo to edit its own copy of a
component, it is the wrong change. Validate new config fields in `__post_init__`, per
the fail-fast rule in CLAUDE.md.

Run the app the way an external repo does:

```bash
streamlit run examples/external_repo/Home.py
```

There is no entry script in the package itself. `streamlit run src/pyKES/streamlit_app/Home.py`
does not exist.

## 2. The pages must survive stlite's single-threaded browser runtime

External repos deploy these pages as a static Pyodide page, where Python, Streamlit
and the UI share **one event loop**. A loop that processes everything inside one
script run delivers nothing to the screen until it has finished.

So: **long-running page work is chunked across reruns, never looped inline.** Use
[chunked_processing.py](../../../src/pyKES/streamlit_app/chunked_processing.py), which
advances a job one experiment per rerun via two `st.fragment`s with a `run_every`
timer — `render_job_progress` draws and does nothing else, `advance_job` processes one
experiment and draws nothing.

Three things that look like fixes and are not — each established by measuring the
deployed page in headless Chrome, not by reading:

- **Stepping the job with `st.rerun()` shows nothing at all.** `AppSession` clears
  unflushed messages on every `SCRIPT_STARTED`, so a run ending in `st.rerun` has its
  progress bar discarded before it is sent. On a server it survives, because the flush
  runs on another thread — which is exactly why this looks fixed under `streamlit run`.
- **Drawing the bar inside the working fragment wipes it.** `ForwardMsgQueue.clear`
  preserves the deltas of fragments *other* than those rerunning, so a fragment erases
  its own output on its own next tick.
- **`read_in_experiments_multiprocessing` cannot be used from a page.** It needs a
  `ProcessPoolExecutor`, and the browser has no processes to fork. The upload page
  steps through `ingest_experiment` one experiment at a time instead.

Expect a bar that appears immediately and climbs, but **not** once per experiment —
the single browser worker is busy processing, so only some paints land. Changing the
timer interval does not change that.

See [docs/browser_deployment.md](../../../docs/browser_deployment.md).

## 3. Session state

`st.session_state.experimental_dataset` (an `ExperimentalDataset`) is the **single
source of truth**. Pages mutate it in place; do not copy it into another key, and do
not rebuild it per rerun.

## 4. House style

Everything in CLAUDE.md applies here too, and these bite most often in UI code:

- **No nested function definitions.** Pass data through parameters, not closures.
- **Fail fast.** No broad `try/except`, no silent fallbacks — in Streamlit an exception
  surfaces to the user as a traceback, which is the desired behaviour.
- **Full-word names**, NumPy-style docstrings on every function, and named module-level
  constants instead of magic numbers.
