# A searchable database of photocatalytic results

A plan for turning the group's pyKES datasets into one place where every
photocatalytic result can be looked up, filtered, plotted and traced back to the
code that produced it — without a server anyone has to maintain.

The requirement that shapes every decision below is the last one in the brief:
**the database must not need a powerful server, and the computational work must
happen on the client.** That rules out the obvious answer (Postgres plus an API)
and rules *in* an answer that pyKES is already unusually well-placed to build,
because the Streamlit pages already run entirely in the browser under stlite.

---

## 1. What we have today

Three facts about the current data layer decide most of what follows.

**The schema is already a database schema.** `ExperimentalDataset` holds
`experiments` (name → `Experiment`), an `overview_df`, and three configuration
dictionaries. Each `Experiment` holds `metadata` (one row of the overview
sheet), `raw_data` (measured arrays), `processed_data` (derived arrays and
scalars) and a `version` dict. That is a table of records with a blob per record
— exactly the shape a searchable archive needs. Nothing has to be redesigned;
it has to be *split*.

**The numbers that matter are small; the arrays are not.** Measured on
`src/tests/data/260507_Complete.h5` (6 experiments, raw data only, no
`processed_data`):

| Quantity | Measured |
| --- | --- |
| File size | 768 KB |
| Array bytes per experiment | 96.3 KB |
| Metadata per experiment | ~0.2 KB |
| Overview sheet | 41 rows × 19 columns |

The metadata is ~500× smaller than the arrays. With `processed_data` present the
gap widens: `PLOTTING_INSTRUCTIONS` defines roughly thirty time series per
experiment (raw, reaction, smoothed, fits, rates, poly fits, flexible fits), so
a fully processed experiment is estimated at **0.3–0.5 MB**, of which the
searchable part — names, metadata, max rates, rate constants, AQY, LTH,
provenance — is **under 1 KB**.

The files are also written uncompressed. Rewriting the same fixture with
`gzip` level 4 gives 442 KB (**1.7×**), and casting float64 arrays to float32
first gives 353 KB (**2.2×**).

**The reader is all-or-nothing.** `load_from_hdf5` walks every dataset in the
file with `visititems` and materialises the lot. There is no way to ask for one
experiment, and no way to ask for metadata without the arrays. At 50
experiments that is fine — it is what the Home page does today. At 2000
experiments it is 600 MB–1 GB into a browser tab, which will not work.

So the single structural change is: **separate the thing you search from the
thing you plot.**

---

## 2. The shape of the answer

Two tiers, both static files:

```
site/
  index.html, files.js …          the stlite app bundle
  data/
    manifest.json                 database version, build time, counts
    index.json.gz                 ONE ROW PER EXPERIMENT — the searchable table
    experiments/
      NB-316.h5                   one payload per experiment (arrays)
      NB-318.h5
      …
```

* The **index** is fetched once at startup and lives in memory. Every search,
  filter, sort, cross-experiment plot and summary runs against it with pandas.
  At 5000 experiments × ~60 columns it is a few MB raw, **~1 MB gzipped**.
* A **payload** is fetched only when someone opens an experiment's traces.
  0.15–0.3 MB compressed, one HTTP GET, cached.

Everything the user experiences as "the database" — searching, faceting,
comparing, exporting — touches only the index. The arrays are pulled on demand,
in the amounts a human actually looks at.

The web server's entire job is to return files. An nginx directory, GitHub
Pages or an S3 bucket all qualify. There is nothing to provision, patch or
secure beyond access control.

---

## 3. Building the database from the pyKES schema

### 3.1 Options considered

**A — One big HDF5 file (scale up the status quo).**
Keep merging everything with `merge_hdf5_files` and hand people a single `.h5`.
*For:* zero new code, one artefact, works offline, already implemented.
*Against:* `load_from_hdf5` reads all of it, so the browser dies somewhere in
the low hundreds of experiments; every write rewrites the whole file; two people
cannot add experiments at once. Partial reads over HTTP would need a
range-request virtual file driver under h5py in Pyodide — a project in itself.
*Verdict:* keep it as the **interchange and offline format** (it already is),
not as the lookup format.

**B — Index table + per-experiment payloads.** ← **recommended**
*For:* search is instant and entirely client-side; traffic is proportional to
what is opened; static hosting; the index rebuild is cheap and incremental.
*Against:* needs a build step and produces many files (a non-issue for static
hosts).

**C — SQL in the browser: DuckDB-WASM or sql.js-httpvfs.**
DuckDB-WASM reads Parquet over HTTP range requests and can query files far
larger than memory — the purest form of "searchable database, no server", and
the right answer if the index ever outgrows a browser tab.
*Against:* it is a JavaScript library. From Streamlit-in-Pyodide it is reachable
only through a JS seam, which buys real SQL at the cost of the pattern the rest
of the app uses. At group scale (thousands, not billions, of rows) pandas over
the index is simpler and fast enough.
*Verdict:* documented escape hatch, not the starting point.

**D — A real server database (Postgres / Supabase / Firebase).**
Explicitly excluded by the brief. Worth noting that a *managed* instance is not
a server the group maintains, so it stays available for the write path (§6) if
the git-based route proves too stiff. It should not be on the read path: it
would make the archive unusable offline and unusable when the account lapses.

**E — Zarr / chunked array store.**
Designed exactly for range-reads of arrays over object storage. Genuinely
better than whole-file HDF5 *if* we ever need to plot one channel out of thirty
without downloading the other twenty-nine. At 0.3 MB per experiment we do not.
*Verdict:* revisit only if payloads grow past a few MB.

### 3.2 The index

One row per experiment, with four kinds of column:

| Prefix | Content | Source |
| --- | --- | --- |
| — | `experiment_name`, `group`, `color`, `active` | `Experiment` fields |
| `meta.` | every overview-sheet column | `overview_df` |
| `result.` | scalar derived results | `processed_data`, via config |
| `prov.` | `pykes_version`, `last_processed`, external app + version, `schema_version` | `Experiment.version` |
| `payload.` | relative path, byte size, sha256 | build step |

The derived-result columns are declared exactly like the existing results table,
reusing its path syntax:

```python
INDEX_INSTRUCTIONS = {
    'H2 max rate':  {'result': 'processed_data/H2_max_rate', 'unit': 'umol/L/s'},
    'O2 max rate':  {'result': 'processed_data/O2_max_rate', 'unit': 'umol/L/s'},
    'AQY':          {'result': 'processed_data/apparent_quantum_yield', 'format': '.2f'},
}
```

This matters for feasibility: `resolve_experiment_attributes(..., mode='permissive')`
already resolves those paths, and `results_table_component.resolve_result_value`
already does it per experiment. **The index builder is the results table, run
over every experiment and written to disk.** It is a small amount of genuinely
new code.

One deliberate choice: **take `meta.*` from `overview_df`, not from
`Experiment.metadata`.** The two can diverge, because a corrected overview sheet
only reaches `Experiment.metadata` when the experiment is reprocessed with a
`metadata_retrival_function` (see `docs/versioning_and_reprocessing.md`).
Reading the index from `overview_df` means a metadata fix shows up in search on
the next index build — seconds — instead of after a full reprocessing run. The
build should also *flag* divergence between the two, since a large divergence
means the stored results were computed from stale metadata.

A second choice: **index inactive experiments too.** `filter_active_experiments`
currently drops anything whose `metadata/Active` is not truthy. In an archive,
knowing that a run was attempted and failed is a result. Keep `active` as a
filter column, default the search to active-only, make it one click to include
the rest.

### 3.3 Index format

| Option | Size (5k rows) | Browser support | Notes |
| --- | --- | --- | --- |
| gzipped JSON `orient='split'` | ~1 MB | universal | **the format the repo already uses** for `overview_df` in `write_df_to_hdf` |
| Parquet | ~0.5 MB | needs Pyodide ≥ 0.28 | pyarrow and fastparquet were added in Pyodide 0.28; stlite ships it from 0.89.0 |
| Arrow IPC / Feather | ~0.7 MB | same constraint as Parquet | good if we ever want DuckDB-WASM |

**Start with gzipped JSON-split.** It carries zero dependency risk, works on
every stlite version the group might pin, and reuses the serialization already
in `database_experiments.py`, so there is one round-trip convention in the
codebase rather than two. Parquet is a drop-in upgrade once the pinned stlite
version is confirmed — worth doing when the index passes a few MB, and worth
verifying rather than assuming:

```python
# in the deployed browser app, once:
import pandas as pd; pd.DataFrame({'a': [1]}).to_parquet('/tmp/t.parquet')
```

### 3.4 Payload format

**A per-experiment HDF5 file written by the existing `save_to_hdf5`**, from a
single-experiment `ExperimentalDataset`. No new format and no new reader: it
loads back through `load_from_hdf5` unchanged, and every payload is
independently a valid pyKES dataset — so "download this experiment and work on
it locally" is free.

Add gzip compression when writing payloads (measured 1.7×; 2.2× with float32
display copies). Compression is transparent to h5py readers, so it needs no
`SCHEMA_VERSION` bump and older files keep loading.

### 3.5 Two small additions to the data layer

Both are useful on their own, independently of the database:

```python
# database_experiments.py
ExperimentalDataset.load_from_hdf5(filename, experiment_names=None)   # partial read
ExperimentalDataset.load_metadata_only(filename)                      # skip the arrays
```

`load_metadata_only` is what makes an index build over a large archive cheap —
today the builder would have to materialise every array to read a scalar next to
it. `experiment_names` is what lets the offline single-file workflow scale.

### 3.6 Identity

Experiment names are the primary key: `add_experiment` overwrites silently and
`merge_hdf5_files` skips duplicates with a printed warning. With several people
contributing, that is a real hazard. The group's naming convention already
carries an owner prefix (`NB-316`, `MZ-442`, `VSA-122`, `AE-855`), so the fix is
to **enforce** it: the build fails loudly on a collision and names both source
files. Cheap, and it prevents the one failure mode that silently loses data.

---

## 4. Hosting and deployment

The server serves bytes. What differs between the options is access control,
capacity and who administers it.

| Option | Cost | Private? | Capacity | Notes |
| --- | --- | --- | --- | --- |
| **GitHub Pages** | free | ✗ public (private needs Enterprise) | ~1 GB site, 100 MB/file, 100 GB/mo | already how the group deploys stlite apps |
| **Cloudflare Pages + Access** | free tier covers a small group | ✓ SSO, ~50 users free | generous | **recommended for the internal archive** |
| **Institutional web space** | already paid for | ✓ (basic auth / IP) | as provisioned | plain nginx; range requests work out of the box |
| **S3 / Cloudflare R2 / MinIO** | pennies (R2 has no egress fee) | ✓ signed URLs | unlimited | best if the archive outgrows 1 GB |
| **Zenodo** | free | ✗ public | 50 GB/record | **DOI per snapshot** — the citable, archival tier |

Three recommendations:

**Co-host the data with the app.** Put `data/` inside the same site as the
stlite bundle and fetch it with relative URLs. Same-origin means **no CORS
configuration at all**, which removes the most common way this kind of
deployment fails. If the data must live elsewhere, that host needs
`Access-Control-Allow-Origin` for the app's origin, and range support if we ever
adopt DuckDB-WASM.

**Decide privacy before building.** Unpublished photocatalytic results on
public GitHub Pages is not a thing that can be undone. Cloudflare Access in
front of Cloudflare Pages gives cookie-based SSO that `fetch` handles
transparently — considerably smoother than HTTP basic auth, which `pyfetch`
must be told about explicitly. An institutional host behind the university
login is equally fine.

**Publish snapshots to Zenodo.** A versioned DOI per database release means a
paper can cite the exact state of the archive its numbers came from. The
snapshot is one merged HDF5 plus the index — artefacts we are producing anyway.

Do **not** use Git LFS: LFS objects are not served by GitHub Pages, and it bills
bandwidth. Large-file needs are better met by GitHub Releases (2 GB per asset,
CDN-served) or object storage.

At an estimated 0.3 MB per experiment compressed, GitHub Pages' 1 GB ceiling is
roughly **3000 experiments** — likely years of headroom, and the migration to
object storage is a URL change in the manifest.

---

## 5. Searching and visualizing

### 5.1 Search

Everything here runs against the in-memory index, so it is instantaneous and
needs no server.

**Free text** across `experiment_name`, `group`, `meta.Notes` and every other
string column — one `st.text_input`, `str.contains` over a few thousand rows.
The `Notes` column is where the knowledge that never made it into a numeric
field lives, so it must be searchable.

**Schema-driven facets.** Rather than hand-configuring nineteen widgets,
generate them from the index dtypes and cardinality:

| Column kind | Widget |
| --- | --- |
| numeric (`Irradiance`, `Catalyst loading`, `Temperature`, max rates, AQY) | range slider |
| low-cardinality string / bool (`group`, `D2O`, `Active`) | multiselect |
| datetime (`last_processed`) | date range |
| high-cardinality string (`Notes`) | text contains |

A `SearchConfig` dataclass overrides the automatic choice where it guesses
wrong, in keeping with the repo's convention that new behaviour means a new
config field rather than an edited component.

**Shareable queries.** Encode the filter state in `st.query_params`, so a search
is a URL. "Every D2O run above 80 mW/cm²" becomes a link that can be pasted into
a group chat or a paper's SI. This is the single feature most likely to make
people actually use the thing, and it works under stlite.

**Expert mode.** One text box passed to `DataFrame.query()` for arbitrary
boolean expressions. Fail-fast: show the exception rather than swallowing it.

**Provenance queries** fall out for free, and are the reason the `prov.*`
columns are in the index: *"show me everything processed before pyKES 0.2.0"* is
how you find the results that need reprocessing after an algorithm change.

### 5.2 Visualization

Three views, two of which already exist:

**Result grid** — the search result as `st.dataframe` with row selection
(`on_select="rerun"`), column formatting from the index instructions, and CSV
export. This is `results_table_component` with the index as its source instead
of the loaded dataset.

**Property map** — a scatter of any index column against any other, coloured by
a third: AQY versus catalyst loading across the whole archive, max rate versus
irradiance, rate constant versus temperature. `analysis_results_component`
already does exactly this for one metadata axis and one result axis; generalising
its axis selectors to "any index column" turns a per-dataset plot into a view of
everything the group has ever measured. This is the feature that makes an
archive worth more than the sum of its files.

**Experiment detail** — selecting a row fetches that one payload and hands it
to the unchanged `time_series_component`. Selecting several overlays them.

Two guardrails, both learned from the existing code:

* Overlaying hundreds of traces will exhaust the tab. Warn above ~25 selected
  experiments, and use `utilities/time_series_resampling.py` to downsample for
  display.
* Cap the in-memory payload cache (LRU, ~50 experiments ≈ 25 MB). Unbounded
  `st.session_state` growth is the browser failure mode here.

**Subset export.** Any search result can be assembled into an
`ExperimentalDataset` from its payloads and written with `save_to_hdf5` — one
button that turns a query into exactly the file the current workflow already
knows how to use. This is what keeps the database from being a walled garden.

---

## 6. Adding, reprocessing and updating

A static host is read-only, so the write path is the part that needs real
design. Three mechanisms, which coexist.

### 6.1 Bulk build — git plus CI

**The database is a repository.** Raw files and overview sheets go in; a GitHub
Actions job runs the existing pyKES pipeline and publishes index + payloads.

```
raw-data + overview sheet  →  read_in_experiments_single_threaded
                           →  build_index / export_shards
                           →  publish to the static host
```

This is free compute on CI runners, not a server the group maintains, and it
gives review-before-merge, a full audit trail, and reproducibility. Incremental
builds are already supported: `select_unprocessed_experiments` and the
`Processed` flag mean only new rows are processed on each run.

**Reprocessing after an algorithm change** is the same job with
`reprocess_experiments` instead — it works from the stored `raw_data`, so the
original raw files are not needed, and every experiment's `version` dict records
the run. Bump pyKES, re-run, publish; the `prov.*` index columns then show the
whole archive moving to the new version.

This is the one place the brief's "computation client-side" has to bend, and it
is worth being explicit about why: reprocessing 2000 experiments in a browser
would mean downloading every payload (~600 MB) and hours of single-threaded
work. *Interactive* computation stays on the client; *bulk rebuilds* go to free
CI. No maintained server appears in either case.

### 6.2 Contribution from the browser

The elegant part: the app can already ingest raw data client-side. The existing
`chunked_processing` machinery steps `ingest_experiment` one experiment per
rerun precisely so this works in Pyodide. So a contributor can upload raw files
and an overview row, watch them process **in their own browser**, and get back a
single-experiment `.h5`. Then either:

* **Download and attach** it to a pull request or an issue — zero credentials,
  works today with the existing download button; or
* **Push directly** via the GitHub contents API from the browser, using a
  fine-grained, single-repo, contents-only token the user pastes in. Kept in
  `sessionStorage` only, never in the bundle.

Offer the download route first. It needs no token handling, and the review step
is a feature rather than friction when the archive is the group's record.

### 6.3 Metadata updates

Metadata lives in `overview_df`, which comes from an Excel sheet the group
already maintains — and `update_overview_df` already implements the merge, row
by row, preferring incoming values where they differ. Keep that as the primary
path: **edit the sheet, re-upload, rebuild the index.** Because §3.2 sources
`meta.*` from `overview_df`, a corrected catalyst loading is searchable as soon
as the index rebuilds, with no reprocessing at all.

Reprocessing is needed only when the correction must reach the *results* — that
is, when `processing_function` consumes the changed field. Then it is
`reprocess_experiments` with `metadata_retrival_function` supplied, which is
implemented and exposed in the app today.

For quick corrections, an `st.data_editor` over the search grid can emit a
patch — a small CSV of `experiment, column, new_value` — applied at build time
and committed. That keeps every metadata change reviewable and attributable,
which matters more in a shared archive than the convenience of editing in place.

---

## 7. Frontend and UX

### 7.1 Stay with Streamlit

The group has four working pages, an established config-dataclass extension
pattern, and a proven stlite deployment. A React/Observable SPA would be faster
and would open the door to DuckDB-WASM, but it would leave the entire Python
analysis stack — `max_rate`, `reaction_ODE`, `fitting_ODE`, the unit handler —
on the other side of a language boundary, and those are the things that make
opening an experiment worthwhile. Streamlit is the right call, and the honest
cost is that the result will feel like a data app rather than a website.

### 7.2 Pages

| Page | Status |
| --- | --- |
| **Browse & Search** (new landing page) | new |
| **Experiment detail** | new; wraps existing `time_series_component` |
| Property map | generalise `analysis_results_component` |
| Results table | point `results_table_component` at the index |
| Contribute / Upload | existing `data_upload_component`, plus export-for-PR |

**The Home page has to change.** Today it says "upload an HDF5 file" — correct
for a personal dataset, wrong for a group archive, where the answer to "where is
the data" must be "it is already here". The index loads at startup; the page
opens on *N experiments, M groups, last updated on …*, which
`_render_dataset_statistics` already computes. Uploading becomes an option, for
working offline or viewing an unpublished file alongside the archive.

### 7.3 New configuration

Following the repo's convention that new behaviour is a new config field:

```python
@dataclass
class DatabaseSourceConfig:
    manifest_url: str = "data/manifest.json"   # relative → same-origin, no CORS
    index_url: str | None = None               # defaults to the manifest entry
    payload_url_template: str | None = None
    payload_cache_size: int = 50

@dataclass
class SearchConfig:
    index_instructions: dict = field(default_factory=dict)
    free_text_columns: list = field(default_factory=list)
    default_columns: list = field(default_factory=list)
    facet_overrides: dict = field(default_factory=dict)
    default_filters: dict = field(default_factory=lambda: {'active': True})
```

### 7.4 One shim that has to be written

Under `streamlit run`, fetching a URL is `urllib`. Under stlite it is
`pyodide.http.pyfetch` — `requests` and `urllib` do not work in Pyodide, and
`open_url` handles text only, not the binary payloads. So pyKES needs a small
`fetch_bytes(url)` that picks the mechanism at runtime, in the same spirit as
`chunked_processing`: one module that absorbs a browser constraint so the pages
do not have to know about it.

The asynchrony is the catch — `pyfetch` is a coroutine, and a Streamlit script
is synchronous. This needs measuring in a real browser before the design is
settled, exactly as the progress-bar work in
`docs/browser_deployment.md` did. Two candidate approaches: stlite's bundle
mounting (`files` / `archives`) for the index, which side-steps fetching it
altogether, and a `run_every` fragment for payloads, which is the pattern
already proven to work here. **This is the main technical unknown in the plan**
and belongs in Phase 0.

---

## 8. The compute budget

| Who | Does what | Needs |
| --- | --- | --- |
| **Web server** | returns static files, checks a cookie | nginx / CDN — nothing to maintain |
| **Browser** | search, filter, plots, per-experiment processing, subset export | the tab it already has |
| **CI (free runners)** | bulk ingestion, bulk reprocessing, index build, publish | a GitHub Actions workflow |

Browser memory at 5000 experiments: index ~5 MB resident, plus at most 50 cached
payloads at ~0.3 MB — **under 25 MB**. The current app loads a whole dataset
into `st.session_state`; the two-tier split is what keeps that from growing
without bound.

---

## 9. Suggested phasing

**Phase 0 — measure and de-risk (days).** Build an index over the group's real
archive and record its true size; measure a fully processed experiment with
`processed_data` present, since every estimate above extrapolates from a
raw-data-only fixture. Settle the fetch shim in a real browser. Confirm the
pinned stlite version and whether Parquet is available in it. Decide hosting and
privacy. *Nothing else should start before the fetch question is answered.*

**Phase 1 — the data layer.** `pyKES/database/index.py`:
`build_index`, `export_shards`, `load_index`, `fetch_experiment`, plus
`load_metadata_only` and partial `load_from_hdf5`. Gzip compression on write.
Collision enforcement. Round-trip tests against synthetic datasets with known
contents, per the repo's testing convention.

**Phase 2 — the Browse & Search page,** with `DatabaseSourceConfig` and
`SearchConfig`, schema-driven facets, URL-encoded queries, and detail-on-select.

**Phase 3 — the build pipeline and hosting.** The Actions workflow, the
manifest, publication to the chosen host, and the "last updated" banner.

**Phase 4 — the write path.** Export-for-PR from the browser, the metadata
patch mechanism, the reprocessing workflow.

**Phase 5 — the payoff.** Property maps across the whole archive, saved queries,
provenance dashboards, Zenodo snapshots.

Phases 1–3 are the minimum that delivers "one place to look up all
photocatalytic results". Phases 4–5 are what keep it current and make it worth
more than the files it was built from.

---

## 10. Decisions needed, and open risks

**Needed from the group:**

1. **How many experiments, now and in five years?** Everything above assumes
   thousands. At tens of thousands, Parquet plus DuckDB-WASM moves from escape
   hatch to starting point.
2. **Public or private?** This decides the host, and it cannot be reversed after
   the fact.
3. **Who may add data?** Everyone with a git account, or a maintainer who
   merges? This decides how much of §6.2 gets built.

**Risks:**

* *The fetch shim.* The single unproven piece. Mitigated by Phase 0, and by
  stlite's bundle mounting as a fallback for the index.
* *Size estimates extrapolate from raw data only.* The fixtures in
  `src/tests/data/` carry no `processed_data`. If a processed experiment turns
  out to be 2 MB rather than 0.4 MB, payload compression and float32 display
  copies stop being optional.
* *Divergence between `overview_df` and `Experiment.metadata`.* Made visible by
  the index build rather than left silent.
* *`overview_df` round-trips through JSON,* which is lossy for exotic dtypes.
  Already true today; the index inherits it. Worth a test with the group's real
  sheet.
* *The archive going stale.* The real risk is social, not technical: if
  contributing is harder than keeping a local file, people will keep local
  files. Phase 4 is not optional polish.

---

## References

* [browser_deployment.md](browser_deployment.md) — the single-event-loop
  constraint and the chunked-processing pattern every long-running page here
  must use.
* [versioning_and_reprocessing.md](versioning_and_reprocessing.md) — the version
  dictionaries the `prov.*` index columns expose, and the reprocessing pipeline
  §6 builds on.
* [plotting_instructions.md](plotting_instructions.md) — the instruction syntax
  the index instructions extend.
