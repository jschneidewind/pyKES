# A private database of photocatalytic results

A plan for one place where the group can look up every photocatalytic result:
searchable, plottable, traceable to the code that produced it, hosted on the
group's own server, and readable only by people who have signed in.

Three constraints shape every decision below:

1. **Strictly private, on our own hardware.** No third-party provider holds the
   data. Not GitHub, not object storage, not a managed database.
2. **Everyone uploads.** Users process their own data locally with a
   purpose-built pyKES app, obtain an HDF5 file holding a batch of experiments,
   and upload that file to the database.
3. **The metadata will grow.** Experiments run next year will carry fields that
   do not exist today, and the database must absorb them without a migration.

It has to work at **10,000 experiments**. Every scale claim below is measured
rather than estimated; the measurements are in §2 and §5.

---

## 1. What changed, and what survives

An earlier version of this plan targeted a static, server-free deployment: files
on a static host, all computation in the browser under stlite. Two of the new
constraints kill that outright, and it is worth being explicit about why.

**A static file server cannot authenticate users or accept uploads.** Both are
now required. The moment the system has a login and an upload button, it is a
web application with a backend, not a directory of files. That is not a
regression — it is a different, and in most respects easier, problem.

**Loading the whole index into the browser stops being the obvious choice.**
Measured on a synthetic 10,000-experiment index (§5), pulling the entire index
into one pandas DataFrame costs 2.05 MB gzipped over the wire, 15.9 MB resident,
**66.5 MB peak allocation** and 1.93 s to assemble — on server-class CPU, before
Pyodide's overhead, on every page load, growing with every new metadata column.
The same queries answered by SQLite on the server take **4–24 ms** and do not
grow. With a server in the picture, the server should do the querying.

What survives from the earlier design, unchanged and still right:

* **The two-tier split** — separate the thing you search from the thing you plot.
  An experiment carries ~0.2 KB of searchable metadata against 96.3 KB of arrays
  (measured), so they belong in different places.
* **Per-experiment HDF5 payloads** written by the existing `save_to_hdf5`, so
  each one loads back through `load_from_hdf5` unchanged and is independently a
  valid pyKES dataset.
* **The index builder is the results table**, run over every experiment —
  `resolve_experiment_attributes` already resolves the paths.
* **Schema-driven facets**, now backed by a registry that grows by itself (§4.3).

**And the original "computation on the client" requirement is still met** — just
by the workflow rather than by the runtime. The expensive work, turning raw
traces into processed data, happens on each user's own machine in the local
processing app, *before* anything is uploaded. The server never runs a
processing function. It stores bytes, maintains an index, answers queries and
draws plots, which is why it does not need to be a powerful machine (§9).

---

## 2. The shape of the answer

One application on the group's server, behind a reverse proxy that handles TLS
and authentication, with three storage tiers on local disk:

```
/srv/photocat/
  app/                        the Streamlit application
  data/
    index.sqlite              THE SEARCHABLE INDEX — one row per experiment
    payloads/
      NB-316.h5               one payload per experiment, written by save_to_hdf5
      NB-318.h5
      …
    uploads/
      <sha256>.h5             every uploaded file, kept verbatim
```

| Tier | Holds | Size at 10 000 | Read when |
| --- | --- | --- | --- |
| `index.sqlite` | metadata, scalar results, provenance, payload pointers | **16.4 MB** (measured) | every search — in milliseconds |
| `payloads/` | raw + processed arrays, one file per experiment | 3–5 GB | someone opens an experiment's traces |
| `uploads/` | the original uploaded files, untouched | 3–5 GB | never, except to rebuild |

The third tier is the one that is easy to skip and shouldn't be. Keeping every
uploaded file verbatim means **the entire database can be rebuilt from
scratch** — when the index schema changes, when a mapping was wrong, when a bug
in ingestion is found. It costs a few gigabytes of disk and it is the difference
between a database you can fix and one you cannot. A full rebuild of 10,000
experiments takes **about 2.5 minutes** (measured in §3.3).

Everything a user experiences as "the database" — searching, faceting,
comparing, exporting — touches only `index.sqlite`. Payload files are opened one
at a time, on demand.

---

## 3. Two applications, one file format between them

The workflow in the brief separates cleanly into two programs that share the
HDF5 file as their interface. Naming that split explicitly is what keeps both
simple.

### 3.1 The processing app (local, per user)

This is essentially what pyKES already ships. A user runs it on their own
machine, uploads the metadata Excel sheet and the raw data files, the existing
pipeline processes them, and the app produces one HDF5 file containing the batch
— metadata, raw data and processed data for each experiment, plus the dataset
level dictionaries.

Nothing here needs to change except one addition: the file must also carry its
**index mapping** (§6), so the database knows how to read its results.

This app can stay exactly as it is deployed today, in the browser under stlite
or run locally with `streamlit run`. All the constraints in
[browser_deployment.md](browser_deployment.md) continue to apply to *it*.

### 3.2 The database app (on the group's server)

A server-side Streamlit application — an ordinary `streamlit run`, not stlite.
It authenticates, accepts uploads, ingests them, and lets people search and plot.

Because it runs on a server with a real thread, **none of the stlite constraints
apply to it**: no single-event-loop problem, no fetch shim, no chunked
processing forced by the runtime. `chunked_processing` is still useful for
drawing an upload progress bar, but it is a convenience here rather than the
only thing that works.

### 3.3 What ingestion actually does

When a user uploads `batch_2026_09.h5` holding 40 experiments, the server:

1. **Hashes** the file. An identical file already ingested is a no-op.
2. **Validates** it: readable by `load_from_hdf5`, `schema_version` understood,
   experiment names present and unique within the file, index mapping present or
   defaulted.
3. **Checks for collisions** against experiment names already in the database,
   and applies the collision policy (§10 — this is a decision the group has to
   make, not one the code can make).
4. **Splits** it into one payload file per experiment, gzip-compressed.
5. **Applies the index mapping** to fill the `results` for each experiment.
6. **Upserts** the index rows, and registers any metadata or result keys not
   seen before (§4.3).
7. **Records provenance**: who uploaded it, when, the file hash, the pyKES and
   app versions carried in the file's `version` dict.
8. **Stores the original** under its hash.

Measured on `src/tests/data/260507_Complete.h5`, step 4 — the only step whose
cost scales with data volume — takes **14.8 ms per experiment** with gzip
compression (13.8 ms without; compression is essentially free in time and saves
33% of disk). So:

| Upload | Split time |
| --- | --- |
| 40-experiment batch | **~0.6 s** |
| 10 000-experiment full rebuild | **~2.5 min** |

**Ingestion can therefore run synchronously**, inside the request, with a
progress bar. No job queue, no worker process, no broker. That is a substantial
simplification and it holds comfortably at the stated scale. (The fixture
carries raw data only; with `processed_data` present, expect roughly 3–5× the
bytes and time, which still leaves a 40-experiment upload at a few seconds.)

`server.maxUploadSize` defaults to 200 MB and will need raising — at 0.3–0.5 MB
per experiment, a 200-experiment batch is around 100 MB, so 500–1000 MB is a
sensible setting.

---

## 4. The index, and how it survives a growing schema

This is the core engineering problem. "The metadata might change over time" is
the requirement that decides the storage design, and getting it wrong means a
schema migration every time somebody adds a column to their Excel sheet.

### 4.1 Why SQLite

| Option | Verdict |
| --- | --- |
| **SQLite** | **Recommended.** In the Python standard library, single file, WAL mode gives concurrent readers with one writer, JSON1 for schemaless metadata, FTS5 for free text, generated columns for hot fields. Nothing to install, nothing to administer, trivially backed up by copying one file. |
| DuckDB | Equally good, better at wide analytic scans. Worth swapping in if property maps over all 10 000 rows become the dominant query. Adds a dependency for no gain at this scale. |
| PostgreSQL | A service to run, secure, back up and upgrade, buying concurrency the group does not need. Reconsider above ~10 concurrent writers. |
| A file-based index (Parquet / JSON) | What the previous plan proposed for a static host. With a server it is strictly worse: no incremental update without a full rewrite, no indexes, no concurrent access. |

### 4.2 The table shape: typed core plus JSON

Three patterns exist for schemaless-ish data, and only one of them is right here:

* **A wide table, `ALTER TABLE ADD COLUMN` per new field.** Fast and typed, but a
  migration for every new metadata field, and a table that grows steadily
  sparser as eras of experiments accumulate.
* **EAV** — one row per (experiment, key, value). Infinitely flexible; every
  query becomes a pile of self-joins.
* **A typed core plus a JSON column.** ← recommended

```sql
CREATE TABLE experiments (
  id               INTEGER PRIMARY KEY,
  experiment_name  TEXT UNIQUE NOT NULL,   -- the primary key users think in
  exp_group        TEXT,
  active           INTEGER,
  payload_path     TEXT,                   -- payloads/NB-316.h5
  payload_bytes    INTEGER,
  payload_sha256   TEXT,
  uploaded_by      TEXT,                   -- from the authenticated session
  uploaded_at      TEXT,
  upload_id        INTEGER REFERENCES uploads(id),
  pykes_version    TEXT,
  external_app     TEXT,
  external_version TEXT,
  last_processed   TEXT,
  metadata         TEXT NOT NULL,          -- JSON: every overview-sheet field
  results          TEXT NOT NULL           -- JSON: every mapped scalar result
);
```

Stable things that every experiment has, and that the application itself depends
on, are real columns. Everything that varies between experiments and eras —
which is all of the scientific metadata — lives in `metadata` as JSON. **A new
metadata column requires no migration at all**: it simply appears in the JSON of
the experiments that have it, and in the registry below.

### 4.3 The registries: what makes the growth manageable

Two small tables, maintained at ingestion, turn "the schema grows" from a
problem into a feature:

```sql
CREATE TABLE metadata_keys (
  key             TEXT PRIMARY KEY,
  label           TEXT,          -- display name, editable by an admin
  canonical_key   TEXT,          -- set when this key is an alias of another
  inferred_type   TEXT,          -- number | text | bool | date | mixed
  unit            TEXT,
  occurrences     INTEGER,
  first_seen      TEXT,
  last_seen       TEXT,
  distinct_sample TEXT           -- JSON sample, for building facet widgets
);

CREATE TABLE result_keys (
  label           TEXT PRIMARY KEY,
  path            TEXT,          -- processed_data/H2_max_rate
  unit            TEXT,
  format          TEXT,
  defined_by      INTEGER REFERENCES uploads(id),
  conflicting     INTEGER DEFAULT 0
);
```

`metadata_keys` is what the **search page generates its facets from** — numeric
keys get range sliders, low-cardinality keys get multiselects, text keys get a
contains box. Add a new column to your Excel sheet, upload, and the filter for
it appears by itself. That is the schema-driven facet idea from the earlier
plan, now with a registry behind it that maintains itself.

It also carries the two failure modes that a growing free-form schema really
has:

**Key drift.** `Irradiance [mW/cm2]` and `Irradiance (mW/cm2)` are two different
keys, created silently by two people typing two spreadsheet headers. Nothing can
prevent this; what the registry can do is make it *visible* — an admin page
lists near-duplicate keys, and `canonical_key` aliases one onto the other so
both old and new uploads answer the same filter.

**Type conflict.** The same key arriving as a number in one upload and a string
in another. `inferred_type` records what has actually been seen; on conflict it
becomes `mixed` and the UI degrades that facet to a text filter rather than
producing a broken slider.

Both are surfaced rather than silently resolved. Guessing here would corrupt
searches in ways nobody would notice.

### 4.4 Hot-key promotion

JSON extraction is a full scan. At 10 000 rows that is already fast enough
(17.1 ms, measured), but a frequently-filtered key can be promoted to an indexed
generated column:

```sql
ALTER TABLE experiments ADD COLUMN irradiance REAL
  GENERATED ALWAYS AS (CAST(json_extract(metadata,'$."Irradiance [mW/cm2]"') AS REAL)) VIRTUAL;
CREATE INDEX idx_irr ON experiments(irradiance);
```

Measured: **17.1 ms → 4.3 ms**, with no measurable increase in database size.

One practical finding worth recording, because it is not in the obvious place in
the documentation: **SQLite's `ALTER TABLE` accepts only `VIRTUAL` generated
columns, not `STORED`.** A `STORED` column fails with
`cannot add a STORED column`. `VIRTUAL` columns can still be indexed, and the
index is what carries the speed-up, so this costs nothing — but the promotion
migration must say `VIRTUAL`.

Free-text search over notes gets the same treatment with an FTS5 table:
**16.5 ms → 1.3 ms**, measured.

---

## 5. Does it hold at 10 000 experiments?

Measured on a synthetic index of 10 000 experiments with a deliberately
*evolving* schema — the first 3 000 carry 19 metadata keys, the next 4 000 carry
24, the last 3 000 carry 30 — plus 14 mapped results each:

| Operation | Time |
| --- | --- |
| Build and insert all 10 000 rows | 0.49 s |
| Numeric range filter on a metadata key (JSON scan) | **17.1 ms** |
| Three predicates + a result threshold | **4.7 ms** |
| Facet: distinct values and counts for one key | 23.9 ms |
| Free-text scan across notes (`LIKE`) | 16.5 ms |
| Property map: two numeric columns for every active row | 14.9 ms |
| Key registry: every metadata key ever seen, with counts | 79.7 ms |
| Numeric filter after promoting the key to an indexed column | **4.3 ms** |
| Free-text via FTS5 instead of `LIKE` | **1.3 ms** |
| **Database size, including FTS index** | **16.4 MB** |

Every interactive query is comfortably under the threshold where a user notices
delay, with a plain SQLite file and no tuning. The design has roughly two orders
of magnitude of headroom before any of this needs revisiting.

One number deserves attention: the key-registry query, which walks
`json_each` over every row, is the slowest at 79.7 ms — and it is the one the
search page needs on *every* load to build its facets. That is exactly why
`metadata_keys` is a maintained table rather than a query: it is updated at
ingestion, when the cost is paid once, instead of recomputed per page view.

For comparison, the browser-side alternative at the same scale: 10 000 × 56
columns, 2.05 MB gzipped download, 15.9 MB resident, **66.5 MB peak
allocation**, 1.93 s to assemble on server-class CPU — per page load, growing
with every new column. The server-side index is both faster and simpler.

---

## 6. The index mapping travels in the HDF5 file

Each uploaded file declares how its own `processed_data` maps into the database's
result columns. This is what lets the database absorb files produced by
different versions of different processing apps without server-side changes.

**Where it lives.** Inside the existing `plotting_instruction` dictionary, as an
`index_instructions` entry:

```python
plotting_instruction['index_instructions'] = {
    'H2 max rate': {'result': 'processed_data/H2_max_rate',
                    'unit': 'umol L^-1 s^-1'},
    'O2 max rate': {'result': 'processed_data/O2_max_rate',
                    'unit': 'umol L^-1 s^-1'},
    'AQY':         {'result': 'processed_data/apparent_quantum_yield',
                    'format': '.2f'},
}
```

This deliberately reuses the syntax and the location of
`results_table_instructions`, which already lives in `plotting_instruction` and
is already resolved by `resolve_experiment_attributes`. **It needs no HDF5
schema change at all** — `plotting_instruction` is already written to the file
root as a JSON attribute, so an added key travels with the file and older
readers ignore it.

**How the server applies it.** For each experiment in the upload, resolve every
declared path in permissive mode, coerce what resolves to a scalar, and store
the result under its label. A path that does not resolve for a given experiment
is simply absent from that experiment's `results` — which is the correct
behaviour when a batch mixes liquid-phase and gas-phase runs, and exactly what
`mode='permissive'` already does.

**Three cases the server has to decide, not discover:**

* **No mapping in the file.** Fall back to a server-side default mapping. Reject
  only if there is no default either.
* **A label defined with a different path than a previous upload declared.**
  Record both in `result_keys`, mark `conflicting`, and surface it on the admin
  page. Silently overwriting would change the meaning of a column for every
  experiment already in the database.
* **A new label never seen before.** Register it, and it becomes a searchable
  column from that upload onward. Earlier experiments simply lack it — the same
  sparsity the metadata already has, handled the same way.

Because the original uploads are kept (§2), a mapping mistake is recoverable:
fix the default mapping and re-ingest from `uploads/` without asking anyone to
re-upload anything.

---

## 7. Authentication and privacy

The data is strictly private and lives only on the group's server. That makes
sign-in the part of this system where a mistake is most expensive, so it is
worth being deliberate about where authentication lives.

### 7.1 Authenticate at the reverse proxy, not inside Streamlit

Three reasons, in order of importance:

1. **Payload files are not served by Streamlit.** For any reasonable download
   performance, nginx serves the HDF5 files directly. If the gate were inside
   the application, those URLs would be unprotected — anyone who learned or
   guessed a path could fetch experimental data without logging in.
2. **A Streamlit session is not a security boundary.** The script runs, then
   decides what to show. Anything that renders before the check, any exception
   path, any stale session, is a potential leak. A proxy that refuses to forward
   an unauthenticated request has no such failure mode.
3. It is the standard, auditable pattern, and it keeps authentication out of the
   application code entirely.

The application still needs to know *who* is signed in, for attribution on
uploads. The proxy sets a header (`Remote-User`), which Streamlit reads via
`st.context.headers`. **Verify this early**: `st.context.headers` reflects the
`/_stcore/stream` WebSocket request rather than the initial page request, so the
header must be set on the WebSocket-upgrade location too, not only on `/`.

### 7.2 The options

| Option | Self-contained? | 2FA | Weight | Notes |
| --- | --- | --- | --- | --- |
| **Authelia** | ✓ fully | ✓ TOTP | container under 20 MB, under 30 MB RAM | **Recommended.** Forward-auth for nginx, one declarative config file you can version-control, local file or LDAP user backend. Exactly the "add a login page and 2FA to a self-hosted app" case. |
| Institutional SSO (SAML/OIDC) | ✓ for data | ✓ inherited | none of your own | Accounts follow employment, no passwords to manage. The IdP sees *who logs in* — never the data — so this does not put research data with a third party. Costs a registration request to university IT. |
| Authentik | ✓ | ✓ | needs PostgreSQL + Redis | A full IdP with an admin UI. Choose it if you need SAML, LDAP or dozens of managed users. |
| Keycloak | ✓ | ✓ | heavy (JVM) | Enterprise-grade, far more than a research group needs. |
| `streamlit-authenticator` | ✓ | limited | trivial | In-app, so it fails reason 1 and 2 above. Not recommended for private data. |
| nginx basic auth | ✓ | ✗ | trivial | Acceptable as a stopgap on day one. No 2FA, no session management, no logout. |

**Recommendation:** Authelia in front of nginx if the group wants zero external
dependencies, institutional SSO if university IT is responsive. Both are
compatible with the same application code, because in both cases the app only
ever reads a username from a header. Starting with basic auth and swapping later
is a legitimate path — the application does not change.

Streamlit's own `st.login` (native OIDC since 1.42) is a real option if an OIDC
provider is running anyway, but it protects only the application, not the
payload files, so it does not remove the need for a proxy gate.

### 7.3 Serving payloads safely

Payload downloads should be authorized by the application but served by nginx.
The `auth_request` plus `X-Accel-Redirect` pattern does this: Streamlit decides
whether this user may have this experiment, then hands nginx an internal
redirect, and nginx streams the file without the bytes passing through Python.

```nginx
location /payloads/ {
    internal;                          # unreachable from outside
    alias /srv/photocat/data/payloads/;
}
```

The `internal` directive is what makes the path unreachable except by internal
redirect. Without it, the files are simply on the web.

### 7.4 The rest of the deployment

Self-hosting means these are now the group's responsibility, and none of them
are optional:

* **TLS**, from the institutional certificate authority or Let's Encrypt, with
  automatic renewal. Private data over plain HTTP is not private.
* **WebSocket proxying** — Streamlit needs `proxy_http_version 1.1`, the
  `Upgrade` and `Connection` headers, and generous `proxy_read_timeout`. Without
  these the app loads and then appears frozen.
* **Backups** of `index.sqlite` and `uploads/` (from which everything else can be
  rebuilt). `payloads/` need not be backed up at all — it is derived.
* **Rate limiting** on the login endpoint, and OS updates.
* No directory listing anywhere under `data/`.

---

## 8. Searching, visualizing, and the pages

Largely as in the earlier plan, but with SQL underneath instead of a DataFrame.

**Browse & Search** is the landing page. A free-text box over names, groups and
notes (FTS5); facets generated from `metadata_keys`; results as a paginated
`st.dataframe` with row selection. Filter state encoded in `st.query_params`, so
**a search is a URL** that can be pasted into a group chat — the single feature
most likely to make people actually use the thing. An expert mode passes a
`WHERE` fragment straight through for anything the facets cannot express.

Pagination matters at this scale: query with `LIMIT`/`OFFSET` and render a page
at a time. Rendering 10 000 rows into a table is the one easy way to make a
fast system feel slow.

**Experiment detail** loads one payload from local disk — no network, a few
milliseconds — and hands it to the existing `time_series_component` unchanged.
Selecting several overlays them, with a warning past ~25 and display
downsampling via `utilities/time_series_resampling.py`.

**Property map** is a scatter of any index column against any other, coloured by
a third: AQY against catalyst loading across the whole archive, max rate against
irradiance. Measured at 14.9 ms for the underlying query over 10 000 rows. This
is the view that makes an archive worth more than the sum of its files, and
`analysis_results_component` already does the single-dataset version.

**Upload** is the contribution page: file uploader, validation report, collision
report with a decision, ingestion progress, and a summary of what was added,
including any newly registered metadata or result keys.

**Admin** covers the things a growing schema needs: the key registry with
aliasing, result-key conflicts, the upload log, and a "rebuild from uploads"
action.

**Subset export** — any search result assembled into an `ExperimentalDataset`
from its payloads and written with `save_to_hdf5`. One button that turns a query
into exactly the file the local processing app already understands, which is
what keeps the database from being a walled garden.

Configuration follows the repo's convention that new behaviour is a new config
field:

```python
@dataclass
class DatabaseConfig:
    data_root: Path = Path("/srv/photocat/data")
    index_path: Path = Path("/srv/photocat/data/index.sqlite")
    default_index_instructions: dict = field(default_factory=dict)
    collision_policy: str = "reject"        # reject | version | replace
    user_header: str = "Remote-User"
    max_overlay_experiments: int = 25
```

---

## 9. What the server needs

| Resource | At 10 000 experiments |
| --- | --- |
| Disk | payloads 3–5 GB + originals 3–5 GB + index 16 MB → **under 15 GB** |
| RAM | Streamlit holds a session per concurrent user; **4–8 GB** is ample for a research group |
| CPU | queries 4–24 ms; ingestion ~15 ms per experiment. Any modern core. |
| Services | nginx, the Streamlit app, and an auth service (Authelia is under 30 MB RAM) |

This is a small virtual machine, or an existing group workstation. Nothing here
needs a database server, a job queue, a message broker or a container
orchestrator, and the plan should be resisted if it starts to acquire them.

The division of labour that keeps it this small:

| Who | Does what |
| --- | --- |
| **User's own machine** | reads Excel + raw data, runs the processing functions, produces the HDF5 — all the expensive computation |
| **Server** | authenticates, stores bytes, splits uploads, maintains the index, answers queries, renders plots |
| **Browser** | draws what Streamlit sends |

---

## 10. Phasing

**Phase 0 — decide and provision.** Pick the authentication route (§7.2) and the
collision policy (§11); provision the server with TLS and a working
nginx + Streamlit + WebSocket configuration; confirm `Remote-User` reaches
`st.context.headers` through the WebSocket upgrade. *The auth decision gates the
deployment, not the code — everything in Phase 1 can proceed in parallel.*

**Phase 1 — the data layer.** A new `pyKES/database/index.py`: the SQLite schema,
`ingest_upload`, payload splitting with gzip, the mapping application, the two
registries, and `rebuild_from_uploads`. Round-trip tests against synthetic
datasets with known contents, plus an explicit test that a second upload
introducing new metadata keys is absorbed without migration.

**Phase 2 — the database app.** Browse & Search against SQLite, experiment
detail, pagination, facets from the registry.

**Phase 3 — the upload path.** Validation, collision handling, ingestion with
progress, the upload log.

**Phase 4 — deployment.** Auth, TLS, `internal` payload serving, backups,
and the hardening list in §7.4.

**Phase 5 — the rest.** Property maps, the admin page with key aliasing, subset
export, provenance dashboards.

Phases 1–4 are the minimum that delivers a private, searchable, uploadable
database. Phase 5 is what makes it worth more than the files it was built from.

---

## 11. Decisions needed, and open risks

**Needed from the group:**

1. **Which authentication route?** Institutional SSO if IT is responsive,
   Authelia if you want zero external dependencies. The application code is the
   same either way, so this can be decided late — but not after the data is on
   the server.
2. **What happens on an experiment-name collision?** With everyone uploading,
   two people will eventually use the same name, or the same person will
   re-upload a corrected batch. Reject the upload, keep both under a version
   suffix, or replace the existing one? *This is a scientific-record policy
   question, not a technical one, and the code cannot choose for you.*
3. **Who may delete or correct an entry?** Anyone, or an owner and an admin? The
   index carries `uploaded_by`, so either is implementable — but it should be
   decided before people rely on it.
4. **Are original uploads kept indefinitely?** Recommended yes (§2). It doubles
   the disk, which is a few gigabytes, and it is what makes the database
   repairable.

**Risks:**

* **Metadata key drift** — two spellings of one field, created silently. The
  registry makes it visible and aliasable, but somebody has to look at the admin
  page occasionally. This is the most likely way the search quality degrades
  over years.
* **Streamlit is not a hardened multi-user framework.** The proxy gate in §7.1 is
  load-bearing, not defence in depth. Do not move authentication into the app
  for convenience.
* **Self-hosting is now the group's responsibility** — TLS renewal, OS updates,
  backups. A private server that nobody patches is not more secure than a
  managed one; it is less.
* **Scale figures come from raw-data-only fixtures.** The 96.3 KB and 14.8 ms
  per experiment were measured on files with no `processed_data`. Expect 3–5× on
  real files: still comfortable, but measure a real processed batch in Phase 1
  before sizing the disk.
* **One writer at a time.** SQLite in WAL mode gives concurrent readers and a
  single writer, so ingestion must hold a lock. At the rate a research group
  uploads, this will never be contended — but two simultaneous uploads must
  queue rather than corrupt.

---

## References

* [versioning_and_reprocessing.md](versioning_and_reprocessing.md) — the version
  dictionaries the index's provenance columns expose, and the reprocessing
  pipeline the local processing app runs.
* [browser_deployment.md](browser_deployment.md) — the single-event-loop
  constraint. It governs the *local processing app*; the server-side database
  app is free of it.
* [plotting_instructions.md](plotting_instructions.md) — the instruction syntax
  that `index_instructions` extends.

Measurements in §2, §3.3, §4.4 and §5 were taken in this repository against
`src/tests/data/260507_Complete.h5` and against a synthetic 10 000-experiment
index built with an intentionally evolving metadata schema.
