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
4. **Entries reference each other.** An experiment names the catalyst batch it
   used, which names its precursor, which names its source material — and a
   search must see the whole chain's metadata as if it were the experiment's own.

It has to work at **10,000 experiments**. Every scale claim below is measured
rather than estimated; the measurements are in §2, §5 and §6.

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
Measured on a synthetic 10,000-experiment index (§6), pulling the entire index
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
draws plots, which is why it does not need to be a powerful machine (§10).

---

## 2. The shape of the answer

One application on the group's server, behind a reverse proxy that handles TLS
and authentication, with three storage tiers on local disk:

```
/srv/photocat/
  app/                        the Streamlit application
  data/
    index.sqlite              THE SEARCHABLE INDEX — one row per entity (§5)
    payloads/
      NB-316.h5               one payload per experiment, written by save_to_hdf5
      NB-318.h5
      …
    uploads/
      <sha256>.h5             every uploaded file, kept verbatim
```

| Tier | Holds | Size at 10 000 | Read when |
| --- | --- | --- | --- |
| `index.sqlite` | metadata, scalar results, provenance, payload pointers, and the reference graph (§5) | **16.4 MB** flat; **24 MB** with inherited metadata materialised (measured) | every search — in milliseconds |
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
**index mapping** (§7), so the database knows how to read its results.

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
   and applies the collision policy (§12 — this is a decision the group has to
   make, not one the code can make).
4. **Splits** it into one payload file per experiment, gzip-compressed.
5. **Applies the index mapping** to fill the `results` for each experiment.
6. **Upserts** the index rows, and registers any metadata or result keys not
   seen before (§4.3).
7. **Records the references** the file declares, resolves the effective metadata
   of everything it touches, and recomputes any existing entry that transitively
   references something the upload changed (§5.5).
8. **Records provenance**: who uploaded it, when, the file hash, the pyKES and
   app versions carried in the file's `version` dict.
9. **Stores the original** under its hash.

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

The table is shown here as `experiments` for clarity; §5 generalises it to
`entities`, which is what should actually be built — the columns below are
unchanged by that.

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
  leaf_name       TEXT,          -- 'Synthesis temperature [degC]'
  path            TEXT,          -- 'catalyst_batch/precursor/…' — NULL if own (§5.3)
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

## 5. References between entries

A photocatalysis experiment is not a self-contained record. `ABC-67` was run on
catalyst batch `ABC-12`, which was photodeposited at 360 nm from precursor
`BC-2`, which was synthesised at 1150 °C. The question worth answering is
*"find me the photocatalytic tests of samples synthesised at 1150 °C **and**
photodeposited at 360 nm"* — and answering it means the metadata of the whole
chain has to be reachable from the experiment.

This has to be generic: any entry may reference any other, to any depth, in any
combination.

### 5.1 One table for everything, not one per kind

The single change that makes this generic is to stop calling the table
`experiments`. It becomes `entities`, with an `entity_type`:

```sql
CREATE TABLE entities (
  entity_id     TEXT PRIMARY KEY,        -- ABC-67, ABC-12, BC-2 …
  entity_type   TEXT NOT NULL,           -- experiment | catalyst_batch | precursor | …
  metadata      TEXT NOT NULL,           -- JSON: this entry's OWN metadata
  effective     TEXT NOT NULL,           -- JSON: own + everything inherited
  payload_path  TEXT,                    -- NULL for entries that carry no data
  …                                      -- provenance columns as before
);

CREATE TABLE edges (
  source   TEXT NOT NULL REFERENCES entities(entity_id),
  target   TEXT NOT NULL,
  role     TEXT NOT NULL,                -- catalyst_batch, precursor, source …
  resolved INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (source, role, target)
);
CREATE INDEX idx_edges_target ON edges(target);
```

A photocatalysis experiment is an entity that happens to have a payload. A
catalyst batch is an entity that happens not to. There is one identity space,
one metadata mechanism, one search — and a catalyst batch becomes searchable in
its own right, for free. Nothing in the design knows what a "precursor" is; new
kinds of entry need no code.

### 5.2 References are declared, never guessed

A reference is a metadata field whose *value* is another entity's ID. Which
fields those are is declared by the uploaded file, in the same place and the same
style as the index mapping (§7):

```python
plotting_instruction['reference_instructions'] = {
    'Catalyst batch': {'role': 'catalyst_batch'},
    'Precursor':      {'role': 'precursor'},
}
```

Declaring rather than inferring matters. Scanning every metadata value for
something that looks like an entity ID would link `Lot: BC-2` to a precursor by
accident, and the failure would be invisible.

### 5.3 The merge: qualified by the path that reached it

The effective metadata of an entity is its own, plus the effective metadata of
everything it references, each inherited key **prefixed with the role path that
reached it**:

```
ABC-67  own        Irradiance [mW/cm2]                                    = 50
        1 hop      catalyst_batch/Photodeposition wavelength [nm]         = 360
        2 hops     catalyst_batch/precursor/Synthesis temperature [degC]  = 1150
        3 hops     catalyst_batch/precursor/source/Supplier               = …
```

Qualification is not decoration — it is what makes the merge **incapable of
collision**. An experiment with its own `Temperature [degC]` and a precursor with
its own `Temperature [degC]` produce `Temperature [degC]` and
`catalyst_batch/precursor/Temperature [degC]`: two distinct, unambiguous keys.
A flat merge would have to pick one and silently discard the other, which for a
scientific record is not an acceptable failure mode.

It also keeps provenance in the key itself: reading
`catalyst_batch/precursor/Synthesis temperature [degC]` tells you exactly which
entry the 1150 °C came from and how it was reached.

**Searching by leaf name.** Users think "Synthesis temperature", not
"catalyst_batch/precursor/Synthesis temperature". The key registry (§4.3)
therefore stores each key's leaf name alongside its full path, and the facet UI
groups by leaf. When a leaf occurs at exactly one path — the common case — the
user never sees the path at all. When it occurs at several, the facet
disambiguates and the query ORs across them (measured at 32.8 ms).

### 5.4 Materialise, don't resolve at query time

Two ways to answer a query over inherited metadata:

| Approach | The target query | Notes |
| --- | --- | --- |
| Recursive CTE at query time | **92.3 ms**, for *one* inherited predicate | always fresh; each additional inherited predicate needs another join, so it degrades fast |
| **Materialise `effective` at ingestion** | **32.9 ms** for the full three-predicate query — **2.7 ms** with hot keys promoted (§5.6) | must be recomputed when an ancestor changes |

Materialisation wins clearly, and by more than the numbers suggest: the CTE
figure is for a single predicate, while the materialised figure is for the whole
"1150 °C **and** 360 nm **and** active" query. Resolving the full graph for
12 340 entities takes **165 ms**, and writing the results **466 ms** — small
enough that a full rebuild is never a problem.

### 5.5 Keeping it correct

Four cases decide whether this works in practice rather than in a demo.

**Forward references.** `ABC-67` will sometimes be uploaded before `ABC-12`
exists — people upload in whatever order suits them. The edge is recorded with
`resolved = 0`, effective metadata is computed from whatever exists, and when
`ABC-12` arrives everything referencing it is recomputed. Rejecting the upload
instead would make the system unusable.

**Invalidation.** When `BC-2` is corrected, every entity that transitively
references it must be recomputed. The dependents are found by walking `edges`
backwards — measured at **0.2 ms** to find the 27 dependents of one precursor, so
this is free at any realistic rate of correction.

**Cycles.** A mistake can make `ABC-12` reference `BC-2` reference `ABC-12`.
Resolution carries a visited set and a depth cap, records the cycle, and flags it
on the admin page rather than looping.

**Multi-valued references — the trap.** If an experiment references two catalyst
batches *both under the role* `catalyst_batch`, the second silently overwrites
the first, because they produce identical qualified keys. Either the roles must
be distinct (`catalyst_batch_a`, `catalyst_batch_b`) or the path must carry an
index (`catalyst_batch[0]/…`). **Ingestion must reject a repeated role rather
than accept it**, because the resulting data loss is invisible.

### 5.6 Hot keys, and one measurement that changes the guidance

Filtering an inherited key through JSON is a full scan, and the `effective` blob
is larger than the raw metadata, so scans cost more than in the flat design:
**39.5 ms** for one predicate. Promoting that key to an indexed generated column
takes it to **0.4 ms** — roughly a hundredfold.

Two findings that are not obvious, and that cost real time if discovered later:

**Once one selective index narrows the set, remaining JSON predicates are free.**
Two promoted keys plus a plain `json_extract` boolean ran in **2.6 ms** — the
same as the two promoted keys alone. There is no need to promote everything; one
selective predicate is enough to make the rest cheap.

**Promote selective keys, and run `ANALYZE`.** Promoting the low-cardinality
`Active` boolean *and skipping `ANALYZE`* made the same query **17.1 ms**,
because SQLite chose the useless boolean index over the selective range one.
Running `ANALYZE` restored the correct plan and **2.7 ms**. `ANALYZE` after every
index creation belongs in the migration, not in a troubleshooting note.

### 5.7 Where the non-experiment entries come from

This requirement quietly introduces a second kind of contributor. The person who
synthesised precursor `BC-2` has no raw traces, runs no processing, and will
never open the processing app — but their metadata is what the search depends on.
Three routes, all worth supporting:

* **An entity sheet upload** — an Excel or CSV of entities of one type, with an
  ID column and whatever metadata columns exist. This is the route the synthesis
  people will actually use.
* **Metadata-only entities inside an HDF5**, for anyone already producing one.
  This works today with no change: `Experiment` accepts empty `raw_data` and
  `processed_data` dicts.
* **Direct creation and editing in the app**, for corrections.

### 5.8 The graph is worth exploring in itself

Once the edges exist, the reverse direction answers questions the group cannot
currently ask at all: *"show me every photocatalytic test ever run on material
descended from precursor `BC-2`"*, or *"which precursors have we never tested
above 100 mW/cm²"*. An entity page showing what an entry references and what
references it — one hop each way, expandable — is a small amount of UI on top of
a table that already exists.

### 5.9 One role, several kinds of entry

Nothing here constrains what a reference points at: an edge is
`(source, role, target)` with no target-type column, and the merge qualifies
inherited keys by the **role**, never by the kind of thing the role reached. So
`catalyst_batch/Photodeposition wavelength [nm]` means "of whatever this
experiment used as its catalyst batch", and one filter spans every kind of batch
without any union logic.

That makes accepting several kinds under one role a matter of declaration rather
than of storage. It is designed but not built; see
[docs/database_extensions.md](database_extensions.md) §1.

---

## 6. Does it hold at 10 000 experiments?

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

With the reference graph of §5 layered on — 10 000 experiments, 2 000 catalyst
batches, 300 precursors and 40 source chemicals, chained four deep — the same
kind of query still lands well inside the interactive budget:

| Operation, with inherited metadata | Time |
| --- | --- |
| Resolve the effective metadata of all 12 340 entities | 165 ms |
| Write it back | 466 ms |
| **The target query** — synthesis T > 1100 °C *and* photodeposition 300–400 nm *and* active | **32.9 ms** |
| The same, with the two inherited keys promoted and `ANALYZE` run | **2.7 ms** |
| Filter by leaf name, ORed across every matching path | 32.8 ms |
| Same question via recursive CTE, no materialisation, *one* predicate | 92.3 ms |
| Find every dependent of one precursor (invalidation fan-out) | **0.2 ms** |
| Index size: own metadata only → with inherited materialised | 7.5 MB → **24.0 MB** (3.18×) |

The inheritance costs a 3.18× larger index and roughly doubles the scan time of
an unpromoted filter — 24 MB and 33 ms, both comfortably irrelevant.

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

## 7. The index mapping travels in the HDF5 file

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

`reference_instructions` (§5.2) lives in the same dictionary and follows the
same rules — declared by the file, applied at ingestion, registered with
provenance.

Because the original uploads are kept (§2), a mapping mistake is recoverable:
fix the default mapping and re-ingest from `uploads/` without asking anyone to
re-upload anything.

---

## 8. Authentication and privacy

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

## 9. Searching, visualizing, and the pages

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

**Entity page** — for any entry, experiment or not: its own metadata, what it
references, and what references it, one hop each way and expandable. This is
where *"every photocatalytic test ever run on material descended from `BC-2`"*
gets answered, and it is a small amount of UI over a table that already exists.

**Admin** covers the things a growing schema needs: the key registry with
aliasing, result-key conflicts, dangling and cyclic references, the upload log,
and a "rebuild from uploads" action.

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

## 10. What the server needs

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

## 11. Phasing

**Phase 0 — decide and provision.** Pick the authentication route (§8.2) and the
collision policy (§12); provision the server with TLS and a working
nginx + Streamlit + WebSocket configuration; confirm `Remote-User` reaches
`st.context.headers` through the WebSocket upgrade. *The auth decision gates the
deployment, not the code — everything in Phase 1 can proceed in parallel.*

**Phase 1 — the data layer.** A new `pyKES/database/index.py`: the SQLite schema,
`ingest_upload`, payload splitting with gzip, the mapping application, the two
registries, and `rebuild_from_uploads`. Round-trip tests against synthetic
datasets with known contents, plus an explicit test that a second upload
introducing new metadata keys is absorbed without migration.

**Phase 2 — the reference graph.** `entities` and `edges`, the declared
reference mapping, qualified resolution with cycle and depth guards, forward
references, dependent recomputation, and the entity-sheet upload for
metadata-only entries. Tests must cover a forward reference resolved by a later
upload, a diamond, a cycle, and a repeated role being rejected.

**Phase 3 — the database app.** Browse & Search against SQLite, experiment
detail, pagination, facets grouped by leaf name, the entity page.

**Phase 4 — the upload path.** Validation, collision handling, ingestion with
progress, the upload log.

**Phase 5 — deployment.** Auth, TLS, `internal` payload serving, backups,
and the hardening list in §8.4.

**Phase 6 — the rest.** Property maps, the admin page with key aliasing, subset
export, provenance dashboards.

Phases 1–5 are the minimum that delivers a private, searchable, uploadable
database. Phase 6 is what makes it worth more than the files it was built from.

---

## 12. Decisions needed, and open risks

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
5. **What are the entity types, and who owns each?** `experiment`,
   `catalyst_batch`, `precursor`, `source_chemical` is the chain in the worked
   example, but the design does not care — the list is a group convention. It
   needs agreeing early, because the reference roles are what the qualified
   metadata keys are named after, and renaming a role rewrites every
   descendant's keys.
6. **Who may edit a shared ancestor?** Correcting one precursor silently changes
   the effective metadata of every experiment descended from it. That is the
   point of the feature, but it means edits to widely-referenced entries deserve
   more care — and probably an audit note — than edits to a single experiment.

**Risks:**

* **Metadata key drift** — two spellings of one field, created silently. The
  registry makes it visible and aliasable, but somebody has to look at the admin
  page occasionally. This is the most likely way the search quality degrades
  over years.
* **Streamlit is not a hardened multi-user framework.** The proxy gate in §8.1 is
  load-bearing, not defence in depth. Do not move authentication into the app
  for convenience.
* **Self-hosting is now the group's responsibility** — TLS renewal, OS updates,
  backups. A private server that nobody patches is not more secure than a
  managed one; it is less.
* **Scale figures come from raw-data-only fixtures.** The 96.3 KB and 14.8 ms
  per experiment were measured on files with no `processed_data`. Expect 3–5× on
  real files: still comfortable, but measure a real processed batch in Phase 1
  before sizing the disk.
* **A repeated reference role loses data silently.** Two references under one
  role produce identical qualified keys and the second overwrites the first.
  Ingestion must reject this rather than accept it (§5.5); it is the one failure
  mode here that a user would never notice.
* **Renaming an entity ID orphans its references.** Edges are stored by ID, and a
  forward reference to an ID that never arrives looks identical to a typo. The
  admin page must list dangling references, or they accumulate unnoticed.
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

Measurements in §2, §3.3, §4.4, §5 and §6 were taken in this repository against
`src/tests/data/260507_Complete.h5` and against a synthetic 10 000-experiment
index built with an intentionally evolving metadata schema, and against a
synthetic 12 340-entity reference graph chained four levels deep.
