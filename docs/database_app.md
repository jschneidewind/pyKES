# The database application

The web application that sits on the index: Phases 3 and 4 of
[photocatalytic_database.md](photocatalytic_database.md). The data layer it
reads is documented in [database_index.md](database_index.md); the machine it
runs on in [server_provisioning.md](server_provisioning.md).

## Trying it

Nothing external is required to see it working — seed a synthetic archive whose
reference chain matches the group's own, then run it:

```bash
python -m pyKES.database_app.seed_demo --root /tmp/photocat-demo --fresh
PHOTOCAT_DATA_ROOT=/tmp/photocat-demo streamlit run src/pyKES/database_app/Home.py
```

That gives 60 experiments with real traces, 12 catalyst batches, 6 finished
semiconductors and 3 precursor chemicals, linked four levels deep. Point
`--files` at a directory of real exports to seed from those instead.

Running it directly, with no proxy in front, there is no `Remote-User` header —
so the application says so on the home page and falls back to a development
identity with admin rights. In a deployment nginx and Authelia sit in front, the
notice does not appear, and `allow_development_login` should be set to False.

## The pages

| Page | What it does |
| --- | --- |
| **Home** | What is in the database, and who you are signed in as |
| **Browse** | Search and filter; results table; CSV export |
| **Entity** | One entry: metadata, references both ways, traces, corrections |
| **Property map** | Any indexed quantity against any other, across everything |
| **Contribute** | Upload a measured batch or a metadata sheet |
| **Admin** | Reference health, key drift, result conflicts, the upload log |

### Browse

Every filter is generated from the metadata-key registry, scoped to the kind of
entry being searched. Nothing is configured by hand: a spreadsheet column that
first appears in next year's uploads gets a working filter as soon as it has
been ingested, and the widget follows from the type the registry observed —
a slider for numbers, a multiselect for low-cardinality text, a contains box for
free text.

**Inherited fields appear as ordinary filters**, captioned with the reference
path they came through. In the seeded demo, an experiment search offers 23
facets of which 13 are inherited:

```
Photodeposition wavelength [nm]     range    via catalyst_batch
Cocatalyst                          select   via catalyst_batch
Synthesis temperature [°C]          range    via catalyst_batch/finished_semiconductor
Supplier                            select   via catalyst_batch/finished_semiconductor/precursor_chemical_a
```

The filter state is written into the query string, so **a search is a URL**. So
is an entry: `/Entity?entity=EXP-0001` opens it directly.

### Entity

Own metadata and inherited metadata are shown as two tables, the inherited one
carrying the path each value arrived through — so the provenance of every number
is visible without leaving the page.

The references panel works both ways. Downwards it is what the entry was made
from; upwards it answers the question the group could not previously ask at all
— opening a precursor lists every semiconductor made from it, and from there
every batch and every experiment.

An entry with a payload gets its traces plotted and can be downloaded as a
standalone pyKES HDF5 file, which loads in the processing app unchanged.

The correction form is visible to everyone and enabled only for the owner or an
admin. Because correcting a widely-referenced entry changes the effective
metadata of everything descended from it, the number of entries the edit moved
is reported back rather than left implicit.

### Property map

Any two indexed quantities against each other, coloured by a third, over every
entry of one kind. Axes can be mapped results or metadata — including metadata
inherited three references away, which is what makes *quantum yield against the
synthesis temperature of the precursor* a two-click plot rather than a
spreadsheet exercise.

### Contribute

Two routes, both attributed to the signed-in user:

* **Measured batch (HDF5)** — what the processing app produces.
* **Metadata sheet (Excel or CSV)** — one row per entry, for the catalyst
  batches, semiconductors and precursors whose makers have no raw traces.

The reference columns for each kind of entry are declared in
`GROUP_REFERENCE_INSTRUCTIONS` in `config.py` and shown on the page before
upload, so it is clear what will and will not link.

Uploading never overwrites. A name already in the database is kept alongside the
newcomer under a version suffix, and the page says so. Re-uploading a
byte-identical file does nothing.

### Admin

The things a growing free-form schema needs a person to look at, because
resolving them automatically would mean guessing:

* **Dangling references** — an edge whose target never arrived is
  indistinguishable from a typo.
* **Cycles** — reported rather than hung on.
* **Key drift** — field names that differ only in punctuation or case, which is
  what two people typing two spreadsheet headers produces.
* **Type conflicts** — a key seen as both number and text, whose facet has
  fallen back to a text filter.
* **Result conflicts** — a label two uploads defined with different paths.

Admins can also rebuild the key registry and run `ANALYZE`.

## Configuration

`DatabaseAppConfig` holds everything a deployment varies; `config.py` also holds
`GROUP_REFERENCE_INSTRUCTIONS`, which is the one place the group's own chain is
encoded. Adding a kind of entry means adding it to `ENTITY_TYPES` in
`index_schema` and, if it references anything, an entry here.

```python
DatabaseAppConfig(
    data_root=Path("/srv/photocat/data"),   # or $PHOTOCAT_DATA_ROOT
    default_entity_type="experiment",
    page_size=50,
    allow_development_login=False,          # False in a deployment
)
```

## Two things worth knowing about the implementation

**Connections are per thread.** Streamlit runs each session's script on a thread
from a pool, and a SQLite connection may only be used by the thread that created
it. Caching one connection with `st.cache_resource` raises
`ProgrammingError: SQLite objects created in a thread can only be used in that
same thread` as soon as a second page is opened — which is exactly what happened
the first time this was run in a browser. `open_shared_index` therefore keeps a
thread-local pool; WAL mode is what makes several connections to one file cheap.

**Metadata keys never reach SQL as text.** They come from spreadsheet headers,
so they are bound as JSON path parameters — `json_extract(effective, ?)` — and a
column called `Notes'; DROP TABLE entities; --` is harmless. Sort columns are
checked against a fixed set rather than interpolated.

## What is not built yet

* Payload downloads go through Streamlit rather than the `X-Accel-Redirect`
  handoff the provisioning guide sets up. Correct and authenticated, but the
  bytes pass through Python; worth switching for large files.
* Deleting an entry is not exposed. `may_edit` decides who could, and the
  cascade to descendants would need designing before it is offered.
* Key aliasing is reported on the admin page but not yet editable there.
* Bulk correction — the `st.data_editor` patch flow from the plan — is not
  built; corrections are one field at a time on the entry page.
