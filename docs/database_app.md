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
| **Contribute** | Upload a batch or sheet, add one entry by form, download templates |
| **Admin** | Reference health, key drift, result conflicts, the upload log |

### Browse — search, then compare

Comparing a subset is the main thing the database is for: every test of one
catalyst, every catalyst descended from one semiconductor under one set of
conditions. So the page is built around narrowing to a subset and then plotting
it.

Every filter is generated from the metadata-key registry, scoped to the kind of
entry being searched. Nothing is configured by hand: a spreadsheet column that
first appears in next year's uploads gets a working filter as soon as it has
been ingested, and the widget follows from the type the registry observed —
a slider for numbers, a multiselect for low-cardinality text, a contains box for
free text.

**Filters are grouped by the entity they describe**, ordered by how far away
that entity sits: the entry's own metadata first, then one reference away, then
two. That is the order somebody narrowing a search thinks in — what was done in
this experiment, then what it was made from — and naming each group after the
thing it describes (*Finished semiconductor*) says more than counting hops
(*two references away*) ever did.

The entry's own fields are always open, since that is where a search starts.
Each referenced entity is a collapsed group, so a chain four entities deep does
not bury the fields somebody came to filter on. Nothing is hidden behind a
"more filters" fold: every group is there, one click from open, because the
inherited fields are exactly what a comparison is usually built from.

**Reset All Filters** clears every filter and the free-text term at once. It
runs as a widget callback rather than inline, because Streamlit refuses to have
a widget's state assigned after that widget has been drawn, and it clears the
query string with them — the seeding reads the URL, so a reset that left it in
place would restore what it had just cleared.

**Once the filtered set is small enough, its traces are plotted.** Below
`MAX_COMPARISON_ENTRIES` (50) matches the *whole* matching set — not just the
visible page — is loaded and drawn by the same
`time_series_component` the processing app uses. The curve picker offers
whatever the payloads declare. A second multiselect narrows what is drawn
without touching the search, so a curve can be taken out of the plot while the
result table stays as it is.

The filter state is written into the query string **and read back from it**, so
a search is a link that reproduces the same subset and the same comparison. So
is an entry: `/Entity?entity=EXP-0001` opens it directly.

Every widget whose state has to outlive that round trip carries an explicit
key. Streamlit derives an unkeyed widget's identity from its arguments, so a
search box seeded with `value=` from the query string becomes a *different*
widget as soon as the search has been written into the URL — and comes back
holding the previous term instead of what was just typed. The key pins the
identity; the URL seeds it once, and after that the user's own typing wins.

**Table columns** are chosen from the same registry. A column's header is the
field name alone and its reference chain is the header's tooltip: a header
carrying the whole chain — `Synthesis temperature [°C] · via Catalyst batch ›
Finished semiconductor` — is wider than the table, so the next chosen column
lands off-screen and the selection looks like it did nothing. Two chosen columns
sharing a field name are told apart by the entity they came from, which is also
what stops one silently overwriting the other. Chosen columns sit directly after
the identifier, ahead of kind, group and owner, so a column somebody just asked
for is visible without scrolling sideways.

### Entity

A type-ahead finds an entry without the whole identifier having to be
remembered: typing `NB-6` offers every entry that starts that way, prefix
matches first.

The type-ahead's widgets are keyed by the entry currently open. That is what
makes the reference buttons work at all: under a fixed key the box kept whatever
was last searched for, so every click navigated to the new entry and was then
sent straight back by the stale search term, and the reference links looked
dead.

Own metadata, inherited metadata and results are each a collapsed expander, so
the page opens on what an entry *is* and its references rather than on a wall of
fields. The inherited table carries the path each value arrived through, so the
provenance of every number is visible without leaving the page.

A metadata table holds one column of values spanning every field, so it mixes
numbers, text and booleans — the mixture Arrow refuses to serialise, which
Streamlit reports as a console traceback and then silently repairs. Only a
column pandas could not type is rendered as text (`arrow_safe_frame`), so a
column of numbers keeps its alignment, its sort and Streamlit's own formatting.

The references panel works both ways. Downwards it is what the entry was made
from; upwards it answers the question the group could not previously ask at all
— opening a precursor lists every semiconductor made from it, and from there
every batch and every experiment.

An entry with a payload gets its traces plotted by the same panel the browse
page uses, with the entry selection switched off — comparing entries belongs on
the browse page, where the filters decide the subset. The payload can be
downloaded as a standalone pyKES HDF5 file, which loads in the processing app
unchanged.

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

Three routes, all attributed to the signed-in user:

* **Measured batch (HDF5)** — what the processing app produces.
* **Metadata sheet (Excel or CSV)** — one row per entry, for the catalyst
  batches, semiconductors and precursors whose makers have no raw traces.
* **A single entry through a form**, for one batch or chemical without making a
  spreadsheet for it.

All three are checked against the entity schemas (below), and the sidebar offers
the **Excel template** for each kind of entry, generated from the same schema
the upload will be checked against so the two cannot drift apart. The template's
second sheet says what each column expects, since a template of bare headings
gets filled in wrongly.

The form is written as a one-row sheet and ingested through the ordinary sheet
route. That is not a detour: it means a form entry is stored, validated,
versioned and **rebuildable** exactly like an uploaded one. A form that wrote
straight to the database would produce entries `rebuild_index` could not
reconstruct.

Uploading never overwrites. A name already in the database is kept alongside the
newcomer under a version suffix, and the page says so. Re-uploading a
byte-identical file does nothing.

## Entity schemas

`src/pyKES/database/entity_schemas/*.yaml` — one hand-editable file per entity
type, saying which metadata fields are expected, their types, and the options a
choice field allows. They are the group's own vocabulary rather than application
code, so they carry comments and are read at runtime.

```yaml
fields:
  - name: Measured Analyte [O2 or H2]
    type: select
    options: [O2, H2]
    required: true

  - name: Catalyst Batch [experiment no.]
    type: reference
    role: catalyst_batch
    required: true
```

One file drives four things, which is the reason to have it: the contribution
form's widgets, the Excel template, the validation an upload is checked against,
and the **reference declarations** — a field of type `reference` already says
which column links to what, so `GROUP_REFERENCE_INSTRUCTIONS` is derived from
the schemas rather than written a second time that could disagree.

Field types: `text`, `number`, `integer`, `boolean`, `select`, `multiselect`,
`date`, `reference`, `mapping`. A deployment maintaining its own copy points
`DatabaseAppConfig.schema_directory` at it.

### References that accept more than one kind of entry

A catalyst batch may be an ordinary one or a `modified_catalyst_batch` — one
that was recoated, re-reduced or washed after it was made. The experiment does
not have to know which:

```yaml
  - name: Catalyst Batch [experiment no.]
    type: reference
    role: catalyst_batch
    accepts: [catalyst_batch, modified_catalyst_batch]
```

Most of this needed nothing new. An edge is `(source, role, target)` with no
target-type column, and **inherited keys are qualified by the role, not by the
kind of entry the role reached**. `catalyst_batch/Photodeposition wavelength
[nm]` means "of whatever this experiment used as its catalyst batch", so one
filter spans both kinds with no union logic.

`accepts` is guidance and diagnosis, never a storage constraint. It decides
which entries the contribution form's picker offers, and a link to a kind it
does not name is **reported on the admin page, not refused**: the metadata still
merges, because a wrong-kind reference is a labelling mistake and discarding the
values would hide it rather than show it. The check runs over resolved edges
rather than at upload, since an entry may legitimately name a target that has
not been uploaded yet — refusing an upload for a fact not yet knowable would
break the forward references the ingestion order depends on.

Leaving `accepts` out means the reference is not checked. That is deliberately
not the same as defaulting it to the role: a role names the *relationship*, so
`precursor_chemical_a` is filled by a `precursor_chemical`, and reading the role
as a type would flag every correct reference in the group's own chain.

**One field, several paths.** A modified batch adds a hop, so the semiconductor
sits two references away for experiments run on an ordinary batch and three away
for the rest. Those are the same field of the same entity, so `build_facets`
merges them into one filter that matches whichever path an entry has
(`COALESCE` over both, in `build_expression`). Without that, a filter on
synthesis temperature would answer for half the experiments and say nothing
about the other half — a silently wrong answer, which is worse than no filter.
The grouping is by the *last* role and the field name, not by the name alone, so
a batch's `Notes` and a semiconductor's `Notes` stay apart.

### Metadata that is a set of named numbers

Dopants, cocatalysts and synthesis profiles are not one number:

```yaml
  - name: Dopants [mol%]
    type: mapping
    key_label: Dopant
    value_label: mol%
    key_options: [Ir, Ru, Cr, Rh, La, Sb, Ta, Sr]
```

This replaces the lettered `Co-catalyst A` / `Co-catalyst B` slots, which fixed
the count in advance and made *"everything with at least 0.02 wt% Cr"* depend on
which slot somebody happened to fill.

**In a sheet** a mapping is one cell of `name=value` pairs — `Ir=0.02; Ru=0.02;
Cr=0.03`. Both `=` and `:` separate a name from its value, because people write
both; pairs are separated by a semicolon or a newline, not a comma, which would
be ambiguous wherever Excel writes a decimal comma. That form was chosen over
JSON in a cell, which is precise but a quoting trap for someone typing into
Excel. A pair that cannot be read is an **error**, never a silent skip: for a
composition, dropping one is the difference between "no dopant" and "a dopant we
lost". The downloaded template's field guide carries an example, since this is
the one column whose format cannot be guessed from its name.

**In the form** it is an editable table with dynamic rows, written back as the
same `name=value` text — so a form entry and an uploaded row produce identical
metadata and neither is a special case afterwards.

**In the filter** it is two levels: a picker of the names actually present, and
then a slider per chosen name over the range that name spans. Choosing a name
and leaving its slider alone is already a filter — it asks for entries carrying
that dopant at all — which is what picking it from the list means. Any number of
names can be chosen; each adds a predicate.

Storage needed no change. `coerce_index_value` already recursed into
dictionaries, so a mapping round-trips through the `effective` column and is
inherited like any other value. What did need adding: `infer_value_type`
classified a dictionary as text, which would have offered stringified
dictionaries as a multiselect; the registry now records the union of the names
seen (`metadata_keys.sub_keys`), which is what fills the dropdown; and `Filter`
gained a `sub_key`, as its own field rather than a second separator stacked into
the key string — that is how the `__SLASH__` bug happened once already.

### Derived scalars

A temperature profile is a *sequence*, and the questions asked of it are
aggregates — the peak reached, the time held there — which no per-name slider
expresses. So a mapping field can declare scalars computed from it at ingestion:

```yaml
    derived:
      - name: Peak temperature [°C]
        of: max_key
      - name: Time at peak temperature [h]
        of: value_at_max_key
```

They are stored as ordinary metadata, so each gets a slider, a table column and
inheritance without any further work. `DERIVED_FUNCTIONS` in `entity_schema.py`
is where more go. A mapping whose names are not numbers — a set of dopants —
derives nothing rather than deriving zero, which would be a measurement nobody
made.

### What validation does and does not do

A schema says what is **expected**, not what is **allowed**. That distinction is
the whole design:

* A **declared field filled in wrongly is an error** — missing when required, a
  value outside the options, a number that is not one. The batch is checked
  before anything is written, so a file with one bad row is refused whole rather
  than leaving half its experiments in the database.
* A **field nobody declared is reported, then accepted**. Absorbing metadata
  that did not exist when the schema was written is what the database is for; a
  schema that rejected it would defeat the thing it is protecting. The upload
  page lists the undeclared fields so a typo is visible.

**A rebuild is not re-validated.** Files in the upload store were checked
against the schema in force when they arrived and accepted. Re-checking them
against today's would make every historical upload un-rebuildable the moment a
field is made required — which would destroy the guarantee the upload store
exists to provide.

### Admin

The things a growing free-form schema needs a person to look at, because
resolving them automatically would mean guessing:

* **Dangling references** — an edge whose target never arrived is
  indistinguishable from a typo.
* **Wrong-kind references** — a link to a kind of entry its field does not
  accept. Recomputed from the schemas on each view rather than stored, so
  editing `accepts` in a YAML file is reflected immediately.
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

## Styling

The palette is Streamlit's own theming, in `.streamlit/config.toml`: a dark base
with green as the one accent, so green means *active, primary or healthy* rather
than decoration. `database_app/styling.py` adds only what configuration cannot
express — the typeface and a little spacing and weight — as one stylesheet
applied at the top of every page. No custom components and no markup injected
around widgets, so a Streamlit upgrade cannot silently break the layout.

Two things that stylesheet must not do, both found by running it:

* The font rule must not select `[class*="st-"]`. That also matches Streamlit's
  Material icon spans, whose glyphs are ligatures of their own font, and
  overriding it renders every expander with the literal text
  `keyboard_arrow_right` instead of a caret.
* The font is loaded from Google Fonts with a full system stack behind it. On an
  instrument network that cannot reach the font host the page still renders
  correctly, which is not hypothetical — it happened here.

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
* Comparison colours come from each experiment's own `color` metadata, as the
  shared plotting component has always done. A batch whose entries all carry the
  same colour plots as one indistinguishable band; colouring a comparison by a
  chosen metadata field instead would be a better default.
* **Inherited metadata is still named by the route that reached it**, so a
  modified catalyst batch puts the semiconductor's fields at a second depth and
  `merge_key_paths` folds them back together for display. That fold is a patch
  over the naming scheme rather than a fix, and it cannot express several
  references of one kind in one field.
  [docs/referencing_redesign.md](referencing_redesign.md) plans naming them by
  contributor type instead — and records that facet bounds already cost 1130 ms
  per page load at ten thousand entries, which is a live problem either way.
* Promoting a mapping's sub-key to a generated column, which is what sorting the
  results table by one dopant's concentration would need. The measurement in
  [docs/database_extensions.md](database_extensions.md) §2.9 applies unchanged:
  promote the two or three that matter, then run `ANALYZE`.
