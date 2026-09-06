# The database index

How `pyKES.database.index_*` turns uploaded HDF5 batches and metadata sheets
into one searchable, cross-referenced database. This is Phase 1 of
[photocatalytic_database.md](photocatalytic_database.md); see
[server_provisioning.md](server_provisioning.md) for the machine it runs on.

Four modules, each with one job:

| Module | Job |
| --- | --- |
| `index_schema` | the SQLite tables, the filesystem layout, the fixed vocabulary |
| `index_registry` | value coercion, and the two registries that make a growing schema manageable |
| `index_references` | edges between entries, and the metadata inherited along them |
| `index_ingest` | uploads in, entities out |

---

## 1. Why an entity table rather than an experiment table

A photocatalysis experiment is not a self-contained record. It names the
catalyst batch it used; that batch names the finished semiconductor it was
deposited on; that semiconductor names the precursor chemicals it was made
from. To answer *"tests of samples synthesised at 1150 °C and photodeposited at
365 nm"*, the metadata of the whole chain has to be reachable from the
experiment.

The change that makes this generic is that the table is not called
`experiments`. It is `entities`, with an `entity_type`:

```python
ENTITY_TYPES = ("experiment", "catalyst_batch", "finished_semiconductor",
                "precursor_semiconductor", "precursor_chemical",
                "commercial_chemical", "stock_solution", "other_entity")
```

An experiment is an entity that happens to carry a payload file; a precursor
chemical is one that happens not to. **Nothing in these modules knows what a
precursor is** — adding a kind of entry means adding a string to that tuple.

## 2. Why metadata is JSON

Experiments run next year will carry fields that do not exist today. A wide
table would need a migration per field and would grow steadily sparser; an
entity-attribute-value table would turn every query into self-joins. So the
columns the application itself depends on are typed, and everything scientific
lives in three JSON columns:

| Column | Holds |
| --- | --- |
| `metadata` | the entry's own metadata |
| `results` | scalar results, resolved through the upload's declared mapping |
| `effective` | `metadata` plus everything inherited, qualified — **what searches read** |

A new metadata column needs no migration at all. It appears in the JSON of the
entries that have it, and in the key registry, and a facet for it follows.

---

## 3. The reference graph

### 3.1 References are declared, never guessed

A reference is a metadata field whose *value* is another entity's id. Which
fields those are is declared by the uploaded file, in `plotting_instruction`
alongside the result mapping:

```python
plotting_instruction['reference_instructions'] = {
    'Catalyst Batch [experiment no.]': {'role': 'catalyst_batch'},
}
```

Scanning every value for something that looks like an id would link a lot number
to a precursor by accident, and the mistake would be invisible. So it is
declared.

### 3.2 Inherited keys are qualified by the path that reached them

`resolve_effective_metadata` merges an entry's own metadata with everything it
references, prefixing each inherited key with the role it came through,
recursively:

```
EA-555   own      Irradiance A [mW/cm2]                                   = 12
         1 hop    catalyst_batch::Photodeposition wavelength [nm]          = 365
         2 hops   catalyst_batch::finished_semiconductor::
                      Synthesis temperature [°C]                          = 1150
```

Qualification is not decoration — it makes the merge **incapable of collision**.
An experiment with its own `Temperature [°C]` and a precursor with its own
`Temperature [°C]` produce two distinct keys, because own keys stay bare and
inherited keys always carry a prefix. A flat merge would have to pick one and
silently discard the other, which for a scientific record is not an acceptable
failure mode. The path also carries the provenance: the key itself says which
entry the value came from.

Display hints (`color`, `group`) are excluded from inheritance. A precursor's
plotting colour says nothing about an experiment made from it.

### 3.3 Materialised, not resolved per query

`effective` is computed at ingestion and stored. Answering the same question
with a recursive CTE was measured at 92 ms for a *single* predicate and needs
another join per additional one; reading the materialised column answers the
whole three-predicate query in 33 ms. The cost is that descendants must be
recomputed when an ancestor changes, which `find_dependents` locates in 0.2 ms.

### 3.4 The four cases that would otherwise fail silently

**Forward references.** An experiment routinely arrives before the catalyst batch
it names — people upload in whatever order suits them. The edge is recorded with
`resolved = 0`, the effective metadata is computed from whatever exists, and
`resolve_pending_references` promotes it when the target appears, recomputing
everything affected. `read_dangling_references` lists edges whose target never
arrived, because a forward reference that is never satisfied is indistinguishable
from a typo.

**Repeated roles.** Two references under one role would produce identical
qualified keys and the second would overwrite the first. `extract_references`
raises `ReferenceError`, and the `edges` primary key `(source, role)` makes it
impossible at the storage layer too. The group's own semiconductor sheet is
exactly this case, handled correctly because the two columns get distinct roles:

```python
{'Precursor Chemical A': {'role': 'precursor_chemical_a'},
 'Precursor Chemical B': {'role': 'precursor_chemical_b'}}
```

**Cycles.** Resolution carries a visited set and stops at `MAX_REFERENCE_DEPTH`.
`find_cyclic_entities` reports them for the admin page rather than hanging.

**Diamonds.** Two paths to the same ancestor both survive, because each carries
its own role prefix.

---

## 4. The registries

`metadata_keys` records every key ever seen: its leaf name, its role path, the
types it has carried, how often it occurs, a bounded sample of distinct values,
and which entity types it appears on. `result_keys` records every declared
result label with the path and upload that defined it.

They are maintained at ingestion, not computed on demand — walking `json_each`
over every entity was measured at 152 ms, too slow for a page view and free once
per upload. They are also where the two failure modes of a free-form schema
become visible:

* **Key drift** — `Irradiance [mW/cm2]` and `Irradiance (mW/cm2)`, created by two
  people typing two spreadsheet headers. `canonical_key` lets an admin alias one
  onto the other.
* **Type conflict** — the same key arriving as a number and as text.
  `inferred_type` becomes `'mixed'` and the facet degrades to a text filter
  rather than drawing a broken slider.

Neither is resolved by guessing, which would corrupt searches in a way nobody
would notice.

### Leaf lookup must be scoped by entity type

Users think *"Synthesis temperature"*, not
*"catalyst_batch::finished_semiconductor::Synthesis temperature"*. But in a chain
one leaf necessarily appears at **one path per entity type** — bare on the
semiconductor, one hop away on the batch, two hops away on the experiment — so
an unscoped lookup returns all three:

```python
read_metadata_keys(connection, leaf_name='Synthesis temperature [°C]')
# 3 rows

read_metadata_keys(connection, leaf_name='Synthesis temperature [°C]',
                   entity_type='experiment')
# 1 row: catalyst_batch::finished_semiconductor::Synthesis temperature [°C]
```

A facet built for the experiment search passes `entity_type` and gets the single
path it needs.

---

## 5. Ingestion

### 5.1 Two kinds of upload

`ingest_hdf5_upload` takes what the processing app produces: a dataset of
experiments with metadata, raw data and processed data.

`ingest_entity_sheet` takes an Excel or CSV of entries that carry metadata but
no measurements. This route matters more than it looks: **the person who
synthesised a precursor has no raw traces and will never open the processing
app**, yet their metadata is what the search depends on.

Both write into the same `entities` table and go through the same steps: hash,
validate, resolve collisions, split payloads, apply the mapping, register keys,
record references, recompute what changed.

### 5.2 The result mapping, with a free fallback

An upload declares `plotting_instruction['index_instructions']`. Files written
before that existed fall back to `results_table_instructions`, which already has
exactly the required shape — so the group's existing datasets contribute their
max rates, quantum yields and efficiencies with no change at all.

Resolution is permissive: a path that does not resolve for a given experiment is
simply absent from its results, which is correct when one batch mixes
liquid-phase and gas-phase runs measuring different analytes.

A label redefined with a different path is not overwritten. Both are kept and
the label is marked `conflicting`, because silently adopting the new path would
change the meaning of a column for every entity already stored.

### 5.3 Collisions are versioned, never overwritten

`allocate_entity_id` gives the newcomer `EA-555__v2`. Both entries stay. The
group's rule is that correcting a mistake is a deliberate edit by the owner or
an admin — `update_entity_metadata` — not a side effect of uploading a file
twice. Re-uploading a byte-identical file is a no-op, detected by hash.

### 5.4 Permissions

`may_edit` implements owner-or-admin. Everyone reads everything; only the owner
of an entry or an admin may correct or delete it. `update_entity_metadata`
raises `PermissionError` otherwise, and returns the list of entities whose
effective metadata it moved — which for a widely-referenced precursor can be
long, and is exactly why editing a shared ancestor is restricted.

### 5.5 Rebuilding

Every upload is stored verbatim under its hash and kept indefinitely.
`rebuild_index` re-ingests them all from scratch. This is the repair path for a
changed index schema or a mapping that turned out to be wrong, and it is why the
uploads tier exists.

---

## 6. Measured behaviour

Against the group's real 44-experiment plate dataset and the three metadata
sheets:

| Quantity | Measured |
| --- | --- |
| Ingestion, end to end | **30 ms per experiment** (a 44-experiment batch in 1.3 s) |
| Payload per experiment | 136 KB |
| gzip saving on this data | **1.05×** — see below |
| Target query, two inheritance hops | resolves to the right experiment |

Ingestion at 30 ms per experiment is what makes a synchronous progress bar
viable: no job queue, no worker, no broker.

The compression figure is worth recording because it contradicts an earlier
estimate. Writing one payload per experiment takes 6.30 MB to 5.99 MB — not the
1.7× measured on an older single-file fixture — because per-file HDF5 overhead
across 44 separate files offsets most of the saving on high-entropy sensor
traces. **Budget ~140 KB per experiment** rather than assuming compression will
help; 10 000 experiments then come to roughly 1.4 GB of payloads.

---

## 7. Worked example

The three sheets and the plate dataset the group supplied, uploaded
deliberately out of order — experiments first, then batches, then semiconductors
— so that forward references are exercised:

```python
from pyKES.database.index_schema import IndexPaths, open_index, analyse_index
from pyKES.database.index_ingest import ingest_hdf5_upload, ingest_entity_sheet

paths = IndexPaths(root='/srv/photocat/data')
connection = open_index(paths)

ingest_hdf5_upload(connection, paths, '260903_AE857_AE859.h5', uploaded_by='ae')

ingest_entity_sheet(
    connection, paths, '260507_SrTiO3_Photocatalysis_Overview_2.xlsx',
    entity_type='experiment', uploaded_by='nb',
    reference_instructions={
        'Catalyst Batch [experiment no.]': {'role': 'catalyst_batch'}})

ingest_entity_sheet(
    connection, paths, '260906_Catalyst_Overview.xlsx',
    entity_type='catalyst_batch', uploaded_by='ea',
    reference_instructions={
        'Finished Semiconductor': {'role': 'finished_semiconductor'}})

ingest_entity_sheet(
    connection, paths, '260906_Semiconductor_Overview.xlsx',
    entity_type='finished_semiconductor', uploaded_by='ea',
    reference_instructions={
        'Precursor Chemical A': {'role': 'precursor_chemical_a'},
        'Precursor Chemical B': {'role': 'precursor_chemical_b'}})

analyse_index(connection)
```

The question the whole design exists to answer then becomes three ordinary
predicates on one row:

```sql
SELECT entity_id FROM entities
 WHERE entity_type = 'experiment'
   AND CAST(json_extract(effective,
       '$."catalyst_batch::finished_semiconductor::Synthesis temperature [°C]"')
       AS REAL) = 1150
   AND CAST(json_extract(effective,
       '$."catalyst_batch::Photodeposition wavelength [nm]"') AS REAL) = 365;
-- EA-555
```

### One thing the group has to add

The plate dataset carries `Catalyst Batch [experiment no.]` in its metadata but
declares no `reference_instructions`, so it contributes **no edges**. That is
the design working as specified — references are declared, not inferred — but it
means the processing app must add the declaration before HDF5 uploads link into
the graph:

```python
PLOTTING_INSTRUCTIONS['reference_instructions'] = {
    'Catalyst Batch [experiment no.]': {'role': 'catalyst_batch'},
}
```

Until then, HDF5-uploaded experiments are searchable on their own metadata and
results but inherit nothing.

---

## 8. Promoting a hot key

Filtering through JSON is a full scan — fast enough at ten thousand rows, but a
frequently filtered key can be promoted to an indexed generated column:

```sql
ALTER TABLE entities ADD COLUMN synthesis_temperature REAL
  GENERATED ALWAYS AS (CAST(json_extract(effective,
    '$."catalyst_batch::finished_semiconductor::Synthesis temperature [°C]"') AS REAL))
  VIRTUAL;
CREATE INDEX idx_synthesis_temperature ON entities(synthesis_temperature);
ANALYZE;
```

Three details, each measured, each easy to get wrong:

* **`VIRTUAL`, not `STORED`.** SQLite's `ALTER TABLE` refuses to add a `STORED`
  generated column (`cannot add a STORED column`). Virtual columns index fine,
  and the index is what carries the speed-up.
* **`ANALYZE` afterwards.** Without it the planner has been measured choosing a
  low-cardinality index over a selective one and running the same query at
  17.1 ms instead of 2.7 ms. `analyse_index` does this.
* **Promote selective keys only.** Once one selective index narrows the row set,
  remaining `json_extract` predicates are effectively free — there is no need to
  promote everything.
