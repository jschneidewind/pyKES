# Two extensions to the database

**Both of these are now built.** This document is kept as the record of why they
are shaped as they are — the reasoning, the measurements, and the alternatives
that were rejected. What they *do* is described in
[docs/database_app.md](database_app.md); what follows is why.

Two things changed between the design and the implementation, and both are
marked below where they arise:

* `accepts` is **not** defaulted to the role's own name. A role names the
  relationship, so `precursor_chemical_a` is filled by a `precursor_chemical`,
  and reading the role as a type would have flagged every correct reference in
  the group's own chain. Left out, a reference is simply not checked.
* Accepting a second kind of entry under one role puts the same field at **two
  depths**, which the design did not account for. A filter per depth would have
  answered for half the experiments each and said nothing about the other half.
  `build_facets` therefore merges the paths to one field into one filter (§1.7).

Every claim about what already works was checked by running it — against the
seeded demo archive (60 experiments, four references deep) and, where scale
matters, against a synthetic 10 000-entity index. The numbers below come from
those runs, not from estimates.

---

## 1. References that accept more than one kind of entry

### 1.1 The question

An experiment names its catalyst batch. Today that name must be a
`catalyst_batch`. The group wants it to be able to name a `catalyst_batch`
*or* a `modified_catalyst_batch` — a new kind of entry — without the experiment
having to know which, and without the search having to be told twice.

### 1.2 Most of this already works

The storage layer is already indifferent to what a reference points at. An edge
is `(source, role, target)`; there is no target-type column and no constraint on
one. Inheritance walks *roles*, not types.

Checked directly: a `modified_catalyst_batch` was created carrying fields no
catalyst batch has, an experiment's `catalyst_batch` role was pointed at it, and
the effective metadata came out as

```
catalyst_batch/Modification                                    = Cr2O3 shell
catalyst_batch/Shell thickness [nm]                            = 3.5
catalyst_batch/finished_semiconductor/Synthesis temperature [°C] = 1150
catalyst_batch/finished_semiconductor/precursor_chemical_a/Supplier = Acme
```

The chain continued through the new kind to its own references, two further hops
up. An existing filter two hops away — synthesis temperature between 1000 and
1200 °C — matched 61 experiments where it had matched 60, the extra one being
the experiment that reaches its semiconductor through the modified batch.

**The property that makes this work is that inherited keys are qualified by the
role, not by the target type.** `catalyst_batch/Photodeposition wavelength [nm]`
means "the photodeposition wavelength of whatever this experiment used as its
catalyst batch". One filter therefore spans both kinds with no union logic
anywhere, which is exactly the behaviour wanted: *every* test of a batch-like
thing photodeposited at 365 nm, however that batch was made.

### 1.3 What actually blocks it

Three things, all small, and the first is the only hard error:

1. **`ENTITY_TYPES` is a closed tuple** and `insert_entity` refuses anything
   else — the one exception raised during the probe. Adding a kind means adding
   a string there and a YAML schema file beside the other eight. That closure is
   worth keeping: it is what stops a typo in a spreadsheet header from silently
   creating a ninth kind of entry.
2. **The schema declares a role but not what may fill it.** A `reference` field
   says `role: catalyst_batch` and nothing about acceptable targets.
3. **The contribute form has nowhere to look** to decide which entries its
   reference picker should offer.

### 1.4 The declaration

```yaml
  - name: Catalyst Batch [experiment no.]
    type: reference
    role: catalyst_batch
    accepts: [catalyst_batch, modified_catalyst_batch]
```

**Changed from the design.** `accepts` was going to default to `[role]`. That is
wrong: the shipped schemas have roles like `precursor_chemical_a`, filled by a
`precursor_chemical`, so the default would have flagged every correct reference
in the group's own chain as a mismatch. Omitting `accepts` now means *not
checked*, and every shipped reference field declares its accepted kinds
explicitly instead.

### 1.5 Where the check belongs — and where it must not

`accepts` must **not** become a gate at upload time. An experiment may be
uploaded before the batch it names; forward references are recorded unresolved
and promoted when the target arrives, and that guarantee is load-bearing —
people upload in the order their work happened, not in dependency order. A type
check at ingestion would refuse an upload for a fact not yet knowable.

So the check happens where the answer exists:

- **At resolution**, when the target arrives and the edge is promoted. A target
  whose kind is not accepted is recorded as a mismatched edge and surfaced on
  the admin page, beside the dangling references that are already reported
  there. The metadata still merges — a wrong-kind reference is a *labelling*
  problem, and dropping the values would hide the mistake instead of showing it.
- **In the form**, where `accepts` decides which entries the picker offers. That
  is guidance at the moment of typing, which is where it prevents the mistake.

Storage stays type-free. That is what keeps a new kind of entry from needing a
migration, and it is the same reasoning that made the entities table generic in
the first place.

### 1.6 The cost, stated honestly

The facet list for a role becomes the **union** of the fields of every kind
reachable through it. Fields only one kind carries are sparse — present on some
rows, absent on others — which is already true today (`Notes` is absent on most
batches).

One consequence is worth knowing before it is discovered as a bug report: a
numeric field carried by only *one* entry gets no filter, because `build_facets`
skips a key whose minimum equals its maximum. Measured: `catalyst_batch/Shell
thickness [nm]` was registered as a number with one occurrence and produced no
facet, while the text field `catalyst_batch/Modification` produced a multiselect.
That rule is right — a slider with one stop is not a filter — but it means a
newly introduced kind's numeric fields stay invisible until at least two entries
differ.

### 1.7 One field, several paths — the part the design missed

A modified batch sits *between* the experiment and the ordinary batch, so it
adds a hop. The semiconductor is then two references away for experiments run on
an ordinary batch and three away for the rest, and the registry holds two keys
for one field:

```
catalyst_batch/finished_semiconductor/Synthesis temperature [°C]
catalyst_batch/catalyst_batch/finished_semiconductor/Synthesis temperature [°C]
```

Left alone that produces two filters, each answering for part of the database
and neither saying so — a silently wrong answer, which is worse than no filter
at all. `merge_key_paths` therefore collects the paths to one field into a
single facet, and `build_expression` reads whichever one an entry has with
`COALESCE`, so every operator keeps working unchanged.

The grouping is by the **last role and the field name**, not by the field name
alone. The last role names the entity the field belongs to; the rest of the path
is only how it was reached. That merges the semiconductor's synthesis
temperature at both depths while keeping a batch's `Notes` and a semiconductor's
`Notes` apart, which sharing a name is not enough to justify.

Measured on the seeded demo: the dopant filter matches 19 experiments merged,
against 16 through the shallow path alone — the three missing ones being exactly
those tested on a modified batch. The merge also *shrinks* the sidebar, from a
duplicated list back to 35 filters.

---

## 2. Structured metadata values

### 2.1 The question

Some metadata is not one number. It is an arbitrary number of name-and-number
pairs:

```
Dopants [mol%]              {'Ir': 0.02, 'Ru': 0.02, 'Cr': 0.03}
Co-catalysts [wt%]          {'Co': 0.05, 'Rh': 0.05, 'Cr': 0.02}
Temperature steps [°C, h]   {900: 0.5, 1000: 2, 1150: 10}
```

And the filter the group wants is specific: pick a dopant from a dropdown, get a
range slider for *its* concentration, and repeat for as many dopants as the
question needs.

### 2.2 Why the present workaround does not scale

The demo does what the group's sheets do — `Co-catalyst A`, `Co-catalyst A
loading [wt%]`, and a second lettered slot if needed. Three things go wrong.
The number of slots is fixed in advance, so a third co-catalyst is a schema
change. The letters carry no meaning: Cr in slot A and Cr in slot B are the same
substance, so "everything with at least 0.02 wt% Cr" has to OR across every slot
and gets a different answer if somebody fills them in a different order. And a
person reading the table cannot see at a glance what the sample actually
contains.

### 2.3 The shape: one key, a mapping value

```json
{"Dopants [mol%]": {"Ir": 0.02, "Ru": 0.02, "Cr": 0.03}}
```

Nothing in the storage layer has to change to hold this. `coerce_index_value`
already recurses into dictionaries, the value round-trips through the
`effective` JSON column, and inheritance carries it like any other value.
Checked: a dopant mapping placed on a finished semiconductor arrived intact on
an experiment two hops away, as
`catalyst_batch/finished_semiconductor/Dopants [mol%]`.

### 2.4 Filtering, measured

The filter the group described is a JSON path one level deeper than the ones the
search already builds, and it needs no new query machinery:

| What the page does | How it runs | Demo (60) | 10 000 entities |
| --- | --- | --- | --- |
| One dopant's range | `json_extract(effective, '$."<key>"."Ir"')` with the existing `between` | 0.33 ms | **7.9 ms** |
| Several dopants at once | the same predicate once per element, ANDed | verified | — |
| Filling the dropdown | `json_each` over the mapping | 0.43 ms | **23.2 ms** |
| Bounds for the slider | `MIN`/`MAX` over the sub-path | 0.4 ms | — |

Entries that do not carry the key at all contribute nothing and raise nothing —
`json_each` over a missing path returns zero rows rather than erroring, which is
what makes a sparse field safe to filter on.

23 ms to enumerate the dropdown is fine on demand but wasteful on every rerun,
so the sub-keys belong in the registry, refreshed at ingestion — the same
reasoning that put the metadata keys there rather than computing them per page
view.

### 2.5 What changed, precisely

1. **`infer_value_type` returned `text` for a mapping**, so a dopant field would
   have been offered as a multiselect of stringified dictionaries. It now
   returns `TYPE_MAPPING`.
2. **The registry records the sub-keys** it has seen, in a `sub_keys` JSON
   column on `metadata_keys` — the same bounded-sample idea `distinct_sample`
   already used. `add_missing_columns` adds it to an index that predates it, so
   an existing database keeps working without a rebuild.
3. **`Filter` gained a `sub_key`**, as its own field rather than something
   encoded into the key string. The key string already carries the role path;
   stacking a second separator into it is precisely how the `__SLASH__` bug
   happened, and it would happen again the first time a name contained the
   separator.
4. **`build_facets` emits a mapping facet** carrying the available sub-keys and
   the bounds of each.
5. **The sidebar draws it in two levels** — a multiselect of sub-keys, then one
   slider per chosen sub-key. Both are widgets the page already used, so this
   stayed inside "simple adjustments Streamlit supports natively".
6. **Tables already coped**: `format_value` renders a mapping as compact JSON,
   so a dopant field shows as `{"Ir": 0.02, "Ru": 0.02}` rather than breaking
   the table.

One thing had to be fixed that neither the design nor the first run of the
seeding anticipated: an **empty Excel cell arrives as a float NaN**, not as an
empty string, so the first pass read it as the text `nan` and reported a
filled-in field nobody had filled in. `is_blank` now covers None, whitespace and
NaN, and every blank check in the schema layer goes through it.

### 2.6 Declaring it in the Excel sheet

The constraint is that it has to fit in one cell, be typable by a person in
Excel without help, be unambiguous to parse, and survive the round trip through
the downloadable template.

| Option | Verdict |
| --- | --- |
| One column per entry (`Dopant Ir [mol%]`, `Dopant Ru [mol%]`, …) | Reintroduces fixed slots, and the column set grows without bound. |
| JSON in the cell (`{"Ir": 0.02}`) | Precise, but quoting is a trap for someone typing into Excel and Excel will not help them get it right. |
| **`name=value` pairs, semicolon-separated** | **Chosen.** `Ir=0.02; Ru=0.02; Cr=0.03`. Typable, readable, no quoting, and close to how this gets written on a whiteboard. Temperature steps become `900=0.5; 1000=2; 1150=10`. |

Parsing splits on `;` or a newline, then on the first `=` or `:`, strips both
sides, and puts the value through the same numeric coercion the rest of the
sheet uses. Both separators are accepted because people write both — verified on
one cell mixing them, `Cr=0.02; Rh:0.01`. A comma is deliberately *not* a pair
separator: it would be ambiguous wherever Excel writes a decimal comma. A pair
that does not parse is a validation **error** naming the cell — never a silent
skip, which for a composition is the difference between "no dopant" and "a
dopant we lost".

The schema declares the field:

```yaml
  - name: Dopants [mol%]
    type: mapping
    key_label: Element          # what the filter's dropdown is called
    value_label: mol%
    key_options: [Ir, Ru, Cr]   # optional
```

`key_options` follows the rule the rest of the schemas follow: it says what is
*expected*, not what is *allowed*. An element nobody declared is reported and
then accepted, because absorbing what the schema did not anticipate is the point
of the database.

### 2.7 The form

The contribute form renders a mapping field as a two-column editable table with
dynamic rows (`st.data_editor`), which is the Streamlit-native way to type an
arbitrary number of pairs, and writes back the same `name=value` string the
sheet uses. Form and sheet then produce byte-identical metadata, and a form
entry stays rebuildable through the ordinary sheet route rather than becoming a
second kind of entry that only the form can make.

### 2.8 Temperature steps are not quite the same problem

Dopants are a *set* of independent quantities. A temperature profile is a
*sequence*: the keys are numbers, and holding 1150 °C for 10 h after two lower
steps is not the same treatment as reaching it first.

Stored as a mapping it records correctly, and the questions people would ask of
it are aggregates rather than per-key ranges — *held above 1100 °C*, *total time
above 1000 °C*, *peak temperature*. Those are a maximum over the keys and a
conditional sum over the pairs, neither of which a per-sub-key slider expresses.

The profile is therefore stored as the mapping, for the record and for display,
and the schema declares **derived scalars** computed at ingestion, which then
get ordinary sliders through the machinery that already exists. The group asked
for two: `Peak temperature [°C]` (`max_key`) and `Time at peak temperature [h]`
(`value_at_max_key`). `DERIVED_FUNCTIONS` in `entity_schema.py` is where more
go, and an unknown one is refused when the file is read rather than at
ingestion.

A mapping whose names are not numbers — a set of dopants — derives nothing
rather than deriving zero, which would be a measurement nobody made.

### 2.9 What this does not solve

Sorting the results table by a dopant's concentration needs that sub-key
promoted to a generated column, exactly as the hot-key promotion in
[the plan's §4.4](photocatalytic_database.md) describes for ordinary keys. The
same measurement applies: promote the two or three that matter, not everything,
and run `ANALYZE` afterwards.

---

## 3. What the group settled

- **`catalyst_batch` accepts `catalyst_batch` and `modified_catalyst_batch`**,
  and nothing else. `modified_catalyst_batch` is a full kind of entry with its
  own schema and its own example data in the seeded demo.
- **Peak temperature and time at peak** are the derived scalars for a profile.
- **Both `=` and `:`** are accepted between a name and its value.

## 4. Where this leads

§1.7 above merges the paths to one field so that a role accepting two kinds of
entry does not halve the answer. It works, but it is a fold applied after the
fact to a naming scheme that keeps generating the duplicates, and it cannot
express several references of one kind in one field —
`Precursor chemicals: EA-1; EA-2; EA-3`.

[docs/referencing_redesign.md](referencing_redesign.md) plans the change that
removes the need for it: naming inherited metadata by the **type** of the entry
that owns it rather than by the route that reached it, and holding a set of
contributions per key rather than a value. It measures the three ways of storing
that set and finds the current design already spends 1130 ms per page load at ten
thousand entries.

## 5. Still open

- Whether the shipped `key_options` lists — dopants, cocatalysts — are the
  group's real vocabulary. They are what is *expected*, so an unlisted name is
  reported and accepted, but a list nobody recognises makes that report useless.
- Whether sorting a results table by one dopant's concentration is wanted
  enough to promote that sub-key to a generated column (§2.9).
