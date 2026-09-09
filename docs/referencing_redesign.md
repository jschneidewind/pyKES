# Identifying inherited metadata by entity type, not by path

**This is built.** The document is kept as the record of why the design is
shaped as it is — the reasoning, the measurements, and the alternatives that
were rejected. What it *does* is described in
[docs/database_app.md](database_app.md).

Every number below was measured, on the seeded demo where the question was about
correctness and on a synthetic 10 000-experiment index where it was about cost.
The numbers taken again from the finished implementation are in §8; two of them
moved enough to change decisions, and both are recorded there rather than
quietly updated above.

---

## 1. The problem, and why the current fix is a dead end

Inherited keys are qualified by the **role path** that reached them:

```
catalyst_batch/finished_semiconductor/Dopants [mol%]
catalyst_batch/catalyst_batch/finished_semiconductor/Dopants [mol%]
```

Both are the dopants of the semiconductor the experiment was run on. The second
arises only because a *modified* catalyst batch sits one hop further along. The
browse page currently shows one filter for each, and `merge_key_paths` was added
to fold them back together for display.

That fold is a patch over a naming scheme that keeps generating the duplicates,
and it does not scale to what comes next. Two references of one kind under one
field — `Precursor chemicals: EA-1; EA-2; EA-3` — cannot be expressed at all: the
`edges` primary key is `(source, role)`, so one role holds exactly one target,
and `extract_references` raises `ReferenceError` on a repeated role.

**Qualifying by entity type is the right answer.** `Dopants [finished
semiconductor]` names the field by the *kind of thing that owns it*, which is
what a person searching actually means, and is stable however the graph is
routed.

## 2. The catch, demonstrated on the group's own data

Type qualification collides. Not hypothetically — it collides on the seeded demo
today, before multi-references exist:

```
precursor_chemical/Supplier
    <- …/precursor_chemical_a/Supplier = 'Merck'
    <- …/precursor_chemical_b/Supplier = 'Alfa'
precursor_chemical/Purity [%]
    <- …/precursor_chemical_a/Purity [%] = 99.99
    <- …/precursor_chemical_b/Purity [%] = 99.0
```

Two entities of one type, reached by two roles, both claiming one key. Under the
current path scheme these are distinct keys and the merge is *incapable* of
collision — that was the whole reason paths were chosen. Type qualification gives
that up, and `Precursor chemicals: EA-1; EA-2; EA-3` makes the collision the
**normal case** rather than an edge case.

So the change cannot be "replace the path prefix with the type prefix". It has
to be:

> **A type-qualified key holds a set of contributions, not a value.**

This is the whole design. Everything below follows from it.

### 2.1 Two identities, not one

The proposal says to identify metadata by "entity type + entity ID". That is
right for a *value* and wrong for a *column*, and keeping them apart is what
makes the rest work:

| | Identified by | Example |
| --- | --- | --- |
| **A value** | contributor id + key | `SEMI-004` · `Dopants [mol%]` = `{Ir: 0.02}` |
| **A column / filter** | contributor *type* + key | `Dopants [finished semiconductor]` |

If the column were identified by id as well, the key space would grow with the
data: every new semiconductor would create new columns and the facet list would
be unusable. The id belongs in the provenance, which is exactly where the
proposal puts it — the entity page.

### 2.2 What a filter now means

`Synthesis temperature [finished semiconductor] = 1150` becomes **"some finished
semiconductor this entry reaches was synthesised at 1150 °C"**. For today's chain
— one semiconductor per experiment — that is identical to the current meaning.
With several contributors it is not, and the difference is worth stating plainly
because it cannot be hidden:

Filtering `Supplier = Merck` **and** `Purity > 99.9` over three precursors
matches an entry where *one* precursor is from Merck and a *different* one is
99.95% pure. That is usually what people want — "a sample involving something
from Merck and something very pure" — but it is not "a precursor from Merck that
is 99.95% pure".

**Recommendation:** default to independent existential matching, and label the
filter group *Any finished semiconductor* / *Any precursor chemical* so the
meaning is on screen rather than in a document. Add an opt-in "same entry" mode
only if the group asks for it; the mechanism is one `EXISTS` carrying both
conditions on the same contributor, so it is cheap to add later and confusing to
add speculatively. **This is a decision for the group, not one to assume.**

---

## 3. Where the set lives: measured, not argued

Three shapes, 10 000 experiments, ~19 inherited fields, three precursor
chemicals per semiconductor:

| | **(a) today**<br>path-qualified scalars | **(b)** type-qualified<br>JSON arrays | **(c)** a `contributions`<br>table |
| --- | --- | --- | --- |
| Index size | 20.7 MB | **13.9 MB** | 68.8 MB |
| Inherited facets | 27 (3 precursor slots) | 19 | 19 |
| **Facet bounds, per page load** | **1130 ms** | 608 ms | **37 ms** |
| One range filter | 43.8 ms | 32.5 ms | **9.5 ms** |
| Three-predicate search | 23.2 ms | 19.2 ms | **15.5 ms** |
| Dopant sub-key filter | — | 23.2 ms | indexed |
| Provenance for one entry | a graph walk | a graph walk | **0.04 ms** |

**(b) is the tempting answer and it is wrong.** Arrays are smaller than today and
the queries are fine, but every facet bound needs a `json_each` scan, and the
sidebar draws nineteen of them on every interaction. 608 ms per keystroke is not
a usable page.

**(c) wins on everything that is paid repeatedly**, at 3.3× the size — 69 MB for
ten thousand experiments, on a server that already stores 140 KB of payload per
experiment. It also *stores* provenance, which is precisely what the entity page
needs and what (a) and (b) would have to recompute.

### 3.1 A finding that is not about this change at all

**The current design already costs 1130 ms per page load at 10 000 entries.**
`build_facets` runs one `MIN`/`MAX` scan per numeric facet, every time the
sidebar is drawn. On the 60-entry demo that is invisible; at the target scale it
is over a second before anything is rendered. This redesign removes it as a side
effect (37 ms), but it is a live problem in the code today and would need fixing
regardless — which is worth knowing before deciding how urgent this work is.

---

## 4. The proposed shape

### 4.1 `contributions`

```sql
CREATE TABLE contributions (
    entity_id        TEXT    NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    contributor      TEXT    NOT NULL,          -- the entry that owns the value
    contributor_type TEXT    NOT NULL,          -- what names the column
    role_paths       TEXT,                      -- every route that reached it; display only
    depth            INTEGER NOT NULL,          -- shortest route, for ordering the sidebar
    key              TEXT    NOT NULL,          -- leaf name, escaped exactly as today
    sub_key          TEXT    NOT NULL DEFAULT '',  -- one name inside a mapping
    value            TEXT,                      -- text form
    number           REAL,                      -- numeric form where it is one
    PRIMARY KEY (entity_id, contributor, key, sub_key)
);

CREATE INDEX idx_contrib_number ON contributions(contributor_type, key, sub_key, number);
CREATE INDEX idx_contrib_value  ON contributions(contributor_type, key, sub_key, value);
CREATE INDEX idx_contrib_entity ON contributions(entity_id);
```

Four things this buys, each of which is a current problem:

* **Deduplication is the primary key.** One entry reached by two routes
  contributes once. Today a diamond in the graph — one chemical used as both
  precursors, two semiconductors sharing a precursor — produces two keys and
  would produce two array elements under (b). Here it cannot.
* **`role_paths` keeps the path information** the proposal asks to retain, as a
  JSON array because a contributor genuinely can be reached more than one way.
  It is display-only: nothing filters on it.
* **`number` alongside `value`** makes range filters use an index rather than
  `CAST(json_extract(...))` over every row — the 43.8 → 9.5 ms difference.
* **`sub_key` explodes mappings**, so a dopant filter is an index lookup rather
  than a scan with `json_extract` inside it.

`entities.effective` is dropped. `entities.metadata` (own metadata) is unchanged,
and own fields keep their bare names — a filter on the entry's own `Operator` is
not existential and should not pretend to be.

### 4.2 Multiple references in one cell

```yaml
  - name: Precursor Chemicals
    type: reference
    role: precursor_chemical
    accepts: [precursor_chemical, commercial_chemical]
    multiple: true
```

* `edges` primary key becomes `(source, role, target)` with an `ordinal` column,
  so declaration order survives for display.
* `extract_references` returns a list rather than a mapping, and splits a cell on
  `;` or a newline — **the same separators a mapping field already uses**, so
  there is one convention to learn rather than two.
* The repeated-role `ReferenceError` is removed. It existed because two fields
  under one role produced identical qualified keys and the second silently
  overwrote the first; sets make that impossible, which is the point.
* **`precursor_chemical_a` and `precursor_chemical_b` collapse into one field.**
  The lettered roles only ever existed to dodge the collision this change
  removes — the same reasoning that replaced `Co-catalyst A`/`B` with a mapping.

### 4.3 What the role becomes

After this, a role no longer participates in identity. It survives for three
things: naming the relationship on screen (*Original Catalyst Batch*), carrying
`accepts`, and building `role_paths` for the entity page. That is a genuine
simplification, and it is what makes an experiment referencing either kind of
batch a non-event rather than a special case.

### 4.4 What gets deleted

`merge_key_paths`, `Facet.alternative_keys`, and the `COALESCE` branch of
`build_expression` — the machinery added to fold duplicate paths back together —
all become unnecessary. Type qualification merges by construction. That is a
useful signal that the new scheme is the one the problem wanted.

---

## 5. Problems this creates, and what to do about them

Ordered by how likely each is to cause a wrong answer rather than an
inconvenience.

**1. A conjunction can be satisfied by different contributors.** §2.2. The
default is defensible and the label must say so. A silently wrong answer here
would be indistinguishable from a right one, which is why it belongs on screen.

**2. One type at two depths now merges.** A precursor chemical reached through a
`precursor_semiconductor` and one reached through a `finished_semiconductor`
both become `Supplier [precursor chemical]`. That is the intended behaviour —
"any precursor chemical in the lineage" — but it is a **semantic widening** that
nobody asked for explicitly, and it cannot be undone by a filter afterwards.
Worth confirming with the group before building. If it is unwanted, the fix is to
qualify by type *and* depth, which reintroduces a weaker version of the current
problem.

**3. You can no longer filter "only via a modified batch".** Paths leave the key,
so that distinction leaves the filter. In practice it comes back for free: a
modified batch carries fields an ordinary one does not, so filtering
`Modification [type] [modified catalyst batch]` selects exactly those experiments.
Worth checking that the group has no query this does not cover.

**4. Multi-valued columns need display rules**, in three places that currently
assume one value:
   * *Browse table* — propose distinct values joined with `; `, truncated with a
     count. `format_value` already renders a list.
   * *Sorting* by such a column is ill-defined — propose sorting by the minimum
     for numbers and disabling it for text, or omitting those columns from the
     sort menu entirely.
   * *Property map* axes need one number per point — propose the mean for
     numeric axes with the count shown in the tooltip, and excluding
     multi-valued keys from the colour axis. **This is the weakest part of the
     plan**; a scatter point whose x is an average of three precursors is easy to
     misread, and it may be better to offer only single-contributor keys as axes
     and say why.

**5. Write amplification on editing a shared ancestor.** Correcting a precursor
today rewrites one `effective` blob per descendant; it will now rewrite one row
per descendant *per key*. The current recompute is 165 ms for 12 340 entities, so
the row-wise version needs measuring before this is called cheap — it is the one
number in this plan that has not been taken.

**6. Registry rebuild and stale bounds.** Facet bounds come from `MIN`/`MAX` over
an indexed column, so they stay correct without a registry cache. Maintaining
them incrementally instead would be faster still but can only widen a range, not
narrow it after an edit — worth *not* doing, given 37 ms is already acceptable.

**7. Shared search links break.** Filter keys in the query string change shape.
An old link cannot always be translated, because a role is not a type. Say so;
a search URL is a message, not a record.

**8. Size.** 69 MB at ten thousand experiments, against 21 MB today. Both are
small beside the payload store. If it mattered, `value` and `number` are
partially redundant and one index of the three could go — but it does not
matter at this scale and the plan should not pay complexity for it.

Two things that are explicitly *not* problems, having been checked:
`__SLASH__` escaping is unchanged — the prefix goes from several path components
to one type, and `split_qualified_key` still works. And own metadata cannot
collide with inherited metadata of the same type, because own keys stay bare
while inherited keys are always type-prefixed.

---

## 6. Order of work

Each step leaves the application working.

| Step | What | Why here |
| --- | --- | --- |
| 1 | `contributions` table; rewrite resolution as a reachability walk that collects contributors by id; migration that recomputes from `metadata` + `edges` | The data layer alone. No re-ingestion needed: `effective` was always derived, so nothing depends on replaying uploads. |
| 2 | Existential filters, facets and the key registry read from `contributions`; delete `merge_key_paths` and `alternative_keys` | Restores the browse page on the new shape and removes the workaround in the same move. |
| 3 | Multi-reference: `edges` key, `extract_references`, `multiple: true` | Needs the set semantics of step 1 to be correct before it is possible. |
| 4 | Entity page: contributor type + id, with `role_paths` shown beside it | The user-visible half of the proposal; trivial once `contributions` exists. |
| 5 | Merge `precursor_chemical_a`/`_b` into one `Precursor Chemicals` field | A schema edit, last, once the machinery supports it. |

Steps 1–2 are the redesign; 3–5 are what it was for.

## 7. What the group settled

1. **Independent existential matching is the default**, with an opt-in "same
   entry" mode — a checkbox per filter group, *Match one precursor chemical*.
2. **One type at two depths merges.** A precursor chemical reached through a
   `precursor_semiconductor` and one reached through a `finished_semiconductor`
   are one filter.
3. **Only single-contributor keys are offered as property-map axes.** Nothing is
   averaged.
4. **Filtering by route is not needed.** Paths are shown on the entry page and
   filter nothing.

## 8. What the finished implementation measures

Taken again at 10 000 experiments, 2 000 batches, 300 semiconductors and 40
chemicals, with each semiconductor naming three precursors — so the fan-out the
old design could not express is present throughout.

| | old design | as planned | shipped |
| --- | --- | --- | --- |
| Index size | 20.7 MB | 68.8 MB predicted | **118.3 MB** |
| Contribution rows | — | — | 347 600 |
| Facet bounds, per page load | 1130 ms | 37 ms | **114 ms** |
| Three-predicate target query | 43.8 ms per filter | 15.5 ms | **119 ms** (count and page) |
| One dopant's concentration range | — | — | 40 ms |
| Any precursor from Merck | not expressible | — | 69 ms |
| One precursor both from Merck and pure | not expressible | — | 63 ms |
| Provenance for one entry | a graph walk | 0.04 ms | **0.1 ms** |
| Ingestion | 30 ms per experiment | — | 2.7 ms per experiment |

**Two things the plan got wrong, both found by measuring the finished code
rather than by reasoning about it.**

*The index is 118 MB, not the 69 MB predicted.* The benchmark modelled 19
inherited fields; the real chain contributes about 30 per experiment, and three
indexes over 348 000 rows is most of the difference. It is still small beside
the payload store, which holds 140 KB per experiment.

*A first working version was 2102 ms per page load — worse than the 1130 ms it
replaced.* The cause was the query planner, not the design: scoping a facet's
bounds to one kind of entry with a subquery over `entities` made SQLite drive
from the per-entity index and probe once per experiment, 97.8 ms per facet
against 1.7 ms for a covering read. The fix is why `contributions` carries an
`entity_type` column it does not logically need: with the kind of entry leading
both value indexes, a lookup narrows to one field of one kind before it looks at
a value. The same mistake in reverse cost the `match_same` predicate 237 ms —
anchoring it on a row that already satisfies the first condition, rather than on
any row of the right kind, brought it to 28 ms.

Neither would have shown up on the seeded demo. Both are the reason the plan
called for measuring the implementation and not only the design.
