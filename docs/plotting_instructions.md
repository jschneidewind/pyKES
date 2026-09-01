# Plotting instructions: time series and results table

The Streamlit pages that display data are configured entirely from the
`plotting_instruction` dictionary carried by the `ExperimentalDataset`. The
external app owns that dictionary; the components only read it. This document
describes the two entries that drive the visualization pages:

* `'time_series_instructions'` — the curves of the Time-Series page, including
  units and multiple y axes.
* `'results_table_instructions'` — the columns of the Analysis Results Table.

---

## 1. Time-series instructions

Every key is a curve offered in the **Select Data to Display** multiselect;
its value says where the data lives on an `Experiment`:

```python
PLOTTING_INSTRUCTIONS = {
    'time_series_instructions': {
        'Raw (O2, liquid phase)': {
            'x': 'raw_data/time_s',
            'y': 'raw_data/oxygen_umol_L',
        },
        'Reaction (O2, liquid phase)': {
            'x': 'processed_data/time_reaction_s',
            'y': 'processed_data/data_reaction_umol',
            'unit_x': 'processed_data/time_unit',
            'unit_y': 'processed_data/data_unit',
        },
        'Rate (O2, liquid phase)': {
            'x': 'processed_data/time_reaction_s',
            'y': 'processed_data/rate_gaussian_umol_s',
            'x_point': 'processed_data/max_rate_time_s',
            'y_point': 'processed_data/max_rate_umol_s',
            'unit_x': 'processed_data/time_unit',
            'unit_y': 'processed_data/rate_gaussian_unit',
        },
    },
}
```

| Key | Required | Meaning |
| --- | --- | --- |
| `x`, `y` | yes | slash-separated paths to the plotted series |
| `x_point`, `y_point` | no | a single highlighted point, drawn on top of the curve |
| `unit_x`, `unit_y` | no | unit of the respective axis |

Paths are resolved with `resolve_experiment_attributes`; a curve whose `x` or
`y` cannot be resolved for a given experiment is silently skipped for that
experiment, which is what lets one instruction set serve experiments of
different types.

### Units

`unit_x` and `unit_y` are optional. Each is either

* a **path** into the experiment holding the unit string
  (`'processed_data/time_unit'`), which is the normal case when the processing
  function stores the unit alongside the data, or
* a **literal** unit string (`'umol / s'`), for a quantity whose unit is fixed
  by the instruction.

The two are told apart by the first path component: if it names an attribute of
the `Experiment` (`raw_data`, `processed_data`, `metadata`, …) the entry is
treated as a path, otherwise as a literal. A declared path that is *absent* for
a particular experiment yields no unit for that curve rather than an axis
titled with the path.

Units are resolved separately from the data, so a missing unit never removes a
curve. Where an instruction defines no units at all, the page behaves exactly as
before: the axes keep the generic titles `Time (s)` and `Value`.

The unit string is used verbatim as the axis title, so an app that prefers
`t / s` over `s` simply stores that string. Units also appear in the hover box
next to the hovered coordinates.

### Several y axes

Curves whose y unit differs are drawn against **their own y axis**. This is what
makes it possible to show an amount (`umol`) and a rate (`umol / s`) in one
figure: on a shared axis the rate, three orders of magnitude smaller, would be a
flat line at zero.

* The **first** unit — in the order the curves were selected — keeps the
  left-hand axis.
* Every further unit gets an axis at the right-hand edge; the plotting area is
  shortened by `SECONDARY_AXIS_WIDTH` (6 % of the figure width) for each axis
  beyond the first right-hand one.
* Curves **without** a unit stay on the primary axis, so mixing annotated and
  unannotated curves does not create a spurious second scale.
* Only the primary axis draws a grid — several overlapping grids make the
  figure unreadable.

A highlighted `x_point` / `y_point` always shares its curve's axis.

Because the selection order decides which unit is on the left, selecting
*Rate* before *Reaction* puts the rate on the left-hand axis and the amount on
the right.

---

## 2. Results-table instructions

Each key is one **column** of the Analysis Results Table; each selected
experiment is one **row**:

```python
'results_table_instructions': {
    'Max. rate (mmol/h/g)': {'result': 'processed_data/max_rate_mmol_h_g',
                             'error':  'processed_data/max_rate_uncertainty',
                             'format': '.2f'},
    'Apparent quantum yield (%)': {'result': 'processed_data/apparent_quantum_yield'},
}
```

| Key | Required | Meaning |
| --- | --- | --- |
| `result` | yes | path to the value |
| `unit` | no | target unit; `Quantity` values are converted into it and the unit is appended to the cell |
| `format` | no | Python format spec, default `.4g` |
| `error` | no | path to an uncertainty, rendered as `value ± error` |

Every instruction key becomes a column whether or not the value exists, so the
table keeps its shape across experiments; cells that cannot be resolved show
`—`. Values must be scalar — single-element arrays are unwrapped, longer arrays
have no meaningful cell representation and are treated as missing.

The table is exported as CSV with one row per experiment, which is the layout
spreadsheets and plotting tools expect for per-experiment records.
