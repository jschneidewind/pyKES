# Real max-rate traces

Measured traces used by [`test_max_rate_real_data.py`](../../test_max_rate_real_data.py),
one per file, as plain two-column CSVs:

| column | meaning |
| --- | --- |
| `time_s` | time since the start of the acquisition, in seconds |
| `value` | oxygen or hydrogen, in µmol/L for the logger and hand-logged traces and in µmol for the well-plate traces |

They are committed as CSV rather than in their original instrument formats so
that the tests need no parser, no Excel reader and no HDF5 file, and so that
what the tests see cannot drift when a loader changes.

## Where each came from

All were parsed from `Untracked/260822_max_rate/`, which is not part of the
repository, with the loaders in `analyse_all_examples.py` in that directory.

| fixture | original | format |
| --- | --- | --- |
| `2026-08-06_211209_EA-693-TROXROB-Ch2-2` | same name, `.txt` | PyroScience Workbench oxygen log |
| `2026-08-07_153007_MZ-442-Ch2-2` | same name, `.txt` | PyroScience Workbench oxygen log |
| `2026-08-07_153007_MZ-443-Ch2-2` | same name, `.txt` | PyroScience Workbench oxygen log |
| `2026-08-19_112822_VSA-122-Ch2-2` | same name, `.txt` | PyroScience Workbench oxygen log |
| `2026-08-19_144524_VSA-124-Ch2-2` | same name, `.txt` | PyroScience Workbench oxygen log |
| `EA-696-Logger-4` | same name, `.xlsx` | UniAmp hydrogen logger export |
| `EA-698-Logger-2` | same name, `.xlsx` | UniAmp hydrogen logger export |
| `MRG-059-V-4-1` | same name, `.csv` | short hand-logged run |
| `MRG-059-Z-1-3` | same name, `.csv` | short hand-logged run |
| `AE-855_B2` | `260901_AE_851_to_AE-855.h5` | well plate, `processed_data/{time_reaction_s, data_reaction_umol}` |
| `AE-855_C2` | `260901_AE_851_to_AE-855.h5` | well plate, same group |

The PyroScience and UniAmp traces keep the `dt (s)` / `Time since start (s)`
column and the main measurement channel; the well-plate traces are the
offset-corrected reaction arrays the upstream processing writes, which is
exactly what the pipeline is handed in production.

## Why these eleven

The seven logger traces and two hand-logged runs are one of each format the
group records in, at the two extremes of length (~70 to ~12 000 points) and
sampling (1 s to 3.5 s).

`AE-855_B2` and `AE-855_C2` are neighbouring wells of the same plate whose
correlated sensor noise decorrelates on either side of the nuisance resolution
floor — 2.99 and 3.65 sampling intervals. B2 is the well the floor used to break
by discarding its correlated component outright; C2 is the control that must not
move when the behaviour below the floor changes.

Between them the eleven exercise all three branches of the noise
characterization on real data: a correlated component resolved above the floor
(most of them), one clamped to the floor (`AE-855_B2`), and one folded into the
white noise because the variogram is too short to support the model
(`MRG-059-*`).
