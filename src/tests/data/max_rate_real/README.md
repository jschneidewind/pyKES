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
| `AE-854_B2` | `260909_Calibration_Data_Plate1.h5` | well plate, blank well, same group |
| `AE-867_B2` | `260909_Calibration_Data_Plate1.h5` | well plate, blank well, same group |
| `AE-868_C2` | `260909_Calibration_Data_Plate1.h5` | well plate, blank well, same group |

The PyroScience and UniAmp traces keep the `dt (s)` / `Time since start (s)`
column and the main measurement channel; the well-plate traces are the
offset-corrected reaction arrays the upstream processing writes, which is
exactly what the pipeline is handed in production.

## Why these eleven reaction traces

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

## Why these three blanks

`AE-854_B2`, `AE-867_B2` and `AE-868_C2` are the three wells of a 176-well
calibration plate (160 reaction wells, 16 blanks) on which the kinetic length
scale collapsed onto the sensor oscillation every well on that plate carries -- a
quasi-periodic wander with a 270-320 s period. The rate curve then oscillated
about zero instead of decaying, and the reported maximum was a crest of it:

| well | collapsed | at the noise floor | the plate for comparison |
| --- | --- | --- | --- |
| `AE-854_B2` | 4.00e-6 | 9.4e-7 | smallest of the 160 reaction wells: 2.58e-6 |
| `AE-867_B2` | 1.18e-6 | -3.9e-8 | 5th percentile: 5.42e-6 |
| `AE-868_C2` | 6.46e-7 | 3.4e-7 | median: 1.06e-5 |

`AE-854_B2` is the reason they are here: at 4.00e-6 umol/s it out-performed the
weakest real catalyst on its own plate. The other 13 blanks were never affected
and came out at or below 1.3e-7, so these three are the failing half of a
boundary that the plate itself supplies -- the same shape of evidence as the
`AE-855_B2` / `AE-855_C2` pair above.

They are also the reason the blanks are tested separately from the eleven in
`REAL_TRACES`: `AE-854_B2` masks 18 % of its samples as outliers, because its
first ~800 s genuinely are spiky, so the "residual equals the fitted noise" and
"fewer than 5 % masked" properties that every reaction trace has to satisfy are
the wrong questions to ask of it.

The source HDF5 is not committed -- it is 24 MB. The fixtures were written from
its `processed_data/{time_reaction_s, data_reaction_umol}` arrays with
`DataFrame.to_csv(index=False, float_format='%g')`, which is what gives them the
same six-significant-figure layout as the rest of the directory.
