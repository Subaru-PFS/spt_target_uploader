# Validation fixtures

Small target lists, one per validation outcome, for tests that need to reach a
specific result from `validate_input()` without hand-building a DataFrame.

Every file shares the same 4-row shape and column set as `valid_minimal.csv`
and differs from it by one deliberate defect, so `diff valid_minimal.csv
<fixture>` shows exactly what makes it fail. `tests/test_example_target_lists.py`
pins what each one triggers.

These are **not** examples for users. Published examples live in
`docs/docs/examples/`, which the documentation site links to; a deliberately
broken file there would read as a recommendation.

## The fixtures

| File | Triggers |
|---|---|
| `valid_minimal.csv` | nothing — passes every check |
| `missing_required_column.csv` | `required_keys` fails (`reference_arm` dropped) |
| `empty_header_only.csv` | `empty` fails (header, no data rows) |
| `invalid_string.csv` | `str` fails (`ob_code` holds a space, and a `/`) |
| `invalid_value_range.csv` | `values` fails — one row each violating `ra`, `dec`, `priority`, `exptime`, `resolution`, `reference_arm` |
| `missing_flux.csv` | `flux_columns` fails (one row has no flux at all) |
| `suspicious_flux_values.csv` | `flux_values` warns (fluxes are AB magnitudes, not nJy) |
| `flux_out_of_ab_range.csv` | `flux_range` fails — one row far too bright, one far too faint. Needs `min_mag`/`max_mag` passed in; without limits the check is skipped |
| `not_visible.csv` | `visibility` fails (dec ≈ −80°, never rises at Maunakea) |
| `duplicate_ob_code.csv` | `unique` fails (two rows share an `ob_code`) |
| `duplicate_obj_id_resolution.csv` | `unique` fails (two rows share an `(obj_id, resolution)` pair) |
| `internal_duplication.csv` | `internal_duplication` warns (two `L` targets 0.5 arcsec apart) |
| `internal_duplication_lm_pair.csv` | nothing — identical coordinates, but `L` and `M` are never duplicates of each other |

The first four are the inputs that make `validate_input()` take each of its
early returns, leaving every check below it at `None`. Consumers of a
`validation_status` have to handle those; these are how you reach them.

## Using them

Pass explicit dates. `validate_input()` otherwise defaults to "the next
semester relative to now", which makes visibility results depend on the day
the test runs:

```python
validate_input(df, date_begin=date(2026, 2, 1), date_end=date(2026, 7, 31))
```

The visibility check dominates the runtime — roughly 15 s for the first list
at a given patch of sky, then well under a second for others nearby, as the
HEALPix results are cached per pixel. Northern and southern fixtures each pay
that cost once.
