---
name: internals
description: Use when modifying or debugging visibility checking (HEALPix), internal duplication detection, PPP pointing simulation performance, or suppressing noisy third-party library output in the PFS Target Uploader.
---

# Algorithm and Performance Internals

## HEALPix Visibility Checking (`utils/checker.py`)

Single implementation, called through the `check_visibility()` wrapper.

- **`visibility_checker_healpix()`**: Groups targets by HEALPix pixels (nside=32, ~110 arcmin). Uses the first target's coordinates as representative per pixel (conservative approximation), computes total observable time per pixel over the observation period, and compares each target's exptime against it. Reduces N-target calculations to N_pixels << N. 15-minute ephemeris resolution, tuned for 6-month periods. Correctness for small-exptime targets in partially-observable pixels was fixed in PR #411 — don't regress this. The per-target and vectorized checkers were removed for #363; `git show v3.10.0:src/pfs_target_uploader/utils/checker.py` has them if ever needed for verification.

## Internal Duplication Detection (`utils/internal_duplication.py`)

Algorithm pipeline:

1. **Pre-filter** with `search_around_sky()` for candidate neighbors within `max_cluster_diameter`
2. **Connected components** via BFS
3. **AgglomerativeClustering** (scikit-learn) with **complete linkage** — guarantees cluster diameter ≤ threshold (stricter than single/average linkage)
4. **Nearest-neighbor separation** computed for each clustered target

Parameters:

- `sep`: nearest-neighbor search radius, default 1.0 arcsec (PFS fiber diameter)
- `max_cluster_diameter`: default explicitly 1.0 arcsec
- `EXACT_DUPLICATE_TOLERANCE`: 1e-5 arcsec (≈0.05 mas) separates exact vs near duplicates
- `max_points_for_agglomerative` (optional): memory guard for huge components

Key invariants:

- **L-mode and M-mode targets are never clustered together** (resolution separation)
- Targets 0.5" apart → clustered, flagged duplicates; 1.5" apart → isolated
- Complexity: O(n) best case (isolated targets), O(k²) memory per connected component of size k

Entry points: `dupcheck_internal()` (public API → isolated/exact/near DataFrames), `_cluster_with_agglomerative()`, `_find_duplicates_with_separation()`. Validation integration: `check_internal_duplicate()` in `checker.py`. UI: `ValidationResultWidgets.py` shows ob_code, obj_id, ra, dec, resolution, reference_arm, separation.

## PPP Performance (`utils/ppp.py`)

- **CobraCoach/Bench reuse**: created once in `PPPrunStart()` and passed to every `netflowRun_single()` call — avoids ~2 s re-initialization per netflow iteration. Preserve this pattern when refactoring.
- **Timing measurement**: set `PPP_TIMING_VERBOSE=1` (`.env.shared` or env var) to log per-stage timings via the `PPPTimer` class. Keep off in production.
- `MAX_EXETIME` caps PPP runtime (default 1800 s; 0 = unlimited).

## Suppressing Third-Party Output (`utils/suppress_logging.py`)

Context managers for noisy libraries (all accept `enabled=` for conditional use):

- **`suppress_third_party_logging(enabled=True, redirect_stderr=True, ...)`** — primary tool; combines stdlib logging suppression + fd-level stderr redirect. Default suppresses `cobraCoach`, `butler`, `ics.cobraCharmer`, `ics.cobraOps`.
- **`suppress_stderr_fd()`** — redirects fd 2 to /dev/null process-wide (for C extensions like CobraCoach)
- **`suppress_loggers(logger_names, level=WARNING, include_root=True, suppress_handlers=True)`** — temporarily raises stdlib logger/handler levels
- **`suppress_root_logger()`** — root logger only (CoordTransp, DistortionCoefficients)
- **`suppress_stdout()`** — silences print-happy libraries

Usage examples:

```python
# CobraCoach init (logging + C-extension stderr)
with suppress_third_party_logging(enabled=True, redirect_stderr=True):
    cobra_coach = CobraCoach(...)
    bench = Bench(...)

# Coordinate transformation logs
with suppress_root_logger(logging.WARNING):
    tpos = [tele.get_fp_positions(tgt) for tele in telescopes]

# Gurobi optimizer logs
with suppress_loggers(("gurobipy",), level=logging.WARNING, include_root=False):
    prob.solve()
```

Fd redirection is thread-safe but affects the whole process. Logger levels/handlers are saved and restored on exit.
