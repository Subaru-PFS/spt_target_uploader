# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PFS Target Uploader is a web application for validating and submitting target lists for the Subaru Telescope's Prime Focus Spectrograph (PFS). It performs validation, pointing simulation (PPP), and visualization of astronomical target lists. Built with Panel (web UI) and Typer (CLI).

Detailed references live in project skills — consult them instead of guessing:

- **`deployment`** skill: Docker builds, Cloud Run, nginx reverse-proxy setup, GitHub Actions workflows
- **`internals`** skill: HEALPix visibility checking, internal duplication detection, PPP performance optimizations, third-party log suppression

## Branching Policy

Two-tier branching model:

- **`main`**: Production/release branch. Only updated via PRs from `dev-main`.
- **`dev-main`**: Integration branch and default base for everyday work.

Rules:

1. Create feature branches off `dev-main` (e.g. `issue/<number>-<description>`, `fix/<description>`).
2. **Open PRs against `dev-main`, not `main`** (`gh pr create --base dev-main`). Feature branches must never target `main` directly.
3. A release is a single PR from `dev-main` into `main`. Only target `main` when the user explicitly asks for a release PR.

## Setup and Commands

`uv` is the primary tool (PDM and pip also work; scripts auto-detect uv > pdm > venv).

```bash
uv sync --all-extras        # Install all dependencies (extras: dev, profilers)
cp .env.shared.example .env.shared    # Main config
cp .env.private.example .env.private  # Credentials (not tracked)
mkdir -p data/temp

./scripts/serve-app.sh        # Run main uploader app (dev, autoreload)
./scripts/serve-app-admin.sh  # Run admin app (dev)
./scripts/serve-doc.sh        # Serve MkDocs documentation locally
./scripts/build-doc.sh        # Build documentation
./scripts/gen-requirements.sh # Regenerate requirements.txt
./scripts/gen-cli-readme.sh   # Generate CLI docs from typer docstrings
```

### Code Quality

```bash
black src/              # Format
ruff check --fix src/   # Lint (line-length via black; E501 ignored)
ty check src/           # Type check (experimental, Astral ty)
```

### CLI

```bash
pfs-uploader-cli validate target_list.csv                     # Validate a target list
pfs-uploader-cli validate target_list.csv --obs-type queue --min-mag 10.0 --max-mag 30.0
pfs-uploader-cli simulate target_list.csv --obs-type queue -d output/   # Pointing simulation
pfs-uploader-cli uid2sqlite -d data --db upload_id.sqlite     # Manage upload ID database
pfs-uploader-cli clean-uid data/upload_id.sqlite              # Clean duplicate upload IDs
```

## Architecture

Package root: `src/pfs_target_uploader/`

- **`pn_app.py`**: Main Panel web app entry point (widget orchestration)
- **`pn_admin.py`**: Admin interface for proposals and system status
- **`cli/cli_main.py`**: Typer-based CLI
- **`utils/config.py`**: `AppConfig` dataclass; `load_app_config()` (full), `load_minimal_config()` (admin), `get_min_fluxmag_for_obstype()`
- **`utils/checker.py`**: Validation logic — columns, value ranges, flux, flux range vs AB mag limits (`check_fluxrange()`), HEALPix-optimized visibility, internal duplication
- **`utils/internal_duplication.py`**: Duplicate detection via coordinate clustering (see `internals` skill)
- **`utils/ppp.py`**: Pointing simulation engine (HDBSCAN/DBSCAN clustering + Gurobi optimization)
- **`utils/io.py`**: Target list I/O and data persistence
- **`utils/db.py`**: SQLite upload ID management
- **`utils/session.py`**: Session management and security tokens
- **`utils/suppress_logging.py`**: Context managers to silence noisy third-party libs (see `internals` skill)
- **`widgets/`**: Panel UI components — one class per file (`FileInputWidgets`, `ValidationResultWidgets`, `ObsTypeWidgets`, `PppResultWidgets`, `StatusWidgets`, `TargetWidgets`, etc.)

### Data Flow

1. **Upload**: CSV target list via web UI or CLI
2. **Validation** (`checker.py`): format → coordinates → flux → visibility (HEALPix) → internal duplicates → optional flux range vs AB mag limits
3. **Clustering + Simulation** (`ppp.py`): HDBSCAN/DBSCAN grouping, Gurobi-optimized pointing patterns
4. **Results**: interactive plots + downloadable files, saved to timestamped directories under `data/`

### File Formats

- **Input CSV** required columns: `obj_id`, `ra`, `dec`, `tract`, `patch`, `target_type_id`; optional flux/magnitude and priority columns
- **Outputs**: `target_<id>.ecsv` (targets + visibility flags), `ppc_<id>.ecsv` (pointing centers), `psl_<id>.ecsv` (pointing summary), `ppp_figure_<id>.html` (plots), `README.txt`

## Configuration

- **`.env.shared`**: main config; **`.env.private`**: credentials (`ADMIN_USERNAME`/`ADMIN_PASSWORD`, gitignored); **`.env.docker`**: Docker build settings
- Key settings: `OUTPUT_DIR` (data storage), `MAX_EXETIME` (PPP timeout sec, 0 = unlimited), `CLUSTERING_ALGORITHM` (`FAST_HDBSCAN`), `PPP_QUIET`, `PPP_TIMING_VERBOSE`, `LOG_LEVEL`, `UPLOADID_DB`, `MIN_FLUXMAG_QUEUE`/`_CLASSICAL`/`_FILLER` (bright AB-mag limits per obs type), `MAX_FLUXMAG` (shared faint limit), optional `EMAIL_FROM`/`EMAIL_TO`/`SMTP_SERVER`
- See `.env.shared.example` for the full annotated list

## Testing

Automated tests are minimal (`tests/` is essentially empty). Verify changes by:

- Running the CLI (`pfs-uploader-cli validate` / `simulate`) against example lists in `tmp/example_lists/`
- Manual testing via the web interface (`./scripts/serve-app.sh`)
- Browser-driven checks with Playwright without touching project deps: `uv run --no-project --with playwright python <script>` (Chromium is cached user-side; the `webapp-testing` skill provides the workflow)

## Gotchas and Conventions

- **Gurobi license required** for pointing simulations; PPP can be computationally heavy for large lists
- All astronomical calculations use Astropy conventions (J2000/ICRS, degrees)
- Internal duplication threshold is 1.0 arcsec (PFS fiber diameter); L-mode and M-mode targets are never treated as duplicates of each other
- Use `loguru` for logging (not stdlib `logging`)
- HEALPix visibility checking (`visibility_checker_healpix()`) is the default; legacy checkers exist only for validation — details in the `internals` skill
- Panel Tabulator with dynamic styles: set `.style` **before** `.value` (reverse order raises an `iloc` error on Panel 1.8.x)
- Do not hand-edit `uv.lock`, `pdm.lock`, or `requirements.txt` (use `./scripts/gen-requirements.sh`)
