# PFS Target Uploader

[The PFS Target Uploader](https://pfs-etc.naoj.hawaii.edu/uploader/) is a web app to validate and submit the target list supplied by users with an observing time estimate by a pointing simulation.

## Install

### Clone the repository

```sh
git clone https://github.com/Subaru-PFS/spt_target_uploader.git
cd spt_target_uploader
```

### Installing dependencies

```sh
# Install dependencies with uv (recommended)
uv sync                  # Install all dependencies
uv sync --extra dev      # Install with dev tools (black, ruff, etc.)

# Or with pip
pip install -e .
pip install -e .[dev]    # With dev tools

# Setup environment configuration files
cp .env.shared.example .env.shared
cp .env.private.example .env.private
# Edit .env.shared and .env.private with your configuration

# Create required directories
mkdir -p data/
mkdir -p data/temp/
```

Working in a Claude Code git worktree? `.worktreeinclude` copies the gitignored
config (`.env.*`, `.python-version`) in automatically; just run
`./scripts/setup-worktree.sh` in the worktree to install dependencies, set up the
`$OUTPUT_DIR/` runtime directories, and build the docs site.

### Build documentation

```sh
./scripts/build-doc.sh   # Auto-detect runner (uv/venv)
```

### Requirements

#### ILP solver for the pointing simulation

The pointing simulation (PPP) solves an integer linear program with `ets-fiber-assigner`'s netflow.
Two solver backends can be selected:

| Backend | Value | License | Installation |
| --- | --- | --- | --- |
| Gurobi (default) | `gurobi` | Commercial or academic license required for realistic target lists | `gurobipy` is installed with the app; the license file is set up separately |
| HiGHS | `highs` | None ([open source](https://highs.dev/)) | `highspy` is installed with the app; nothing else to do |

Gurobi remains the default, so an existing deployment that does not set the new option
behaves exactly as before. Without a license, Gurobi falls back to its size-limited
trial mode, which only works for small target lists; for production use with large lists
you need the Gurobi optimizer installed and a valid license
(see [Gurobi's website](https://www.gurobi.com/)) — or use HiGHS instead, which has no
license requirement.

##### Selecting the backend

**Web app**: set `SOLVER_BACKEND` in `.env.shared` (read through `utils/config.py`).

```bash
# gurobi (needs a license) or highs (open source, no license)
SOLVER_BACKEND=highs
```

The value is matched case-insensitively, so `highs`, `HiGHS` and `HIGHS` are the same
setting. An unrecognized value is rejected with a warning in the log and falls back to
`gurobi`.

**CLI**: pass `--solver` to `pfs-uploader-cli simulate` (also case-insensitive).

```sh
pfs-uploader-cli simulate target_list.csv --obs-type queue -d output/ --solver highs
```

The two settings are independent: the CLI does **not** read `SOLVER_BACKEND`, so
`simulate` uses Gurobi unless `--solver highs` is given explicitly.

Either way the chosen backend is logged once at `INFO` when the simulation starts
(`Pointing simulation starts with the highs solver`), so a run can be attributed to a
backend after the fact. With `PPP_TIMING_VERBOSE=1` the per-stage timing logs are
labeled with it as well (`highs_BuildProblem`, `highs_Solve`, ...).

##### Notes

- HiGHS support comes from `ets-fiber-assigner`, which is pinned in `pyproject.toml` to a
  commit on `tickets/FIBERALLOC-62` (the HiGHS backend merged onto `v3.8`) because that work
  is not tagged upstream yet. The pin moves back to a tag once it is released. The package
  correctly reports version `3.8.0`, but `utils/ppp.py` still probes for `nf.HighsProblem`
  and refuses `highs` with a clear message if the installed netflow does not carry it (a
  plain `v3.8` install also reports `3.8.0` but has no `HighsProblem`).
- Solver options are set to the same intent for both backends: a 5% MIP gap, a fixed seed,
  and silenced solver output. HiGHS has no equivalent of Gurobi's `method` or `degenmoves`,
  so only the settings that transfer are applied. The options handed to Gurobi are unchanged.
- On a benchmark of 22 real target lists, HiGHS completed the same 20 lists Gurobi completed
  within the execution time limit, at 1.08x the total runtime, with pointing counts agreeing
  to within the run-to-run scatter of a single solver.

## Run the app

```sh
# Start main uploader app (development)
./scripts/serve-app.sh          # Auto-detect runner (uv/venv)

# Or start admin app (development)
./scripts/serve-app-admin.sh    # Auto-detect runner
```

Open the target uploader at <http://localhost:5008/>.
Uploaded files will be stored under `data` with the following structure.

```text
$ tree data/
data/
└── <year>
    └── <month>
        └── <year month day>-<hour minute second>-<upload_id>
            ├── README.txt
            ├── pfs_target-yyyymmdd-hhmmss-<upload_id>.zip
            ├── ppc_<upload_id>.ecsv
            ├── ppp_figure_<upload_id>.html
            ├── psl_<upload_id>.ecsv
            ├── target_<upload_id>.ecsv
            ├── target_summary_<upload_id>.ecsv
            └── <original file>
```

`ppc`, `psl`, and `target` files correspond to the lists of pointing centers, the pointing summary, and input targets, respectively.
Plots are available in the `ppp_figure` file and all files are included in the `zip` file.

The path to the `data` directory can be controlled by the `OUTPUT_DIR` environment variable in `.env.shared`. An example of `.env.shared` is the following.

```bash
# OUTPUT_DIR_PREFIX must be identical to the directory value specified as `data` above.
OUTPUT_DIR="data"
```

## Configuration

The following parameters can be set in the `.env.shared` file to configure the app. Configuration is loaded and validated through the `utils/config.py` module, which provides type-safe access to all settings with appropriate defaults and validation.

```bash
# Output directory for the submitted files
OUTPUT_DIR="data"

# maximum execution time (s) to terminate the calculation (default: 900s = 15min, 0 = no limit)
# MAX_EXETIME=0

# email setting (email will be sent at each submission)
# EMAIL_FROM=
# EMAIL_TO=
# SMTP_SERVER=

# Supress output of netflow
# 0: verbose
# 1: quiet
PPP_QUIET=1

# Target clustering algorithm
# FAST_HDBSCAN, HDBSCAN, or DBSCAN
CLUSTERING_ALGORITHM=FAST_HDBSCAN

# ILP solver used for the pointing simulation
# gurobi (needs a license) or highs (open source, no license)
SOLVER_BACKEND=gurobi

# Text to be announce at the beginning (Markdown)
ANN_FILE="user_announcement.md"

# SQLite database file to be used for the duplication check of upload_id
# The file will be created under $OUTPUT_DIR
UPLOADID_DB="upload_id.sqlite"

# Flux range validation based on AB magnitude
# Leave empty or comment out to disable range checking
# Minimum AB magnitude (brightest limit) - observation mode specific
# MIN_FLUXMAG_QUEUE=10.0      # For queue observation type
# MIN_FLUXMAG_CLASSICAL=12.0  # For classical observation type
# MIN_FLUXMAG_FILLER=15.0     # For filler observation type
# Maximum AB magnitude (faintest limit) - shared across all modes
# MAX_FLUXMAG=30.0

# loggging level
# DEBUG, INFO (default), WARNING, ERROR, or CRITICAL
LOG_LEVEL="INFO"
```

## Preparing database

When `UPLOADID_DB` is set, the uploader looks up `$OUTPUT_DIR/$UPLOADID_DB` file for the duplication check of `upload_id`.
The following command can be used to generate the database file.

```sh
pfs-uploader-cli uid2sqlite -d $OUTPUT_DIR --db $UPLOADID_DB
```

If you have a list of `upload_id`s to be inserted into the database (`upload_id.csv`), you can run the command as follows.

```sh
pfs-uploader-cli uid2sqlite -d $OUTPUT_DIR --db $UPLOADID_DB upload_id.csv
```

The example content of `upload_id.csv` is as follows.

```csv
upload_id
c748124208176c40
4cd4bc355c092ad7
1b8d0c4f808972bb
2e07c75691e5ba26
c695c6b755930209
```

If you want to scan a directory (e.g., `$OUTPUT_DIR`) containing submitted uploads, you can run the command as follows.

```sh
pfs-uploader-cli uid2sqlite -d $OUTPUT_DIR --db $UPLOADID_DB --scan-dir $OUTPUT_DIR
```

You can remove duplicates by the following command.

```sh
pfs-uploader-cli clean-uid $OUTPUT_DIR/$UPLOADID_DB
```

See [the CLI documentation](./docs/cli.md) for more options.

## Production Deployment

When deploying behind an nginx reverse proxy with authentication, additional configuration is required for the admin application's login page assets. See [CLAUDE.md](./CLAUDE.md#production-deployment-with-nginx) for detailed nginx configuration and template selection instructions.
