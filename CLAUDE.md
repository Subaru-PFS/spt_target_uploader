# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PFS Target Uploader is a web application for validating and submitting target lists for the Subaru Telescope's Prime Focus Spectrograph (PFS). The application performs validation, pointing simulation, and visualization of astronomical target lists.

## Development Workflow and Branching Policy

This repository uses a two-tier branching model:

- **`main`**: Production/release branch. Only updated via PRs from `dev-main`.
- **`dev-main`**: Integration branch for ongoing development. This is the default
  base for everyday work.

**Branching and PR rules**:

1. Create feature branches off `dev-main` (e.g. `issue/<number>-<description>`,
   `fix/<description>`).
2. **Open PRs against `dev-main`, not `main`.** Feature branches must never target
   `main` directly.
3. Once all intended changes are merged into `dev-main`, open a single PR from
   `dev-main` into `main` to cut a release.

**For Claude Code**: When creating a PR (e.g. `gh pr create`), default to
`--base dev-main`. Only target `main` when the user explicitly asks for a
`dev-main` → `main` release PR.

## Development Commands

### Installation and Setup
```bash
# Install dependencies with uv (recommended)
uv sync                  # Install all dependencies
uv sync --extra dev      # Install with dev tools (black, ruff, etc.)
uv sync --extra profilers # Install with profiling tools
uv sync --all-extras     # Install all optional dependencies

# Or with PDM
pdm install                  # Install all dependencies
pdm install -G dev           # Install with dev tools
pdm install -G profilers     # Install with profiling tools
pdm install --all-extras     # Install all optional dependencies

# Or with pip (legacy)
pip install -r requirements.txt
pip install -e .
pip install -e .[dev]        # With dev tools
pip install -e .[profilers]  # With profiling tools
pip install -e .[dev,profilers] # With all optional dependencies

# Setup environment configuration files
cp .env.shared.example .env.shared
cp .env.private.example .env.private
# Edit .env.shared and .env.private with your configuration

# For Docker builds (optional)
cp .env.docker.example .env.docker
# Edit .env.docker with your Docker Hub credentials

# Create required directories
mkdir -p data/
mkdir -p data/temp/

# Build documentation
./scripts/build-doc.sh
```

### Running the Applications

All scripts in `scripts/` directory support automatic detection of uv/pdm/venv, or can be explicitly specified:

```bash
# Start main uploader app (development)
./scripts/serve-app.sh          # Auto-detect runner (uv > pdm > venv)
./scripts/serve-app.sh uv       # Force use of uv
./scripts/serve-app.sh pdm      # Force use of pdm
./scripts/serve-app.sh venv     # Force use of .venv directly

# Start admin app (development)
./scripts/serve-app-admin.sh    # Auto-detect runner

# Serve documentation locally
./scripts/serve-doc.sh          # Auto-detect runner

# Build documentation
./scripts/build-doc.sh          # Auto-detect runner

# Generate requirements.txt
./scripts/gen-requirements.sh   # Auto-detect runner (uv > pdm)
```

**Deprecated: PDM scripts** (commented out in pyproject.toml, use shell scripts instead):
```bash
# PDM scripts are no longer active - use scripts/ directory instead
# pdm run serve-app        # Start main uploader app
# pdm run serve-app-admin  # Start admin app
# pdm run serve-doc        # Serve documentation
# pdm run build-doc        # Build documentation
# pdm run gen-requirements # Generate requirements.txt
```

### Code Quality and Linting
```bash
# Format code with black
black src/

# Lint with ruff (replaces flake8)
ruff check src/

# Auto-fix issues with ruff
ruff check --fix src/

# Type check with ty (experimental, from Astral)
ty check src/
```

### Database Management
```bash
# Create upload ID database
pfs-uploader-cli uid2sqlite -d data --db upload_id.sqlite

# Add upload IDs from CSV
pfs-uploader-cli uid2sqlite -d data --db upload_id.sqlite upload_id.csv

# Scan directory for existing uploads
pfs-uploader-cli uid2sqlite -d data --db upload_id.sqlite --scan-dir data

# Clean duplicates
pfs-uploader-cli clean-uid data/upload_id.sqlite
```

### CLI Validation
```bash
# Validate a target list
pfs-uploader-cli validate target_list.csv

# Validate with flux range check (AB magnitude limits)
pfs-uploader-cli validate target_list.csv --min-mag 10.0 --max-mag 30.0

# Validate queue targets with magnitude limits
pfs-uploader-cli validate target_list.csv --obs-type queue --min-mag 10.0 --max-mag 30.0

# Validate filler targets with different bright limit
pfs-uploader-cli validate target_list.csv --obs-type filler --min-mag 15.0 --max-mag 30.0

# Run pointing simulation
pfs-uploader-cli simulate target_list.csv --obs-type queue -d output/

# Run pointing simulation with flux range check
pfs-uploader-cli simulate target_list.csv --obs-type queue --min-mag 10.0 --max-mag 30.0
```

### Docker and Deployment

**Configuration**: Copy `.env.docker.example` to `.env.docker` and customize for your environment.
The script will automatically load `.env.docker` if it exists.

```bash
# Setup (one-time)
cp .env.docker.example .env.docker
# Edit .env.docker with your Docker Hub username and settings

# Update dependencies and generate requirements.txt
./scripts/build-container.sh -p

# Build Docker image (uses settings from .env.docker if exists)
./scripts/build-container.sh -d

# Build Docker image (local only, override .env.docker)
DOCKER_PUSH=false ./scripts/build-container.sh -d

# Build and push Docker image to Docker Hub (override via environment)
DOCKER_PUSH=true DOCKER_USER=your-dockerhub-username ./scripts/build-container.sh -d

# Deploy to Google Cloud Run
./scripts/build-container.sh -g

# Combine operations (update packages, build, and deploy)
./scripts/build-container.sh -p -d -g
```

**Environment variables for Docker build**:
- `DOCKER_USER` - Docker Hub username (default: myuser)
- `DOCKER_IMAGE` - Image name (default: pfs_target_uploader)
- `DOCKER_TAG` - Image tag (default: latest)
- `DOCKER_PUSH` - Set to "true" to push (default: false)
- `DOCKER_PLATFORMS` - Build platforms (default: linux/amd64; arm64 disabled due to PyQt5 build issues)

**Priority**: Command-line environment variables > `.env.docker` > script defaults

### Production Deployment with Nginx

When deploying the admin application behind an nginx reverse proxy with the `--prefix` option, additional configuration is required for static assets (logo and favicon) on the login page.

#### Admin Login Templates

Environment-specific login templates are provided with appropriate asset paths:

- **Development**: `templates/basic_login_admin_dev.html` (prefix: `/uploader-admin-dev/`)
- **Production**: `templates/basic_login_admin.html` (prefix: `/uploader-admin/`)

The development startup script (`scripts/serve-app-admin.sh`) automatically uses the development template. For production deployment, specify the production template in your startup command:

```bash
--basic-login-template ./templates/basic_login_admin.html
```

#### Nginx Configuration

When using nginx as a reverse proxy with Panel's basic authentication, static assets must be served directly by nginx to bypass authentication. Add the following configuration **before** the main application location block:

**Development**:

```nginx
# Static assets - serve directly from nginx (before authentication)
location /uploader-admin-dev/assets/ {
    alias /path/to/pfs_target_uploader/assets/;
    expires 1y;
    add_header Cache-Control "public, immutable";
}

# Admin app - proxy to Panel app with authentication
location /uploader-admin-dev/ {
    proxy_pass http://127.0.0.1:8091/uploader-admin-dev/;
    include snippets/reverse_proxy_common.conf;
}
```

**Production**:

```nginx
# Static assets - serve directly from nginx (before authentication)
location /uploader-admin/assets/ {
    alias /path/to/pfs_target_uploader/assets/;
    expires 1y;
    add_header Cache-Control "public, immutable";
}

# Admin app - proxy to Panel app with authentication
location /uploader-admin/ {
    proxy_pass http://127.0.0.1:8081/uploader-admin/;
    include snippets/reverse_proxy_common.conf;
}
```

**Important notes**:

- Replace `/path/to/pfs_target_uploader/assets/` with the actual absolute path to the assets directory
- The `/assets/` location block must appear **before** the main app location block
- After updating nginx configuration, reload nginx: `sudo systemctl reload nginx` or `sudo nginx -s reload`

**Why this is needed**: Panel's `static_dirs` parameter serves files at the root level (`/assets/`) without the prefix, but nginx only forwards requests matching the configured location pattern (e.g., `/uploader-admin-dev/`). Additionally, Panel's basic authentication blocks all unauthenticated requests, including static assets. By serving assets directly from nginx, we bypass both issues.

### Documentation Generation
```bash
# Generate CLI documentation from typer docstrings
./scripts/gen-cli-readme.sh          # Auto-detect runner (uv > pdm > venv)
./scripts/gen-cli-readme.sh uv       # Force use of uv
./scripts/gen-cli-readme.sh pdm      # Force use of pdm
./scripts/gen-cli-readme.sh venv     # Force use of .venv directly
```

### GitHub Actions Workflows

#### Automated Copyright Year Update

The repository includes a GitHub Actions workflow that automatically updates the copyright year in the LICENSE file.

**Workflow file**: `.github/workflows/update-copyright.yml`

**Execution triggers**:

- **Automatic**: Runs every January 1st at 00:00 UTC (via cron schedule)
- **Manual**: Can be triggered manually from GitHub Actions UI

**How it works**:

1. Checks if LICENSE file exists
2. Updates copyright year from `2023` to `2023-{CURRENT_YEAR}` or updates existing range
3. Creates a Pull Request if changes are detected
4. PR targets the `dev-main` branch with label `automated` and `maintenance`

**Manual execution**:

1. Go to repository on GitHub
2. Navigate to **Actions** tab
3. Select **"Update Copyright Year"** workflow
4. Click **"Run workflow"** button
5. Review and merge the automatically created PR

**What happens**:

- If changes are needed: Creates PR with branch name `auto-update-copyright-{YEAR}`
- If already up-to-date: Workflow completes without creating PR
- PR includes summary of changes and can be reviewed before merging

#### Documentation Builds

**Workflow files**: `.github/workflows/build_docs_dev.yml`, `.github/workflows/build_docs_prod.yml`

- **`build_docs_dev.yml`**: Builds MkDocs documentation on push to `dev-main` branch
- **`build_docs_prod.yml`**: Builds MkDocs documentation on push to `main` branch
- Both use self-hosted runner (`pfs-etc`) with Python 3.11.9

#### Security Scanning

**Workflow files**: `.github/workflows/codacy.yml`, `.github/workflows/codeql.yml`

- **Codacy**: Runs on push/PR to `main` or `dev-main`, plus weekly schedule (Mondays at 06:38 UTC)
- **CodeQL**: GitHub's code scanning for Python, runs on push/PR to `main` or `dev-main`, plus weekly schedule (Thursdays at 04:43 UTC)

## Architecture Overview

### Core Components

- **`pn_app.py`**: Main Panel web application entry point with widget orchestration
- **`pn_admin.py`**: Admin interface for managing proposals and system status
- **`cli/cli_main.py`**: Command-line interface using Typer framework

### Key Modules

- **`utils/config.py`**: Configuration management with dataclass-based approach
  - `AppConfig` dataclass for type-safe configuration storage
  - `load_app_config()` for full validation and initialization
  - `load_minimal_config()` for lightweight loading (admin app)
  - `get_min_fluxmag_for_obstype()` helper for observation-type-specific flux limits
  - `format_for_logging()` for pretty-printed configuration display
- **`utils/checker.py`**: Target list validation logic with astronomical constraints
  - Column validation, value range checks, flux validation
  - Flux range validation against AB magnitude limits (`check_fluxrange()`)
  - Visibility checking with HEALPix optimization
  - Internal duplication detection
- **`utils/internal_duplication.py`**: Internal duplicate detection using coordinate-based clustering
- **`utils/ppp.py`**: Pointing simulation engine using clustering and optimization algorithms
- **`utils/io.py`**: File I/O operations for target lists and data persistence
- **`utils/db.py`**: SQLite database operations for upload ID management
- **`utils/session.py`**: User session management and security tokens
- **`utils/suppress_logging.py`**: Context managers for suppressing verbose third-party library output

### Widget System

All UI components are modularized in the `widgets/` directory:
- **File handling**: `FileInputWidgets.py`, `ValidationResultWidgets.py`
- **User input**: `ObsTypeWidgets.py`, `DatePickerWidgets.py`, `PPCInputWidgets.py`
- **Results display**: `PppResultWidgets.py`, `StatusWidgets.py`, `TargetWidgets.py`
- **UI controls**: `buttons.py`, `TimerWidgets.py`
- **Notifications**: `AnnouncementNoteWidgets.py`, `UploadNoteWidgets.py`, `DocLinkWidgets.py`

## Key Dependencies

- **Panel**: Web app framework for Python data applications
- **Astropy/Astroplan**: Astronomical calculations and coordinate transformations
- **astropy-healpix**: HEALPix sky tessellation for optimized visibility checking
- **Gurobi**: Optimization solver for pointing simulations (requires license)
- **qplan**: Telescope scheduling and visibility calculations
- **HDBSCAN/DBSCAN**: Target clustering algorithms
- **scikit-learn**: AgglomerativeClustering for internal duplicate detection
- **HoloViews/hvPlot**: Interactive data visualization
- **loguru**: Structured logging (replaces stdlib logging)
- **psutil**: Process and system monitoring
- **skyfield**: High-precision astronomical calculations

## Configuration

### Environment Files
- **`.env.shared`**: Main configuration (output directories, timeouts, email settings)
- **`.env.private`**: Sensitive credentials (not tracked in git)
  - `ADMIN_USERNAME`: Admin panel username
  - `ADMIN_PASSWORD`: Admin panel password

### Important Settings
```bash
OUTPUT_DIR="data"                    # Data storage location
MAX_EXETIME=1800                    # PPP timeout in seconds (default: 1800 = 30min, 0 = no limit)
CLUSTERING_ALGORITHM=FAST_HDBSCAN   # Target clustering method
PPP_QUIET=1                         # Suppress verbose PPP output
PPP_TIMING_VERBOSE=0                # PPP timing logs (0=off, 1=on)
LOG_LEVEL="INFO"                    # Logging verbosity
UPLOADID_DB="upload_id.sqlite"      # Upload deduplication database
MIN_FLUXMAG_QUEUE=""                # Min AB mag for queue obs_type (bright limit, optional)
MIN_FLUXMAG_CLASSICAL=""            # Min AB mag for classical obs_type (bright limit, optional)
MIN_FLUXMAG_FILLER=""               # Min AB mag for filler obs_type (bright limit, optional)
MAX_FLUXMAG=""                      # Max AB mag (faint limit, optional, shared across all modes)
EMAIL_FROM=""                       # Optional: sender email for notifications
EMAIL_TO=""                         # Optional: recipient email for notifications
SMTP_SERVER=""                      # Optional: SMTP server for email
```

## Data Flow

1. **Upload**: Users submit target lists (CSV format) via web interface
2. **Validation**: `checker.py` validates format, coordinates, magnitudes, and observability
   - **HEALPix Optimization**: Visibility checking uses HEALPix tessellation (nside=32, ~110 arcmin pixels) to group spatially clustered targets, significantly improving performance for large target lists
   - **Internal Duplication Check**: Detects targets within 1.0 arcsec (PFS fiber diameter) using AgglomerativeClustering with complete linkage
   - **Flux Range Check**: Optional validation of flux values against AB magnitude limits (MIN_FLUXMAG_QUEUE/CLASSICAL/FILLER for bright limit, MAX_FLUXMAG for faint limit) to identify targets that may be too bright or too faint for the specified observation mode
3. **Clustering**: `ppp.py` groups targets spatially using HDBSCAN/DBSCAN algorithms
4. **Simulation**: Pointing patterns optimized using Gurobi solver with telescope constraints
5. **Results**: Interactive plots and downloadable files generated
6. **Storage**: All outputs saved to timestamped directories under `data/`

## File Formats

### Input Target Lists
Required columns: `obj_id`, `ra`, `dec`, `tract`, `patch`, `target_type_id`
Optional: various magnitude columns (`g_flux`, `r_flux`, etc.), priority fields

### Output Files
- **`target_<id>.ecsv`**: Processed target list with visibility flags
- **`ppc_<id>.ecsv`**: Pointing center coordinates  
- **`psl_<id>.ecsv`**: Pointing summary with exposure times
- **`ppp_figure_<id>.html`**: Interactive visualization plots
- **`README.txt`**: Processing summary and metadata

## Testing

Tests are minimal in this repository. Most testing is done through:
- Manual testing via web interface
- Example target lists in `tmp/example_lists/`
- CLI validation of various input formats

## Performance Optimizations

### CobraCoach Object Reuse

The PPP simulation reuses CobraCoach/Bench objects across multiple netflow iterations to avoid repeated initialization overhead:

- **Implementation**: CobraCoach/Bench objects are created once at the start of `PPPrunStart()` and passed to all `netflowRun_single()` calls
- **Benefit**: Eliminates repeated initialization (typically ~2 seconds per initialization)
- **Impact**: Significant speedup for simulations requiring many netflow iterations
- **Location**: `utils/ppp.py` - `PPPrunStart()` creates the Bench object, `netflowRun_single()` receives it as a parameter

### Performance Timing Measurement

PPP includes optional detailed timing measurement to identify performance bottlenecks:

- **Configuration**: Set `PPP_TIMING_VERBOSE=1` in `.env.shared` or pass as environment variable
- **Output**: Logs execution time for each major stage (CobraCoach initialization, clustering, Gurobi solver, etc.)
- **Usage**: Enable when profiling PPP performance, disable for production use
- **Implementation**: `PPPTimer` class in `utils/ppp.py`

### HEALPix Visibility Checking

The visibility checker has been optimized for clustered targets using HEALPix tessellation:

- **Algorithm**: Groups targets by HEALPix pixels (nside=32, ~110 arcmin resolution)
  - Uses first target's coordinates as representative for each pixel (conservative approximation)
  - Calculates total observable time across observation period for each pixel
  - Compares each target's exptime against its pixel's total observable time
- **Optimization**: Reduces calculations from N targets to N_pixels << N (typically 10-100x fewer)
- **Performance**: Provides significant speedup (5-50x) for spatially clustered target lists
- **Implementation**: `visibility_checker_healpix()` in `utils/checker.py` (RECOMMENDED)
- **Usage**: Enabled by default; controllable via `healpix=True/False` parameter
- **Time Resolution**: Uses 15-minute ephemeris precision, optimized for 6-month observation periods
- **Correctness**: Fixed in PR #411 to correctly handle targets with small exptime in partially-observable pixels

### Visibility Checker Functions

Three implementations available with varying performance characteristics:

- **`visibility_checker()`** (LEGACY): Original per-target method (slowest but exact)
  - Sequential processing of each target
  - Kept for testing and validation purposes
  - Use only for verification or small target lists

- **`visibility_checker_vec()`** (LEGACY): Vectorized method with early exit optimization
  - Uses `np.vectorize` and observation period splitting
  - Faster than original but slower than HEALPix
  - Kept for testing and validation purposes

- **`visibility_checker_healpix()`** (RECOMMENDED): HEALPix-optimized method
  - Default implementation for production use
  - 5-50x faster than legacy implementations
  - Correctly handles partial observability scenarios
  - Conservative approximation ensures no overestimation of observability

- **`check_visibility()`**: Wrapper function with automatic method selection
  - Defaults to HEALPix implementation
  - Provides consistent interface across all methods

### Internal Duplication Detection

The application detects duplicate targets within a single proposal using coordinate-based clustering:

#### Algorithm Overview

1. **Pre-filtering**: Uses `search_around_sky()` to find candidate neighbors within `max_cluster_diameter`
2. **Connected Components**: Applies Breadth-First Search (BFS) to identify connected components
3. **Agglomerative Clustering**: Uses scikit-learn's `AgglomerativeClustering` with complete linkage for strict diameter control
4. **Nearest Neighbor Calculation**: Computes minimum separation for each clustered target

#### Parameters

- **`sep`**: Maximum separation for nearest neighbor search (default: 1.0 arcsec = PFS fiber diameter)
- **`max_cluster_diameter`**: Maximum cluster diameter (default: explicitly set to 1.0 arcsec to match fiber diameter)
- **`EXACT_DUPLICATE_TOLERANCE`**: Threshold for exact vs near duplicates (1e-5 arcsec ≈ 0.05 mas)

#### Key Features

- **Complete Linkage**: Ensures cluster diameter ≤ threshold (stricter than single/average linkage)
- **Resolution Separation**: L-mode and M-mode targets are never clustered together
- **Optimized Performance**: BFS-based connected component detection reduces memory usage
- **Memory Safety**: Optional `max_points_for_agglomerative` parameter prevents excessive memory consumption

#### Implementation

- **Core Logic**: `internal_duplication.py`
  - `_cluster_with_agglomerative()`: Main clustering algorithm with BFS optimization
  - `_find_duplicates_with_separation()`: Handles coordinate validation and result mapping
  - `dupcheck_internal()`: Public API returning isolated, exact duplicate, and near duplicate DataFrames
- **Validation Integration**: `checker.py`
  - `check_internal_duplicate()`: Called during validation pipeline
  - Returns status flags and nearest neighbor separations for UI display
- **UI Display**: `ValidationResultWidgets.py`
  - Shows duplicate targets with separation distances in tabular format
  - Displays: ob_code, obj_id, ra, dec, resolution, reference_arm, separation

#### Performance Characteristics

- **Best Case**: O(n) when most targets are isolated (no neighbors within threshold)
- **Worst Case**: O(n²) for distance matrix computation within large connected components
- **Memory**: O(k²) per connected component, where k = component size
- **Typical**: Efficient for spatially distributed targets (typical astronomical surveys)

#### Example Behavior

- **Targets 0.5" apart**: Clustered together, flagged as duplicates, separation = 0.5"
- **Targets 1.5" apart**: Not clustered (exceeds max_cluster_diameter = 1.0"), treated as isolated
- **Different resolutions**: L-mode and M-mode targets at same position are treated as separate (not duplicates)

## Development Notes

- The application requires a Gurobi license for optimization
- Pointing simulations can be computationally intensive for large target lists
- Web app uses Panel's autoreload feature for development
- Database operations are SQLite-based for simplicity
- All astronomical calculations use Astropy conventions (J2000, degrees)
- HEALPix optimization is most effective for clustered targets (typical PFS surveys)

## Utilities for Suppressing Third-Party Library Output

### `utils/suppress_logging.py`

This module provides context managers for suppressing verbose output from third-party libraries that use various logging mechanisms.

#### Context Managers

**`suppress_stderr_fd(enabled=True)`**
- Redirects process-level stderr (file descriptor 2) to `/dev/null`
- Affects the whole process including all threads
- Use for libraries that write directly to stderr bypassing Python's logging system
- Example: CobraCoach C extensions

**`suppress_loggers(logger_names, level=WARNING, include_root=True, suppress_handlers=True, enabled=True)`**
- Temporarily raises stdlib logging levels for specified loggers
- Saves and restores both logger and handler levels
- Parameters:
  - `logger_names`: Sequence of logger names to suppress
  - `level`: Temporary level to set (default: WARNING)
  - `include_root`: Also suppress root logger (default: True)
  - `suppress_handlers`: Also raise handler levels (default: True)

**`suppress_root_logger(level=WARNING, suppress_handlers=True, enabled=True)`**
- Convenience wrapper to suppress only the root logger
- Useful for coordinate transformation libraries (CoordTransp, DistortionCoefficients)

**`suppress_stdout(enabled=True)`**
- Redirects stdout to `/dev/null` when enabled
- Use for libraries with verbose print statements

**`suppress_third_party_logging(enabled=True, logger_names=DEFAULT_NOISY_LOGGERS, ...)`**
- **Primary tool for suppressing noisy initialization code**
- Combines stdlib logging suppression + low-level stderr redirection
- Default suppresses: `cobraCoach`, `butler`, `ics.cobraCharmer`, `ics.cobraOps`
- Parameters:
  - `logger_names`: Libraries to suppress via logging (default: DEFAULT_NOISY_LOGGERS)
  - `level`: Log level to set (default: WARNING)
  - `suppress_root`: Include root logger (default: True)
  - `suppress_handlers`: Suppress handlers too (default: True)
  - `redirect_stderr`: Use fd-level stderr redirect (default: True)

#### Usage Examples

```python
from pfs_target_uploader.utils.suppress_logging import (
    suppress_loggers,
    suppress_root_logger,
    suppress_third_party_logging,
)

# Suppress CobraCoach initialization (both logging + C extension stderr)
with suppress_third_party_logging(enabled=True, redirect_stderr=True):
    cobra_coach = CobraCoach(...)
    bench = Bench(...)

# Suppress coordinate transformation logs
with suppress_root_logger(logging.WARNING):
    tpos = [tele.get_fp_positions(tgt) for tele in telescopes]

# Suppress Gurobi optimizer logs
with suppress_loggers(("gurobipy",), level=logging.WARNING, include_root=False):
    prob.solve()
```

#### Implementation Notes

- All context managers support an `enabled` parameter for conditional suppression
- File descriptor redirection is thread-safe but affects the entire process
- Logger levels and handlers are properly saved and restored
- Designed for use in `ppp.py` during pointing simulation initialization

---

## TODOs

### pyproject.toml Remaining Improvements (Low Priority)

Minor technical debt items for the next dependency update cycle:

**Completed Items** (no action needed):
- ✅ License field updated to PEP 639 format
- ✅ `ruff>=0.5.0` version constraint added
- ✅ Build dependencies moved to build-system only (removed from main dependencies)
- ✅ `ics-cobraOps` now uses main repo instead of feature branch
- ✅ PDM scripts marked as deprecated in pyproject.toml

**Remaining Low-Priority Items**:
1. Add version constraint to `astroplan` dependency (currently has no version specified)
2. Consider enabling commented Ruff `select` rules for better code quality (optional)
3. Verify if `mkdocs-material[imaging]` extras are actually used; simplify if not needed

**Priority**: Address during next major dependency update cycle.