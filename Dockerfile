# Multi-stage Dockerfile for PFS Target Uploader
# Optimized for smaller image size and faster builds
# https://hub.docker.com/_/python

# IMPORTANT: This Dockerfile must be built for linux/amd64 platform
# Reason: PyQt5 prebuilt wheels are available for amd64, avoiding memory-intensive source builds
# Usage: docker build --platform linux/amd64 -t pfs_target_uploader .
# Or use: ./scripts/build-container.sh -d (automatically uses correct platform)
# Or use: docker compose up --build
#
# The container defaults to the HiGHS ILP solver (no license required). See
# scripts/docker-entrypoint.sh for the full list of environment variables it
# reads (SOLVER_BACKEND, OUTPUT_DIR, PORT, PREFIX, ALLOW_WEBSOCKET_ORIGIN, ...).

# ============================================================================
# Stage 1: Build documentation
# ============================================================================
FROM python:3.12-slim-bookworm AS docs-builder

WORKDIR /build

# Copy only documentation-related files
COPY docs/ ./docs/
COPY pyproject.toml ./

# Install only documentation dependencies
RUN pip install --no-cache-dir \
    mkdocs>=1.4.3 \
    mkdocs-material>=9.5.4 \
    mkdocs-macros-plugin>=0.7.0 \
    mkdocs-video>=1.5.0 \
    myst-parser>=2.0.0

# Build documentation (videos are excluded via .dockerignore)
RUN cd docs && mkdocs build

# ============================================================================
# Stage 2: Build Python dependencies
# ============================================================================
FROM python:3.12-slim-bookworm AS python-builder

# Install build dependencies (including Qt5 for PyQt5)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        qtbase5-dev \
        qtchooser \
        qt5-qmake \
        qtbase5-dev-tools && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install uv, which resolves dependencies straight from uv.lock. There is no
# intermediate requirements.txt: the lock file is the single source of truth,
# so the image cannot drift from what `uv sync` installs locally.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

WORKDIR /build

# Build into a self-contained virtualenv that the runtime stage copies wholesale.
# It is created at its FINAL runtime path, not under /build: console-script
# shebangs bake in an absolute interpreter path, so a venv built elsewhere and
# copied would leave `panel` and friends pointing at a directory that does not
# exist in the runtime image.
ENV UV_PROJECT_ENVIRONMENT=/home/pfsuser/.venv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Dependencies first, without the project itself, so this layer is cached until
# the lock actually changes.
# Note: PyQt5 build requires significant memory (8GB+ recommended in Docker settings)
# If build fails with memory errors, increase Docker Desktop memory allocation:
# Settings > Resources > Memory > Set to 8GB or higher
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the source and the project itself. setuptools-scm derives the version
# from git metadata, which is absent here, so pin it via the environment.
COPY src/ ./src/
COPY README.md ./
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}
RUN uv sync --frozen --no-dev --no-editable

# Prune historical PFI cobra-calibration snapshots bundled by pfs-instdata
# (pinned git dependency, tag 1.8.71): ~1.3GB of the ~1.5GB this package
# installs is data/pfi/modules/, made up of ~40 per-module directories plus
# ~28 dated snapshot files (ALL_final_<date>_mm.xml, TEST.xml, ...) under
# modules/ALL/. This app's only consumer, utils/ppp.py's
# CobraCoach(loadModel=True, ...) + Bench(...) call, never passes a
# version/moduleVersion argument, so cobraCoach.py's loadModel() always
# resolves to modules/ALL/ALL.xml -- verified by tracing every open() call
# that construction makes. Nothing else in this repo touches pfs.instdata
# (grep -rl instdata src/). If pfs-instdata is upgraded and this starts
# failing, the app will raise FileNotFoundError for whatever new path
# loadModel() wants; add it to the keep-list below or drop this block.
RUN INSTDATA_MODULES="/home/pfsuser/.venv/lib/python3.12/site-packages/pfs/instdata/data/pfi/modules" && \
    test -f "${INSTDATA_MODULES}/ALL/ALL.xml" && \
    find "${INSTDATA_MODULES}" -mindepth 1 -maxdepth 1 -not -name ALL -exec rm -rf {} + && \
    find "${INSTDATA_MODULES}/ALL" -mindepth 1 -not -name ALL.xml -delete && \
    test -f "${INSTDATA_MODULES}/ALL/ALL.xml"

# Uninstall packages that are hard (unconditional) dependencies of pinned PFS
# git packages -- ics-cobraCharmer requires opencv-python; ics-utils requires
# opdb, which pulls in Twisted and both the psycopg2 and psycopg (v3) drivers
# for a live PostgreSQL connection -- but that this app never imports.
# Verified by tracing sys.modules after importing pn_app/pn_admin (the two
# Panel entry points) and after running the full CobraCoach+Bench
# construction utils/ppp.py performs (`grep -rl "cv2\|opencv\|twisted\|
# psycopg\|opdb" src/` also finds nothing). `uv pip uninstall` (rather than
# `rm -rf`) keeps the venv's installed-package metadata consistent. If a
# future code path does need one of these, `uv sync` will not reinstall it
# automatically -- add it back to this list's exclusion or drop this RUN.
RUN uv pip uninstall --python "${UV_PROJECT_ENVIRONMENT}" \
    opencv-python twisted psycopg2-binary psycopg psycopg-binary opdb

# ============================================================================
# Stage 3: Runtime image
# ============================================================================
FROM python:3.12-slim-bookworm

# Environment configuration
ENV PYTHONUNBUFFERED=True \
    APP_HOME=/app

# Install only git (required for git+https pip packages at runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 pfsuser && \
    mkdir -p ${APP_HOME} && \
    chown -R pfsuser:pfsuser ${APP_HOME}

WORKDIR ${APP_HOME}

# Copy the virtualenv built in stage 2, at the identical path it was built at
# (see the shebang note there). It already contains the application package
# installed non-editable, so no further install step is needed.
COPY --from=python-builder --chown=pfsuser:pfsuser /home/pfsuser/.venv /home/pfsuser/.venv

# Put the virtualenv first on PATH so `panel` and `python` resolve to it
ENV VIRTUAL_ENV=/home/pfsuser/.venv
ENV PATH=/home/pfsuser/.venv/bin:$PATH

# Copy application source (ordered by change frequency)
COPY --chown=pfsuser:pfsuser pyproject.toml ./
COPY --chown=pfsuser:pfsuser src/ ./src/
COPY --chown=pfsuser:pfsuser --chmod=0755 scripts/ ./scripts/

# Copy pre-built documentation (without videos - 135MB saved)
COPY --from=docs-builder /build/docs/site ./docs/site

# Create the data directory. Configuration (.env.shared) is generated at
# container startup by scripts/docker-entrypoint.sh from environment
# variables, not baked into the image -- see that script for the full list
# of variables and defaults (notably SOLVER_BACKEND, which defaults to the
# license-free HiGHS backend).
RUN mkdir -p ${APP_HOME}/data && \
    chown -R pfsuser:pfsuser ${APP_HOME}

# Switch to non-root user
USER pfsuser

EXPOSE 8080

# Health check. Reads PORT/PREFIX from the container's own environment so it
# still passes when those are overridden (e.g. behind a reverse proxy).
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "\
import os, urllib.request; \
port = os.environ.get('PORT', '8080'); \
prefix = os.environ.get('PREFIX', '').strip('/'); \
path = f'/{prefix}/' if prefix else '/'; \
urllib.request.urlopen(f'http://localhost:{port}{path}').read()" \
    || exit 1

ENTRYPOINT ["./scripts/docker-entrypoint.sh"]

# TODO
# - Remove temporary files in data directory periodically using crontab
#   https://www.airplane.dev/blog/docker-cron-jobs-how-to-run-cron-inside-containers
