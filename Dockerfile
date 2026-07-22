# Multi-stage Dockerfile for PFS Target Uploader
# Optimized for smaller image size and faster builds
# https://hub.docker.com/_/python

# IMPORTANT: This Dockerfile must be built for linux/amd64 platform
# Reason: PyQt5 prebuilt wheels are available for amd64, avoiding memory-intensive source builds
# Usage: docker build --platform linux/amd64 -t pfs_target_uploader .
# Or use: ./scripts/build-container.sh -d (automatically uses correct platform)

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
COPY --chown=pfsuser:pfsuser scripts/ ./scripts/

# Copy app entry point
COPY --chown=pfsuser:pfsuser tmp/bak/app.py ./app.py

# Copy pre-built documentation (without videos - 135MB saved)
COPY --from=docs-builder /build/docs/site ./docs/site

# Create data directory and configuration
RUN mkdir -p ${APP_HOME}/data && \
    echo 'OUTPUT_DIR="/app/data"' > .env.shared && \
    chown -R pfsuser:pfsuser ${APP_HOME}

# Switch to non-root user
USER pfsuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/').read()" || exit 1

# Run the web service on container startup
CMD ["panel", "serve", "./app.py", \
     "--address", "0.0.0.0", \
     "--port", "8080", \
     "--allow-websocket-origin=*", \
     "--static-dirs", "doc=./docs/site/", \
     "--static-dirs", "data=/app/data", \
     "--num-procs=4"]

# TODO
# - Remove temporary files in data directory periodically using crontab
#   https://www.airplane.dev/blog/docker-cron-jobs-how-to-run-cron-inside-containers
