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

WORKDIR /build

# Copy dependency specifications
COPY requirements.txt ./

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel pybind11

# Install Python dependencies to user site-packages
# Note: PyQt5 build requires significant memory (8GB+ recommended in Docker settings)
# If build fails with memory errors, increase Docker Desktop memory allocation:
# Settings > Resources > Memory > Set to 8GB or higher
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# Copy source and install the package itself
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --user --no-cache-dir --no-warn-script-location --no-deps -e .

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

# Copy Python dependencies from builder to pfsuser's home
COPY --from=python-builder --chown=pfsuser:pfsuser /root/.local /home/pfsuser/.local

# Set PATH for pfsuser
ENV PATH=/home/pfsuser/.local/bin:$PATH

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

# Install setuptools (provides pkg_resources) and the application package
RUN pip install --user --no-cache-dir setuptools && \
    pip install --user --no-cache-dir --no-deps -e .

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
