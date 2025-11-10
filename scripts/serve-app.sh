#!/usr/bin/env bash
#
# serve-app.sh
# Start the development server for the main uploader application
#
# This script starts the PFS Target Uploader web application with:
# - Port 5008
# - WebSocket support for localhost:5008 and localhost:8080
# - Static file serving for documentation and data
# - Autoreload enabled for development
# - 500MB max upload size
#
# Usage:
#   serve-app.sh [uv|pdm|venv]
#
# Arguments:
#   uv    - Use 'uv run' to execute the command
#   pdm   - Use 'pdm run' to execute the command
#   venv  - Use '.venv/bin/' to execute the command directly
#   (none) - Auto-detect (priority: uv > pdm > venv)
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

# Define CLI command and runner configurations
CLI_CMD="pfs-uploader-cli"
UV_RUNNER="uv run ${CLI_CMD}"
PDM_RUNNER="pdm run ${CLI_CMD}"
VENV_RUNNER="${PROJECT_ROOT}/.venv/bin/${CLI_CMD}"

# Parse command-line argument
RUNNER_TYPE="${1:-auto}"

# Detect or validate package manager
case "${RUNNER_TYPE}" in
    uv)
        if ! command -v uv &> /dev/null; then
            echo "Error: 'uv' not found in PATH" >&2
            echo "Please install uv or use a different runner" >&2
            exit 1
        fi
        RUNNER="${UV_RUNNER}"
        ;;
    pdm)
        if ! command -v pdm &> /dev/null; then
            echo "Error: 'pdm' not found in PATH" >&2
            echo "Please install pdm or use a different runner" >&2
            exit 1
        fi
        RUNNER="${PDM_RUNNER}"
        ;;
    venv)
        if [ ! -f "${VENV_RUNNER}" ]; then
            echo "Error: ${CLI_CMD} not found in .venv/bin/" >&2
            echo "Please run 'uv sync' or 'pdm install' first" >&2
            exit 1
        fi
        RUNNER="${VENV_RUNNER}"
        ;;
    auto)
        # Auto-detect: Priority: uv > pdm > venv
        if command -v uv &> /dev/null; then
            RUNNER="${UV_RUNNER}"
        elif command -v pdm &> /dev/null; then
            RUNNER="${PDM_RUNNER}"
        elif [ -f "${VENV_RUNNER}" ]; then
            RUNNER="${VENV_RUNNER}"
        else
            echo "Error: Cannot find ${CLI_CMD}" >&2
            echo "Please install dependencies using 'uv sync' or 'pdm install'" >&2
            exit 1
        fi
        ;;
    *)
        echo "Error: Invalid runner type '${RUNNER_TYPE}'" >&2
        echo "Usage: $0 [uv|pdm|venv]" >&2
        exit 1
        ;;
esac

# Execute the command
exec ${RUNNER} start-app uploader \
    --port 5008 \
    --allow-websocket-origin localhost:5008 \
    --allow-websocket-origin localhost:8080 \
    --prefix=uploader/ \
    --static-dirs doc=docs/site \
    --static-dirs data=data \
    --max-upload-size=500 \
    --autoreload
