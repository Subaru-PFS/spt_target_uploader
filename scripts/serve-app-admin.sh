#!/usr/bin/env bash
#
# serve-app-admin.sh
# Start the development server for the admin application
#
# This script starts the PFS Target Uploader admin web application with:
# - Port 5009
# - WebSocket support for localhost:5009
# - Static file serving for documentation and data
# - Autoreload enabled for development
# - 100MB max upload size
#
# Usage:
#   serve-app-admin.sh [uv|venv]
#
# Arguments:
#   uv    - Use 'uv run' to execute the command
#   venv  - Use '.venv/bin/' to execute the command directly
#   (none) - Auto-detect (priority: uv > venv)
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
    venv)
        if [ ! -f "${VENV_RUNNER}" ]; then
            echo "Error: ${CLI_CMD} not found in .venv/bin/" >&2
            echo "Please run 'uv sync' first" >&2
            exit 1
        fi
        RUNNER="${VENV_RUNNER}"
        ;;
    auto)
        # Auto-detect: Priority: uv > venv
        if command -v uv &> /dev/null; then
            RUNNER="${UV_RUNNER}"
        elif [ -f "${VENV_RUNNER}" ]; then
            RUNNER="${VENV_RUNNER}"
        else
            echo "Error: Cannot find ${CLI_CMD}" >&2
            echo "Please install dependencies using 'uv sync'" >&2
            exit 1
        fi
        ;;
    *)
        echo "Error: Invalid runner type '${RUNNER_TYPE}'" >&2
        echo "Usage: $0 [uv|venv]" >&2
        exit 1
        ;;
esac

# Execute the command
exec ${RUNNER} start-app admin \
    --port 5009 \
    --allow-websocket-origin localhost:5009 \
    --prefix=uploader-admin/ \
    --static-dirs doc=docs/site \
    --static-dirs data=data \
    --basic-login-template ./templates/basic_login_admin_dev.html \
    --max-upload-size=100 \
    --autoreload
