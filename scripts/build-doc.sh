#!/usr/bin/env bash
#
# build-doc.sh
# Build the documentation
#
# This script builds the MkDocs documentation site.
# The output will be generated in docs/site/
#
# Usage:
#   build-doc.sh [uv|venv]
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
CLI_CMD="mkdocs"
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

# Change to docs directory and execute mkdocs build
cd "${PROJECT_ROOT}/docs"
exec ${RUNNER} build
