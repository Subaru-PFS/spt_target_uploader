#!/usr/bin/env bash
#
# gen-requirements.sh
# Generate requirements.txt from pyproject.toml
#
# This script exports the project dependencies to requirements.txt format.
# Note: This script requires PDM or uv to be installed.
#
# Usage:
#   gen-requirements.sh [uv|pdm]
#
# Arguments:
#   uv    - Use 'uv export' to generate requirements.txt
#   pdm   - Use 'pdm export' to generate requirements.txt
#   (none) - Auto-detect (priority: uv > pdm)
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

# Define export commands
UV_CMD="uv export --format requirements-txt --no-hashes --output-file requirements.txt"
PDM_CMD="pdm export --format requirements --without-hashes --pyproject --dev --output requirements.txt --verbose"

# Parse command-line argument
RUNNER_TYPE="${1:-auto}"

# Detect or validate package manager and set command
case "${RUNNER_TYPE}" in
    uv)
        if ! command -v uv &> /dev/null; then
            echo "Error: 'uv' not found in PATH" >&2
            echo "Please install uv or use pdm" >&2
            exit 1
        fi
        CMD="${UV_CMD}"
        ;;
    pdm)
        if ! command -v pdm &> /dev/null; then
            echo "Error: 'pdm' not found in PATH" >&2
            echo "Please install pdm or use uv" >&2
            exit 1
        fi
        CMD="${PDM_CMD}"
        ;;
    auto)
        # Auto-detect: Priority: uv > pdm
        if command -v uv &> /dev/null; then
            CMD="${UV_CMD}"
        elif command -v pdm &> /dev/null; then
            CMD="${PDM_CMD}"
        else
            echo "Error: Neither 'uv' nor 'pdm' found in PATH" >&2
            echo "Please install uv or pdm to generate requirements.txt" >&2
            exit 1
        fi
        ;;
    *)
        echo "Error: Invalid runner type '${RUNNER_TYPE}'" >&2
        echo "Usage: $0 [uv|pdm]" >&2
        exit 1
        ;;
esac

# Execute the command
exec ${CMD}
