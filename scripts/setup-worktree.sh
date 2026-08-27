#!/usr/bin/env bash
#
# setup-worktree.sh
# Prepare a fresh git worktree for development.
#
# Claude Code copies the gitignored files listed in .worktreeinclude
# (.env.shared, .env.private, .env.docker, .python-version) into every new
# worktree, but it cannot copy .venv/ (~3 GB, and its editable install would
# point back at the main checkout) or data/ (runtime output). This script
# builds those inside the worktree:
#
#   1. sanity-check that .worktreeinclude did its job (.env.shared present)
#   2. uv sync --all-extras   (or fall back to pip install -e ".[dev,profilers]")
#   3. mkdir -p data/temp
#
# It is idempotent, so re-running it is safe.
#
# Usage:
#   ./scripts/setup-worktree.sh [uv|venv]
#
# Arguments:
#   uv     - force 'uv sync'
#   venv   - force 'pip install -e' into the active environment
#   (none) - auto-detect (priority: uv > pip)
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

RUNNER_TYPE="${1:-auto}"

# 1. Check the .worktreeinclude copy landed.
if [ ! -f ".env.shared" ]; then
    echo "Warning: .env.shared is missing." >&2
    echo "  If this is a Claude Code worktree, .worktreeinclude should have copied it." >&2
    echo "  Otherwise seed it yourself:  cp .env.shared.example .env.shared" >&2
    echo "  (and cp .env.private.example .env.private for the admin app)" >&2
fi

# 2. Install dependencies.
install_with_uv() {
    echo "==> uv sync --all-extras"
    uv sync --all-extras
}

install_with_pip() {
    echo "==> pip install -e \".[dev,profilers]\""
    pip install -e ".[dev,profilers]"
}

case "${RUNNER_TYPE}" in
    uv)
        if ! command -v uv &> /dev/null; then
            echo "Error: 'uv' not found in PATH" >&2
            exit 1
        fi
        install_with_uv
        ;;
    venv)
        if ! command -v pip &> /dev/null; then
            echo "Error: 'pip' not found in PATH (activate the environment first)" >&2
            exit 1
        fi
        install_with_pip
        ;;
    auto)
        if command -v uv &> /dev/null; then
            install_with_uv
        elif command -v pip &> /dev/null; then
            install_with_pip
        else
            echo "Error: neither 'uv' nor 'pip' found in PATH" >&2
            echo "Install uv (https://docs.astral.sh/uv/) or activate a Python environment." >&2
            exit 1
        fi
        ;;
    *)
        echo "Error: Invalid runner type '${RUNNER_TYPE}'" >&2
        echo "Usage: $0 [uv|venv]" >&2
        exit 1
        ;;
esac

# 3. Runtime output directory.
mkdir -p data/temp
echo "==> data/temp ready"

echo
echo "Worktree ready. Next:"
echo "  ./scripts/serve-app.sh          # main uploader app (http://localhost:5008/uploader/)"
echo "  ./scripts/serve-app-admin.sh    # admin app (http://localhost:5009/uploader-admin/)"
echo "  uv run pfs-uploader-cli --help  # CLI"
