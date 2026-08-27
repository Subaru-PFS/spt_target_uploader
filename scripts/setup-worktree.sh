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
#   4. create an empty $OUTPUT_DIR/$UPLOADID_DB if .env.shared configures one
#      (the main app calls load_app_config(validate_db=True) and raises
#      FileNotFoundError on startup when the file is missing)
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

# Read KEY from a .env file, stripping comments and surrounding quotes.
read_env() {
    grep -E "^[[:space:]]*$1=" "$2" 2>/dev/null \
        | grep -vE "^[[:space:]]*#" \
        | tail -1 \
        | sed -E "s/^[[:space:]]*$1=//; s/^[\"']//; s/[\"'][[:space:]]*\$//"
}

# 1. Check the .worktreeinclude copy landed.
if [ ! -f ".env.shared" ]; then
    echo "Warning: .env.shared is missing." >&2
    echo "  If this is a Claude Code worktree, .worktreeinclude should have copied it." >&2
    echo "  Otherwise seed it yourself:  cp .env.shared.example .env.shared" >&2
    echo "  (and cp .env.private.example .env.private for the admin app)" >&2
fi

# 2. Install dependencies. CLI_RUNNER is how we invoke pfs-uploader-cli afterwards.
install_with_uv() {
    echo "==> uv sync --all-extras"
    uv sync --all-extras
    CLI_RUNNER="uv run pfs-uploader-cli"
}

install_with_pip() {
    echo "==> pip install -e \".[dev,profilers]\""
    pip install -e ".[dev,profilers]"
    CLI_RUNNER="pfs-uploader-cli"
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
OUTPUT_DIR="$(read_env OUTPUT_DIR .env.shared)"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
mkdir -p "${OUTPUT_DIR}/temp"
echo "==> ${OUTPUT_DIR}/temp ready"

# 4. Bootstrap an empty upload_id database if one is configured but absent.
#    .worktreeinclude cannot carry it (it lives under the wholly-ignored data/).
UPLOADID_DB="$(read_env UPLOADID_DB .env.shared)"
if [ -n "${UPLOADID_DB}" ]; then
    DB_PATH="${OUTPUT_DIR}/${UPLOADID_DB}"
    if [ -f "${DB_PATH}" ]; then
        echo "==> ${DB_PATH} already present"
    else
        echo "==> creating empty upload_id database at ${DB_PATH}"
        ${CLI_RUNNER} uid2sqlite --dir "${OUTPUT_DIR}" --db "${UPLOADID_DB}"
    fi
fi

echo
echo "Worktree ready. Next:"
echo "  ./scripts/serve-app.sh          # main uploader app (http://localhost:5008/uploader/)"
echo "  ./scripts/serve-app-admin.sh    # admin app (http://localhost:5009/uploader-admin/)"
echo "  ${CLI_RUNNER} --help  # CLI"
