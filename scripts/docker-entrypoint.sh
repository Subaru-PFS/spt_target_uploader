#!/usr/bin/env bash
#
# docker-entrypoint.sh
# Container entrypoint for the PFS Target Uploader web app.
#
# Responsibilities:
#   1. Generate /app/.env.shared from environment variables, unless one
#      already exists (e.g. bind-mounted by the operator), in which case it
#      is left untouched.
#   2. Initialize the upload-ID SQLite database when UPLOADID_DB is set and
#      the file does not exist yet -- utils/config.py raises
#      FileNotFoundError at startup otherwise.
#   3. exec the app (or, if arguments were passed to `docker run`, exec
#      those instead -- e.g. `docker run <image> pfs-uploader-cli validate ...`
#      or `docker run <image> bash`).
#
# Config precedence: utils/config.py's load_app_config() only reads
# .env.shared via dotenv_values(); it never reads os.environ directly. This
# script is what bridges `docker run -e SOLVER_BACKEND=highs` to that file.

set -euo pipefail

APP_HOME="${APP_HOME:-/app}"
ENV_SHARED="${APP_HOME}/.env.shared"

mkdir -p "${APP_HOME}"

# ----------------------------------------------------------------------------
# 1. Generate .env.shared
# ----------------------------------------------------------------------------
if [ -f "${ENV_SHARED}" ]; then
    echo "docker-entrypoint: ${ENV_SHARED} already exists, leaving it as-is."
else
    echo "docker-entrypoint: generating ${ENV_SHARED} from environment variables."

    OUTPUT_DIR="${OUTPUT_DIR:-${APP_HOME}/data}"
    SOLVER_BACKEND="${SOLVER_BACKEND:-highs}"
    CLUSTERING_ALGORITHM="${CLUSTERING_ALGORITHM:-FAST_HDBSCAN}"
    MAX_EXETIME="${MAX_EXETIME:-1800}"
    PPP_QUIET="${PPP_QUIET:-1}"
    PPP_TIMING_VERBOSE="${PPP_TIMING_VERBOSE:-0}"
    LOG_LEVEL="${LOG_LEVEL:-INFO}"

    {
        echo "OUTPUT_DIR=\"${OUTPUT_DIR}\""
        echo "MAX_EXETIME=${MAX_EXETIME}"
        echo "PPP_QUIET=${PPP_QUIET}"
        echo "CLUSTERING_ALGORITHM=${CLUSTERING_ALGORITHM}"
        echo "SOLVER_BACKEND=${SOLVER_BACKEND}"
        echo "LOG_LEVEL=\"${LOG_LEVEL}\""
        echo "PPP_TIMING_VERBOSE=${PPP_TIMING_VERBOSE}"

        # Pass-through-only settings: no default is applied here, and an
        # empty value is deliberately NOT written. In particular, an empty
        # UPLOADID_DB= would make _resolve_db_path() treat OUTPUT_DIR itself
        # as the database path (config.py); omitting the key instead falls
        # back to scanning the output directory directly.
        [ -n "${MAX_NPPC:-}" ] && echo "MAX_NPPC=${MAX_NPPC}"
        [ -n "${ANN_FILE:-}" ] && echo "ANN_FILE=\"${ANN_FILE}\""
        [ -n "${UPLOADID_DB:-}" ] && echo "UPLOADID_DB=\"${UPLOADID_DB}\""
        [ -n "${MIN_FLUXMAG_QUEUE:-}" ] && echo "MIN_FLUXMAG_QUEUE=${MIN_FLUXMAG_QUEUE}"
        [ -n "${MIN_FLUXMAG_CLASSICAL:-}" ] && echo "MIN_FLUXMAG_CLASSICAL=${MIN_FLUXMAG_CLASSICAL}"
        [ -n "${MIN_FLUXMAG_FILLER:-}" ] && echo "MIN_FLUXMAG_FILLER=${MIN_FLUXMAG_FILLER}"
        [ -n "${MAX_FLUXMAG:-}" ] && echo "MAX_FLUXMAG=${MAX_FLUXMAG}"
        [ -n "${EMAIL_FROM:-}" ] && echo "EMAIL_FROM=${EMAIL_FROM}"
        [ -n "${EMAIL_TO:-}" ] && echo "EMAIL_TO=${EMAIL_TO}"
        [ -n "${SMTP_SERVER:-}" ] && echo "SMTP_SERVER=${SMTP_SERVER}"
    } > "${ENV_SHARED}"

    mkdir -p "${OUTPUT_DIR}"
fi

# ----------------------------------------------------------------------------
# 2. Initialize the upload-ID database if requested and missing
# ----------------------------------------------------------------------------
# Re-read what actually ended up in .env.shared (it may have been
# bind-mounted with different values than the environment above).
if ! CONFIGURED_OUTPUT_DIR="$(grep -m1 '^OUTPUT_DIR=' "${ENV_SHARED}" | cut -d= -f2- | tr -d '"')"; then
    echo "docker-entrypoint: OUTPUT_DIR not found in ${ENV_SHARED}; it is required." >&2
    exit 1
fi
CONFIGURED_UPLOADID_DB="$(grep -m1 '^UPLOADID_DB=' "${ENV_SHARED}" 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"

if [ -n "${CONFIGURED_UPLOADID_DB:-}" ]; then
    DB_PATH="${CONFIGURED_OUTPUT_DIR}/${CONFIGURED_UPLOADID_DB}"
    if [ -f "${DB_PATH}" ]; then
        echo "docker-entrypoint: upload-ID database found at ${DB_PATH}."
    else
        echo "docker-entrypoint: initializing upload-ID database at ${DB_PATH}."
        pfs-uploader-cli uid2sqlite \
            --dir "${CONFIGURED_OUTPUT_DIR}" \
            --db "${CONFIGURED_UPLOADID_DB}" \
            --scan-dir "${CONFIGURED_OUTPUT_DIR}"
    fi
fi

# ----------------------------------------------------------------------------
# 3. Launch
# ----------------------------------------------------------------------------
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

PORT="${PORT:-8080}"
PREFIX="${PREFIX:-}"
NUM_PROCS="${NUM_PROCS:-4}"
MAX_UPLOAD_SIZE="${MAX_UPLOAD_SIZE:-500}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

cmd=(
    pfs-uploader-cli start-app uploader
    --port "${PORT}"
    --prefix "${PREFIX}"
    --num-procs "${NUM_PROCS}"
    --max-upload-size "${MAX_UPLOAD_SIZE}"
    --static-dirs "doc=${APP_HOME}/docs/site"
    --static-dirs "data=${CONFIGURED_OUTPUT_DIR:-${APP_HOME}/data}"
    --log-level "${LOG_LEVEL}"
)

if [ "${USE_XHEADERS:-false}" = "true" ]; then
    cmd+=(--use-xheaders)
fi

if [ -n "${ALLOW_WEBSOCKET_ORIGIN:-}" ]; then
    IFS=',' read -ra origins <<< "${ALLOW_WEBSOCKET_ORIGIN}"
    for origin in "${origins[@]}"; do
        cmd+=(--allow-websocket-origin "${origin}")
    done
else
    cmd+=(--allow-websocket-origin "localhost:${PORT}")
fi

echo "docker-entrypoint: starting: ${cmd[*]}"
exec "${cmd[@]}"
