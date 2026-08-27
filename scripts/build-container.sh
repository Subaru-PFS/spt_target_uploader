#!/usr/bin/env bash
#
# build-container.sh
# Build and deploy Docker container for PFS Target Uploader
#
# This script manages the Docker container build and deployment process.
#
# Usage:
#   build-container.sh [-p] [-d] [-g]
#
# Options:
#   -p    Update package dependencies (uv)
#   -d    Build the Docker image (local only by default)
#   -g    Deploy the app to Google Cloud Run
#
# Environment Variables:
#   DOCKER_USER       Docker Hub username (default: myuser)
#   DOCKER_IMAGE      Docker image name (default: pfs_target_uploader)
#   DOCKER_TAG        Docker image tag (default: latest)
#   DOCKER_PUSH       Set to "true" to push to Docker Hub (default: false)
#   DOCKER_PLATFORMS  Build platforms (default: linux/amd64; PyQt5 has no
#                     linux/arm64 wheels, so arm64 is not supported)
#
# Examples:
#   # Build locally only
#   ./build-container.sh -d
#
#   # Build and push to Docker Hub
#   DOCKER_PUSH=true DOCKER_USER=monodera ./build-container.sh -d
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

# Load .env.docker if it exists
if [ -f ".env.docker" ]; then
    echo "Loading configuration from .env.docker"
    # shellcheck disable=SC1091
    source .env.docker
fi

# Docker configuration with defaults
DOCKER_USER="${DOCKER_USER:-myuser}"
DOCKER_IMAGE="${DOCKER_IMAGE:-pfs_target_uploader}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
DOCKER_PUSH="${DOCKER_PUSH:-false}"
# PyQt5 (pulled in transitively via ics-cobraCharmer) has no linux/arm64
# wheels, so only linux/amd64 is supported here.
DOCKER_PLATFORMS="${DOCKER_PLATFORMS:-linux/amd64}"

show_help() {
    echo "Usage: $0 [-p] [-d] [-g]"
    echo "       -p    Update package dependencies using uv"
    echo "       -d    Build the Docker image (set DOCKER_PUSH=true to push)"
    echo "       -g    Deploy the app to Google Cloud Run"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_USER       Docker Hub username (default: myuser)"
    echo "  DOCKER_IMAGE      Docker image name (default: pfs_target_uploader)"
    echo "  DOCKER_TAG        Docker image tag (default: latest)"
    echo "  DOCKER_PUSH       Set to 'true' to push to Docker Hub (default: false)"
    echo ""
    echo "Examples:"
    echo "  # Build locally only"
    echo "  $0 -d"
    echo ""
    echo "  # Build and push to Docker Hub"
    echo "  DOCKER_PUSH=true DOCKER_USER=monodera $0 -d"
}

update_packages() {
    echo "Update package dependencies to the latest versions"

    if ! command -v uv &> /dev/null; then
        echo "Error: 'uv' not found in PATH" >&2
        echo "Please install uv to update packages" >&2
        exit 1
    fi

    echo "Using uv to update packages..."
    uv sync --upgrade

    echo "uv.lock has been updated"
    echo "The Docker build installs from uv.lock directly, so there is nothing to export."
}

docker_image() {
    # Build docker image
    local image_name="${DOCKER_USER}/${DOCKER_IMAGE}:${DOCKER_TAG}"

    # Derive the version from git metadata (same logic setuptools-scm would
    # use from a checkout) so the image reports something more useful than
    # the Dockerfile's 0.0.0 fallback. Run via uvx in an ephemeral env since
    # setuptools-scm is a build-time-only dependency, not part of .venv.
    local scm_version="0.0.0"
    if command -v uvx &> /dev/null; then
        scm_version="$(uvx --from setuptools-scm setuptools-scm 2>/dev/null || echo 0.0.0)"
    fi

    echo "Building Docker image: ${image_name}"
    echo "Platforms: ${DOCKER_PLATFORMS}"
    echo "Version (SETUPTOOLS_SCM_PRETEND_VERSION): ${scm_version}"

    if [ "${DOCKER_PUSH}" = "true" ]; then
        echo "Push to Docker Hub: enabled"
        docker buildx build \
            --platform="${DOCKER_PLATFORMS}" \
            --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="${scm_version}" \
            -t "${image_name}" \
            --push .
        echo "Successfully built and pushed: ${image_name}"
    else
        echo "Push to Docker Hub: disabled (set DOCKER_PUSH=true to enable)"
        docker buildx build \
            --platform="${DOCKER_PLATFORMS}" \
            --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="${scm_version}" \
            -t "${image_name}" \
            --load .
        echo "Successfully built locally: ${image_name}"
        echo "To push to Docker Hub, run with: DOCKER_PUSH=true DOCKER_USER=<your-username> $0 -d"
    fi
}

gcloud_deploy() {
    # Deploy to Google Cloud Run
    echo "Deploy the container to Google Cloud Run"
    gcloud run deploy pfs-target-uploader --source .
}

# Show help if no arguments provided
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Parse options
while getopts "pdgh" flag; do
    case "${flag}" in
    p) update_packages ;;
    d) docker_image ;;
    g) gcloud_deploy ;;
    h) show_help ;;
    *) show_help; exit 1 ;;
    esac
done
