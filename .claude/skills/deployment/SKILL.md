---
name: deployment
description: Use when building Docker images, deploying to Google Cloud Run, configuring nginx reverse proxy for the admin app, or working with GitHub Actions workflows (docs builds, copyright update, security scanning) in the PFS Target Uploader.
---

# Deployment Reference

## Docker Build and Deployment

Three-stage `Dockerfile` (docs-builder → python-builder → runtime, all `linux/amd64` —
PyQt5, pulled in transitively via `ics-cobraCharmer`, has no `linux/arm64` wheels). The
runtime stage has no baked-in `.env.shared`; `ENTRYPOINT` is
`scripts/docker-entrypoint.sh`, which generates it from environment variables at
container start (unless one is already present, e.g. bind-mounted) and then execs
`pfs-uploader-cli start-app uploader`. `utils/config.py` only reads `.env.shared` via
`dotenv_values()`, never `os.environ` directly — the entrypoint script is what makes
`docker run -e ...` / compose `environment:` work at all.

Quickest path: `docker compose up --build` (uses `docker-compose.yml` at the repo root).

The `python-builder` stage prunes `pfs-instdata`'s bundled `data/pfi/modules/` down to
`ALL/ALL.xml` (~1.3GB of historical calibration snapshots this app never reads — see the
comment above that `RUN` and the matching `CLAUDE.md` Gotchas entry). This is what keeps
the image around 2.5GB instead of 3.8GB; if a `pfs-instdata` version bump breaks the app
with `FileNotFoundError` from `cobraCoach.py`'s `loadModel()`, that prune step is the first
place to look.

Defaults to the license-free **HiGHS** solver backend (`SOLVER_BACKEND=highs`); `gurobipy`
still ships in the image (the app imports it unconditionally at module scope in
`pn_app.py`), so switching to Gurobi needs only `-e SOLVER_BACKEND=gurobi -e
GRB_LICENSE_FILE=... -v gurobi.lic:...`, no rebuild. Full variable list and defaults are
documented at the top of `scripts/docker-entrypoint.sh` and in `README.md`'s "Running with
Docker" section. Notable non-obvious ones:

- `UPLOADID_DB`: if set and the DB file under `OUTPUT_DIR` does not exist, the entrypoint
  runs `pfs-uploader-cli uid2sqlite` to create it before starting the app (otherwise
  `load_app_config()` raises `FileNotFoundError` at startup). Left unset by default —
  an empty (rather than absent) `UPLOADID_DB=` value in `.env.shared` would make
  `_resolve_db_path()` treat `OUTPUT_DIR` itself as the database path.
- `PREFIX` / `USE_XHEADERS`: needed when running behind a reverse proxy; the container
  `HEALTHCHECK` reads both `PORT` and `PREFIX` from its own environment so it still passes
  when they're overridden.

Manual build (what `scripts/build-container.sh -d` wraps): copy `.env.docker.example` to
`.env.docker` and customize; the build script auto-loads it if present.

```bash
./scripts/build-container.sh -p   # Update dependencies (uv sync --upgrade)
./scripts/build-container.sh -d   # Build Docker image
./scripts/build-container.sh -g   # Deploy to Google Cloud Run
./scripts/build-container.sh -p -d -g   # Combine operations

# Override via environment (takes priority over .env.docker)
DOCKER_PUSH=true DOCKER_USER=your-dockerhub-username ./scripts/build-container.sh -d
```

Environment variables (priority: command-line env > `.env.docker` > script defaults):

- `DOCKER_USER` — Docker Hub username (default: myuser)
- `DOCKER_IMAGE` — image name (default: pfs_target_uploader)
- `DOCKER_TAG` — image tag (default: latest)
- `DOCKER_PUSH` — set `"true"` to push (default: false)
- `DOCKER_PLATFORMS` — build platforms (default: linux/amd64; arm64 disabled due to PyQt5 build issues)

`docker_image()` in `build-container.sh` also passes `--build-arg
SETUPTOOLS_SCM_PRETEND_VERSION=<version>`, derived via `uvx --from setuptools-scm
setuptools-scm` from git metadata, so built images report a real version instead of the
Dockerfile's `0.0.0` fallback.

## Production Deployment with Nginx (Admin App)

When the admin app runs behind nginx with Panel's `--prefix` option, static assets (logo, favicon) on the login page need special handling.

**Why**: Panel's `static_dirs` serves files at the root (`/assets/`) without the prefix, but nginx only forwards requests matching the location pattern. Panel's basic auth also blocks unauthenticated requests including static assets. Serving assets directly from nginx bypasses both issues.

### Login templates

Environment-specific templates with correct asset paths:

- Development: `templates/basic_login_admin_dev.html` (prefix `/uploader-admin-dev/`) — used automatically by `scripts/serve-app-admin.sh`
- Production: `templates/basic_login_admin.html` (prefix `/uploader-admin/`) — pass `--basic-login-template ./templates/basic_login_admin.html` in the production startup command

### Nginx configuration

The `/assets/` location block must appear **before** the main app location block. Replace the alias path with the actual absolute path to the assets directory.

Production (development is the same with `-dev` suffix and port 8091):

```nginx
# Static assets - serve directly from nginx (before authentication)
location /uploader-admin/assets/ {
    alias /path/to/pfs_target_uploader/assets/;
    expires 1y;
    add_header Cache-Control "public, immutable";
}

# Admin app - proxy to Panel app with authentication
location /uploader-admin/ {
    proxy_pass http://127.0.0.1:8081/uploader-admin/;
    include snippets/reverse_proxy_common.conf;
}
```

After updating: `sudo systemctl reload nginx`.

## GitHub Actions Workflows

- **`update-copyright.yml`**: Updates LICENSE copyright year every Jan 1 (or manually via Actions UI). Creates a PR against `dev-main` (branch `auto-update-copyright-{YEAR}`, labels `automated`, `maintenance`).
- **`build_docs_dev.yml`** / **`build_docs_prod.yml`**: Build MkDocs docs on push to `dev-main` / `main`. Self-hosted runner (`pfs-etc`), Python 3.11.9.
- **`codacy.yml`**: Codacy scan on push/PR to `main`/`dev-main` + weekly (Mon 06:38 UTC).
- **`codeql.yml`**: CodeQL Python scan on push/PR to `main`/`dev-main` + weekly (Thu 04:43 UTC).
