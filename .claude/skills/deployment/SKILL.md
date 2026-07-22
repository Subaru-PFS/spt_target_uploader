---
name: deployment
description: Use when building Docker images, deploying to Google Cloud Run, configuring nginx reverse proxy for the admin app, or working with GitHub Actions workflows (docs builds, copyright update, security scanning) in the PFS Target Uploader.
---

# Deployment Reference

## Docker Build and Deployment

Configuration: copy `.env.docker.example` to `.env.docker` and customize. The build script auto-loads `.env.docker` if it exists.

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
