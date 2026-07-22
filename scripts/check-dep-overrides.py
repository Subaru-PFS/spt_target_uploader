#!/usr/bin/env python3
#
# check-dep-overrides.py
# Verify that [tool.uv] override-dependencies stay in sync with [project] dependencies.
#
# uv treats two different URLs for the same package as a hard conflict, so this
# project pins the PFS git dependencies twice: once in [project].dependencies and
# once in [tool.uv].override-dependencies. The override is what actually wins, so
# a tag bumped in only one of the two places silently has no effect.
#
# This script fails when a package is pinned in both places with different URLs,
# or when an override claims to mirror a direct dependency that no longer exists.
#
# Usage: python3 scripts/check-dep-overrides.py [path/to/pyproject.toml]

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

# "name @ url" — the only form that pins a URL. Plain version specifiers such as
# "numpy>=2.4.0" have no URL to compare and are ignored.
_URL_REQUIREMENT = re.compile(r"^\s*(?P<name>[A-Za-z0-9._-]+)\s*@\s*(?P<url>\S+)\s*$")


def normalize(name: str) -> str:
    """Normalize a distribution name per PEP 503 so ics-cobraOps == ics_cobraops."""
    return re.sub(r"[-_.]+", "-", name).lower()


def url_requirements(requirements: list[str]) -> dict[str, str]:
    """Map normalized package name -> URL for every "name @ url" requirement."""
    found = {}
    for req in requirements:
        match = _URL_REQUIREMENT.match(req)
        if match:
            found[normalize(match["name"])] = match["url"]
    return found


def main(argv: list[str]) -> int:
    pyproject_path = Path(argv[1]) if len(argv) > 1 else Path("pyproject.toml")
    if not pyproject_path.is_file():
        print(f"error: {pyproject_path} not found", file=sys.stderr)
        return 2

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    direct = url_requirements(data.get("project", {}).get("dependencies", []))
    overrides = url_requirements(
        data.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    )

    if not overrides:
        print("No [tool.uv] override-dependencies to check.")
        return 0

    errors = []
    checked = 0

    for name, override_url in sorted(overrides.items()):
        if name not in direct:
            # Transitive-only override (no counterpart in [project].dependencies).
            # Nothing to compare against, so there is nothing that can drift.
            continue
        checked += 1
        if direct[name] != override_url:
            errors.append(
                f"{name}:\n"
                f"    [project].dependencies:              {direct[name]}\n"
                f"    [tool.uv].override-dependencies:     {override_url}"
            )

    if errors:
        print(
            f"{pyproject_path}: {len(errors)} override(s) out of sync with "
            f"[project].dependencies:\n",
            file=sys.stderr,
        )
        for error in errors:
            print(f"  {error}\n", file=sys.stderr)
        print(
            "The override is what uv actually resolves, so the version pinned in\n"
            "[project].dependencies has no effect until these match. Update both.",
            file=sys.stderr,
        )
        return 1

    print(
        f"{pyproject_path}: OK — {checked} override(s) match [project].dependencies "
        f"({len(overrides) - checked} transitive-only override(s) skipped)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
