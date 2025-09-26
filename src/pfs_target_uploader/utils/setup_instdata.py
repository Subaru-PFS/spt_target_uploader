#!/usr/bin/env python3
"""PFS instrument data setup script"""

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger


def setup_pfs_instdata(target_dir: str | Path | None = None, force: bool = False):
    """Setup PFS instrument data"""

    if target_dir is None:
        # Clone next to the project root
        project_root = Path(__file__).parent
        target_dir = project_root.parent / "pfs_instdata"
    else:
        target_dir = Path(target_dir)

    if target_dir.exists() and not force:
        logger.info(f"pfs_instdata already exists at {target_dir}")
        update_env_config(str(target_dir.absolute()))
        return str(target_dir.absolute())

    try:
        logger.info(f"Cloning pfs_instdata to {target_dir}")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/Subaru-PFS/pfs_instdata.git",
                str(target_dir),
            ],
            check=True,
        )

        # Set PFS_INSTDATA_DIR to the absolute path of target_dir
        instdata_path = str(target_dir.absolute())
        update_env_config(instdata_path)

        logger.info(f"Successfully setup pfs_instdata at {instdata_path}")
        return instdata_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone pfs_instdata: {e}")
        raise


def update_env_config(instdata_path: str):
    """Update environment configuration file"""
    env_file = Path(".env.shared")

    # Read existing configuration
    config_lines = []
    pfs_instdata_found = False

    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("PFS_INSTDATA_DIR="):
                    config_lines.append(f'PFS_INSTDATA_DIR="{instdata_path}"\n')
                    pfs_instdata_found = True
                else:
                    config_lines.append(line)

    # Add new setting if not found
    if not pfs_instdata_found:
        config_lines.append(f'PFS_INSTDATA_DIR="{instdata_path}"\n')

    # Write to file
    with open(env_file, "w") as f:
        f.writelines(config_lines)

    logger.info(f"Updated .env.shared with PFS_INSTDATA_DIR={instdata_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup PFS instrument data")
    parser.add_argument("--target-dir", help="Target directory for pfs_instdata")
    parser.add_argument("--force", action="store_true", help="Force re-clone if exists")

    args = parser.parse_args()
    setup_pfs_instdata(args.target_dir, args.force)
