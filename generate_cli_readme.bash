#!/bin/bash

set -e

typer pfs_target_uploader.cli.cli_main utils docs --name pfs-uploader-cli | sed 's/# `/## `/g' | sed 's/###/---\n\n###/g' | sed 's/`pfs-targetdb-cli\ /`/g'
