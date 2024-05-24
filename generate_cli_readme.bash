#!/bin/bash

set -eu

# shellcheck disable=SC2016
typer pfs_target_uploader.cli.cli_main utils docs --name pfs-uploader-cli |
    sed 's/# `/## `/g' |
    sed 's/###/---\n\n###/g' |
    sed 's/`pfs-uploader-cli\ /`/g'
