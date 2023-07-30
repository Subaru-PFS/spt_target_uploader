#!/usr/bin/env python3

import argparse

import pandas as pd
from logzero import logger

from ..utils import load_input, validate_input


def main():
    parser = argparse.ArgumentParser(
        description="Validate a target list for PFS openuse"
    )
    parser.add_argument("input", type=str, help="Input file (must be a CSV file)")

    args = parser.parse_args()

    df_input, dict_load = load_input(args.input)

    if not dict_load["status"]:
        logger.error(
            f"Cannot load the input file. Please check the input file format. Error: {dict_load['error']}"
        )
    else:
        _ = validate_input(df_input)


if __name__ == "__main__":
    main()
