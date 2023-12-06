#!/usr/bin/env python3

import argparse
from datetime import date

from loguru import logger

from ..utils.checker import validate_input
from ..utils.io import load_input


def main():
    parser = argparse.ArgumentParser(
        description="Validate a target list for PFS openuse"
    )
    parser.add_argument("input", type=str, help="Input file (must be a CSV file)")
    parser.add_argument(
        "--date_begin", type=str, default=None, help="Begin date (e.g., 2023-02-01)"
    )
    parser.add_argument(
        "--date_end", type=str, default=None, help="End date (e.g., 2023-07-3  1)"
    )

    args = parser.parse_args()

    df_input, dict_load = load_input(args.input)

    if not dict_load["status"]:
        logger.error(
            f"Cannot load the input file. Please check the input file format. Error: {dict_load['error']}"
        )
    else:
        date_begin = (
            None if args.date_begin is None else date.fromisoformat(args.date_begin)
        )

        date_end = None if args.date_end is None else date.fromisoformat(args.date_end)

        _, _ = validate_input(df_input, date_begin=date_begin, date_end=date_end)


if __name__ == "__main__":
    main()
