#!/usr/bin/env python3

import argparse
import secrets
from datetime import date, datetime, timezone

from astropy.table import Table
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
    parser.add_argument(
        "--save", action="store_true", help="Save the validated target list"
    )
    parser.add_argument(
        "--add-upload_id",
        action="store_true",
        help="Assign upload_id to the target list",
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

        validation_status, df_validated = validate_input(
            df_input, date_begin=date_begin, date_end=date_end
        )

        # save file if validation is successful and save option is provided
        if validation_status["status"] and args.save:
            # dt = datetime.now(timezone.utc)
            if args.add_upload_id:
                secret_token = secrets.token_hex(8)
                outfile = f"target_{secret_token}.ecsv"
            else:
                secret_token = None
                outfile = "target_validated.ecsv"
            logger.info(f"Saving the validated target list to {outfile}")
            Table.from_pandas(df_validated).write(
                outfile,
                format="ascii.ecsv",
                overwrite=True,
            )


if __name__ == "__main__":
    main()
