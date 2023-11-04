#!/usr/bin/env python3

import argparse
import os
from datetime import date

from astropy.table import Table
from logzero import logger

from ..utils.checker import validate_input
from ..utils.io import load_input, upload_file
from ..utils.ppp import PPPrunStart, ppp_result
from ..widgets import StatusWidgets


def main():
    parser = argparse.ArgumentParser(
        description="Validate a target list for PFS openuse"
    )
    parser.add_argument("input", type=str, help="Input file (must be a CSV file)")
    parser.add_argument(
        "-d", "--dir", type=str, default=".", help="Output directory (default=.)"
    )
    parser.add_argument(
        "--date_begin",
        type=str,
        default=None,
        help="Begin date (e.g., 2023-02-01, default=None)",
    )
    parser.add_argument(
        "--date_end",
        type=str,
        default=None,
        help="End date (e.g., 2023-07-31, default=None)",
    )

    args = parser.parse_args()

    df_input, dict_load = load_input(args.input)

    if not dict_load["status"]:
        logger.error(
            f"Cannot load the input file. Please check the input file format. Error: {dict_load['error']}"
        )
        return
    else:
        date_begin = (
            None if args.date_begin is None else date.fromisoformat(args.date_begin)
        )
        date_end = None if args.date_end is None else date.fromisoformat(args.date_end)

        validation_status, df_validated = validate_input(
            df_input, date_begin=date_begin, date_end=date_end
        )

        if validation_status["status"] is False:
            logger.error("The input target list could not pass all validations.")
            return

    # TODO: I don't like to hard-code the parameters
    weights = [4.02, 0.01, 0.01]

    tb_input = Table.from_pandas(df_validated)
    tb_visible = tb_input[validation_status["visibility"]["success"]]

    (
        uS_L2,
        cR_L,
        cR_L_,
        sub_l,
        obj_allo_L_fin,
        uS_M2,
        cR_M,
        cR_M_,
        sub_m,
        obj_allo_M_fin,
    ) = PPPrunStart(tb_visible, weights)

    nppc, p_result_fig, p_result_ppc, p_result_tab = ppp_result(
        cR_L_,
        sub_l,
        obj_allo_L_fin,
        uS_L2,
        cR_M_,
        sub_m,
        obj_allo_M_fin,
        uS_M2,
    )

    _status_widget = StatusWidgets()
    _status_widget.show_results(df_validated, validation_status)

    df_summary = _status_widget.df_summary

    _, outfile_zip, sio = upload_file(
        df_validated,
        p_result_tab.value,
        p_result_ppc.value,
        df_summary,
        p_result_fig,
        origname=os.path.basename(args.input),
        origdata=open(args.input, "rb").read(),
        export=True,
    )

    # Write the stuff
    with open(os.path.join(args.dir, outfile_zip), "wb") as f:
        f.write(sio.getbuffer())


if __name__ == "__main__":
    main()
