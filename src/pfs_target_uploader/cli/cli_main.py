#!/usr/bin/env python3

import os
import secrets
from datetime import date
from typing import Annotated

import typer
from astropy.table import Table
from loguru import logger

from ..utils.checker import validate_input
from ..utils.io import load_input, upload_file
from ..utils.ppp import PPPrunStart, ppp_result
from ..widgets import StatusWidgets

app = typer.Typer(
    help="PFS Target Uploader CLI Tool",
    context_settings={"help_option_names": ["--help", "-h"]},
    add_completion=False,
)


@app.command(help="Validate a target list for PFS openuse")
def validate(
    input_list: Annotated[
        str, typer.Argument(show_default=False, help="Input CSV file")
    ],
    date_begin: Annotated[
        str, typer.Option("--date-begin", help="Begin date (e.g., 2023-02-01)")
    ] = None,
    date_end: Annotated[
        str, typer.Option("--date-end", help="End date (e.g., 2023-07-31)")
    ] = None,
):

    df_input, dict_load = load_input(input_list)

    if not dict_load["status"]:
        logger.error(
            f"Cannot load the input file. Please check the input file format. Error: {dict_load['error']}"
        )
        return
    else:
        date_begin = None if date_begin is None else date.fromisoformat(date_begin)

        date_end = None if date_end is None else date.fromisoformat(date_end)

        _, _ = validate_input(df_input, date_begin=date_begin, date_end=date_end)


@app.command(help="Run the online PPP to simulate pointings")
def simulate(
    input_list: Annotated[str, typer.Argument(help="Input CSV file")],
    output_dir: Annotated[
        str, typer.Option("-d", "--dir", help="Output directory")
    ] = ".",
    date_begin: Annotated[
        str, typer.Option("--date-begin", help="Begin date (e.g., 2023-02-01)")
    ] = None,
    date_end: Annotated[
        str, typer.Option("--date-end", help="End date (e.g., 2023-07-31)")
    ] = None,
    max_exec_time: Annotated[
        int, typer.Option("--max-exec-time", help="Max execution time (s)")
    ] = None,
    max_nppc: Annotated[
        int, typer.Option("--max-nppc", help="Max number of pointings to consider")
    ] = None,
):
    df_input, dict_load = load_input(input_list)

    if not dict_load["status"]:
        logger.error(
            f"Cannot load the input file. Please check the input file format. Error: {dict_load['error']}"
        )
        return
    else:
        date_begin = None if date_begin is None else date.fromisoformat(date_begin)
        date_end = None if date_end is None else date.fromisoformat(date_end)

        if max_exec_time is None:
            max_exec_time = 0
        if max_nppc is None:
            max_nppc = 0

        validation_status, df_validated = validate_input(
            df_input, date_begin=date_begin, date_end=date_end
        )

        if validation_status["status"] is False:
            logger.error("The input target list could not pass all validations.")
            return

    tb_input = Table.from_pandas(df_validated)
    tb_visible = tb_input[validation_status["visibility"]["success"]]

    logger.info("Running the online PPP to simulate pointings")
    (
        uS_L2,
        _,
        cR_L_,
        sub_l,
        obj_allo_L_fin,
        uS_M2,
        _,
        cR_M_,
        sub_m,
        obj_allo_M_fin,
        _,  # ppp_status
    ) = PPPrunStart(tb_visible, None, max_exec_time, max_nppc=max_nppc)

    logger.info("Summarizing the results")
    _, p_result_fig, p_result_ppc, p_result_tab = ppp_result(
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

    logger.info("Saving the results")
    _, outfile_zip, sio = upload_file(
        df_validated,
        p_result_tab.value,
        p_result_ppc.value,
        df_summary,
        p_result_fig,
        outdir_prefix=output_dir,
        origname=os.path.basename(input_list),
        origdata=open(input_list, "rb").read(),
        skip_subdirectories=True,
    )
