#!/usr/bin/env python3
import glob
import os
import sys
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Annotated, List

import pandas as pd
import panel as pn
import typer
from astropy.table import Table
from loguru import logger

from ..utils.checker import validate_input
from ..utils.db import bulk_insert_uid_db, create_uid_db, remove_duplicate_uid_db
from ..utils.io import load_input, upload_file
from ..utils.ppp import PPPrunStart, ppp_result
from ..widgets import StatusWidgets

app = typer.Typer(
    help="PFS Target Uploader CLI Tool",
    context_settings={"help_option_names": ["--help", "-h"]},
    add_completion=False,
)


class PanelAppName(str, Enum):
    uploader = "uploader"
    admin = "admin"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ObsType(str, Enum):
    queue = "queue"
    classical = "classical"
    filler = "filler"


@app.command(help="Validate a target list for PFS openuse.")
def validate(
    input_list: Annotated[
        str, typer.Argument(show_default=False, help="Input CSV file.")
    ],
    output_dir: Annotated[
        str, typer.Option("-d", "--dir", help="Output directory to save the results.")
    ] = ".",
    date_begin: Annotated[
        str,
        typer.Option(
            "--date-begin",
            help="Begin date (e.g., 2023-02-01). The default is the first date of the next Subaru semester.",
        ),
    ] = None,
    date_end: Annotated[
        str,
        typer.Option(
            "--date-end",
            help="End date (e.g., 2023-07-31). The default is the last date of the next Subaru semester.",
        ),
    ] = None,
    save: Annotated[
        bool,
        typer.Option(
            help='Save the validated target list in the directory specified by "--dir".'
        ),
    ] = False,
    obs_type: Annotated[ObsType, typer.Option(help="Observation type.")] = "queue",
    log_level: Annotated[
        LogLevel, typer.Option(case_sensitive=False, help="Set the log level.")
    ] = LogLevel.INFO,
):
    logger.remove(0)
    logger.add(sys.stderr, level=log_level.value)

    df_input, dict_load = load_input(input_list)

    if not dict_load["status"]:
        logger.error(
            f"Cannot load the input file. Please check the input file format. Error: {dict_load['error']}"
        )
        return
    else:
        date_begin = None if date_begin is None else date.fromisoformat(date_begin)

        date_end = None if date_end is None else date.fromisoformat(date_end)

        validation_status, df_validated = validate_input(
            df_input, date_begin=date_begin, date_end=date_end
        )

    if validation_status["status"] is False:
        logger.error("The input target list could not pass all validations.")
        return

    _status_widget = StatusWidgets()
    _status_widget.show_results(df_validated, validation_status)

    df_summary = _status_widget.df_summary

    # logger.info(f"Summary of the objects:\n{df_summary}")

    if save:
        logger.info("Saving the results")
        _, _, _ = upload_file(
            df_validated,
            None,
            None,
            df_summary,
            None,
            outdir_prefix=output_dir,
            origname=os.path.basename(input_list),
            origdata=open(input_list, "rb").read(),
            skip_subdirectories=True,
            observation_type=obs_type.value,
        )


@app.command(
    help="""Run the online PPP to simulate pointings.

    The result is written under the directory set by the `--dir` option with a 16 character random string."""
)
def simulate(
    input_list: Annotated[str, typer.Argument(help="Input CSV file")],
    output_dir: Annotated[
        str, typer.Option("-d", "--dir", help="Output directory to save the results.")
    ] = ".",
    date_begin: Annotated[
        str,
        typer.Option(
            "--date-begin",
            help="Begin date (e.g., 2023-02-01). The default is the first date of the next Subaru semester.",
        ),
    ] = None,
    date_end: Annotated[
        str,
        typer.Option(
            "--date-end",
            help="End date (e.g., 2023-07-31). The default is the last date of the next Subaru semester.",
        ),
    ] = None,
    single_exptime: Annotated[
        int, typer.Option(help="Single exposure time (s).")
    ] = 900,
    max_exec_time: Annotated[
        int,
        typer.Option(
            "--max-exec-time", help="Max execution time (s). Default is 0 (no limit)."
        ),
    ] = None,
    max_nppc: Annotated[
        int,
        typer.Option(
            "--max-nppc",
            help="Max number of pointings to consider. Default is 0 (no limit).",
        ),
    ] = None,
    obs_type: Annotated[ObsType, typer.Option(help="Observation type.")] = "queue",
    log_level: Annotated[
        LogLevel, typer.Option(case_sensitive=False, help="Set the log level.")
    ] = LogLevel.INFO,
):
    logger.remove(0)
    logger.add(sys.stderr, level=log_level.value)

    if obs_type != "classical":
        logger.warning(
            f'Force to set the single exposure time as 900s for the observation type "{obs_type.value}".'
        )
        single_exptime = 900

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
    ) = PPPrunStart(
        tb_visible,
        None,  # uPPC
        None,  # weight_para
        exetime=max_exec_time,
        max_nppc=max_nppc,
        single_exptime=single_exptime,
    )

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
        single_exptime=single_exptime,
    )

    _status_widget = StatusWidgets()
    _status_widget.show_results(df_validated, validation_status)

    df_validated["single_exptime"] = single_exptime
    df_summary = _status_widget.df_summary

    logger.info("Saving the results")
    _, _, _ = upload_file(
        df_validated,
        p_result_tab.value,
        p_result_ppc.value,
        df_summary,
        p_result_fig,
        outdir_prefix=output_dir,
        origname=os.path.basename(input_list),
        origdata=open(input_list, "rb").read(),
        skip_subdirectories=True,
        single_exptime=single_exptime,
        observation_type=obs_type.value,
    )


@app.command(help="Launch the PFS Target Uploader Web App.")
def start_app(
    app: Annotated[PanelAppName, typer.Argument(help="App to launch.")],
    port: Annotated[
        int, typer.Option(show_default=True, help="Port number to run the server.")
    ] = 5008,
    prefix: Annotated[str, typer.Option(help="URL prefix to serve the app.")] = "",
    allow_websocket_origin: Annotated[
        List[str], typer.Option(help="Allow websocket origin.")
    ] = None,
    static_dirs: Annotated[List[str], typer.Option(help="Static directories.")] = None,
    use_xheaders: Annotated[
        bool, typer.Option(help="Set --use-xheaders option.")
    ] = False,
    num_procs: Annotated[int, typer.Option(help="Number of processes to run.")] = 1,
    autoreload: Annotated[bool, typer.Option(help="Set --autoreload option.")] = False,
    max_upload_size: Annotated[
        int, typer.Option(help="Maximum file size in MB.")
    ] = 500,
    session_token_expiration: Annotated[
        int, typer.Option(help="Session token expiration time in seconds.")
    ] = 1800,
    basic_auth: Annotated[
        str, typer.Option(help="Basic authentication config (.json).")
    ] = None,
    # cookie_secret: Annotated[str, typer.Option(help="Cookie secret.")] = None,
    basic_login_template: Annotated[
        str, typer.Option(help="Basic login template.")
    ] = None,
    log_level: Annotated[
        LogLevel, typer.Option(case_sensitive=False, help="Set the log level.")
    ] = LogLevel.INFO,
):
    logger.remove(0)
    logger.add(sys.stderr, level=log_level.value)

    pn.extension(
        "floatpanel",
        "mathjax",
        "tabulator",
        notifications=True,
        loading_spinner="dots",
        # loading_color="#6A589D",
        sizing_mode="stretch_width",
        # sizing_mode="scale_width",
        js_files={
            "font-awesome": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js",
            # "bootstrap": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css",
        },
        css_files=[
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
            # "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js",
            # "https://fonts.googleapis.com/css?family=Lato&subset=latin,latin-ext",
        ],
        layout_compatibility="error",
    )

    # pn.state.notifications.position = "bottom-left"

    if app == "uploader":
        from ..pn_app import target_uploader_app as panel_app
    elif app == "admin":
        from ..pn_admin import list_files_app as panel_app

    if allow_websocket_origin is None:
        allow_websocket_origin = ["localhost"]
    # else:
    #     allow_websocket_origin = [f"{a}:{port}" for a in allow_websocket_origin]

    logger.info(f"{allow_websocket_origin=}")

    if autoreload:
        logger.info("num_procs is set to 1 as autoreload is enabled.")
        num_procs = 1

    if static_dirs is None:
        static_dirs_dict = {}
    else:
        logger.info(f"{static_dirs=}")
        for d in static_dirs:
            if "=" not in d:
                logger.error(
                    "Invalid static directory format. Please use the format 'name=path'."
                )
                return
        static_dirs_dict = {d.split("=")[0]: d.split("=")[1] for d in static_dirs}

    logger.info(f"Set maximum upload size to {max_upload_size} MB")

    logger.info(f"Starting the {app} app on port {port}.")

    if basic_auth is not None:
        if not os.path.exists(basic_auth):
            logger.error(f"Basic authentication file not found: {basic_auth}")
            return
        admin_options = dict(
            basic_auth=basic_auth,
            # cookie_secret=cookie_secret,
            basic_login_template=basic_login_template,
        )
        kwargs = admin_options
    else:
        kwargs = {}  # dict(cookie_secret=cookie_secret)

    # Ref: https://panel.holoviz.org/reference/widgets/FileInput.html#limits-defined
    pn.serve(
        panel_app,
        port=port,
        prefix=prefix,
        use_xheaders=use_xheaders,
        num_procs=num_procs,
        websocket_origin=allow_websocket_origin,
        session_token_expiration=session_token_expiration,
        static_dirs=static_dirs_dict,
        show=False,
        autoreload=autoreload,
        # Increase the maximum websocket message size allowed by Bokeh
        websocket_max_message_size=max_upload_size * 1024 * 1024,
        # Increase the maximum buffer size allowed by Tornado
        http_server_kwargs={
            "max_buffer_size": max_upload_size * 1024 * 1024,
            # "user_xheaders": use_xheaders,
        },
        **kwargs,
    )


@app.command(help="Generate a SQLite database of upload_id")
def uid2sqlite(
    input_list: Annotated[
        str, typer.Argument(show_default=False, help="Input CSV file.")
    ] = None,
    output_dir: Annotated[
        str, typer.Option("-d", "--dir", help="Output directory to save the results.")
    ] = ".",
    dbfile: Annotated[
        str,
        typer.Option(
            "--db",
            help="Filename of the SQLite database to save the upload_id.",
        ),
    ] = "upload_id.sqlite",
    scan_dir: Annotated[
        str,
        typer.Option(
            "--scan-dir",
            help="Directory to scan for the upload_id. Default is None (use input file)",
        ),
    ] = None,
    remove_duplicates: Annotated[
        bool,
        typer.Option(
            "--clean",
            help="Remove duplicates from the database. Default is False.",
        ),
    ] = False,
    log_level: Annotated[
        LogLevel, typer.Option(case_sensitive=False, help="Set the log level.")
    ] = LogLevel.INFO,
):
    logger.remove(0)
    logger.add(sys.stderr, level=log_level.value)

    db_path = os.path.join(output_dir, dbfile)

    if input_list is not None:
        df_input = pd.read_csv(input_list)
    elif scan_dir is not None:
        d = glob.glob(f"{scan_dir}/????/??/????????-??????-????????????????")
        upload_ids = [s.split("-")[-1] for s in d]
        df_input = pd.DataFrame({"upload_id": upload_ids})
    else:
        logger.warning("No input specified. Just try to create an empty database.")
        df_input = pd.DataFrame({"upload_id": []})

    print(df_input)

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    if not os.path.exists(db_path):
        logger.info(f"Creating a new SQLite database: {dbfile}")
        create_uid_db(db_path)

    if not df_input.empty:
        bulk_insert_uid_db(df_input, db_path)

    if remove_duplicates:
        remove_duplicate_uid_db(db_path)


@app.command(help="Remove duplicates from a SQLite database of upload_id")
def clean_uid(
    dbfile: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Full path to the SQLite database file.",
        ),
    ],
    backup: Annotated[
        bool,
        typer.Option(
            help="Create a backup of the database before cleaning. Default is True."
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Do not remove duplicates; just check the duplicates. Default is False."
        ),
    ] = False,
    log_level: Annotated[
        LogLevel, typer.Option(case_sensitive=False, help="Set the log level.")
    ] = LogLevel.INFO,
):
    logger.remove(0)
    logger.add(sys.stderr, level=log_level.value)

    remove_duplicate_uid_db(dbfile, backup=backup, dry_run=dry_run)
