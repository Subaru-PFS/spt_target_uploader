#!/usr/bin/env python3

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pprint import pformat

import gurobipy
import panel as pn
from loguru import logger

from . import __version__ as pfs_target_uploader_version
from .utils.config import get_min_fluxmag_for_obstype, load_app_config
from .utils.db import single_insert_uid_db
from .utils.mail import send_email
from .utils.session import assign_secret_token
from .widgets import (
    AnnouncementNoteWidgets,
    DatePickerWidgets,
    DocLinkWidgets,
    FileInputWidgets,
    ObsTypeWidgets,
    PPCInputWidgets,
    PppResultWidgets,
    RunPppButtonWidgets,
    StatusWidgets,
    SubmitButtonWidgets,
    TargetWidgets,
    TimerWidgets,
    UploadNoteWidgets,
    ValidateButtonWidgets,
    ValidationResultWidgets,
)


def _toggle_widgets(widgets: list, disabled: bool = True):
    for w in widgets:
        w.disabled = disabled


def target_uploader_app(use_panel_cli=False):
    pn.state.notifications.position = "bottom-left"

    # Load and validate configuration
    config = load_app_config(
        create_output_dir=True,
        validate_db=True,
        validate_ann_file=True,
    )

    # Configure logger to be multiprocessing-safe
    logger.remove()
    logger.add(sys.stderr, level=config.log_level, enqueue=True)

    # Log request headers
    if pn.state.headers:
        logger.info(f"Request headers:\n{pformat(dict(pn.state.headers), width=100)}")

    # Log request URL
    if pn.state.location and pn.state.location.href:
        logger.info(f"Request URL: {pn.state.location.href}")

    # Log application configuration
    logger.info(f"\n{config.format_for_logging()}")

    template = pn.template.MaterialTemplate(
        # template = pn.template.BootstrapTemplate(
        title="PFS Target Uploader",
        # sidebar_width=400,
        sidebar_width=420,
        header_background="#3A7D7E",
        busy_indicator=None,
        favicon="doc/assets/images/favicon.png",
    )

    # setup panel components
    panel_doc = DocLinkWidgets()

    panel_obs_type = ObsTypeWidgets()

    panel_input = FileInputWidgets()
    panel_validate_button = ValidateButtonWidgets()
    panel_status = StatusWidgets()
    panel_ppp_button = RunPppButtonWidgets()
    panel_submit_button = SubmitButtonWidgets()

    panel_dates = DatePickerWidgets()
    panel_ppcinput = PPCInputWidgets()

    panel_timer = TimerWidgets(total_seconds=config.max_exetime)

    panel_results = ValidationResultWidgets()
    panel_targets = TargetWidgets()
    panel_ppp = PppResultWidgets()

    panel_input.reset()
    panel_input.db_path = config.db_path
    panel_input.output_dir = config.output_dir
    panel_input.use_db = config.use_uid_db

    button_set = [
        panel_input.file_input,
        panel_validate_button.validate,
        panel_ppp_button.PPPrun,
    ]
    widget_set = [
        panel_obs_type.single_exptime,
        panel_obs_type.obs_type,
        panel_dates.date_begin,
        panel_dates.date_end,
        panel_ppcinput.file_input,
    ]

    placeholder_floatpanel = pn.Column(height=0, width=0)
    placeholder_announcement = pn.Column(height=0, width=0)

    if config.ann_file is not None:
        panel_annoucement = AnnouncementNoteWidgets(config.ann_file)
        placeholder_announcement[:] = [panel_annoucement.floatpanel]

    # if no file is uploaded, disable the buttons
    # This would work only at the first time the app is loaded.
    #
    # If the observatin type is 'queue' or 'classical', enable the validate and simulate buttons.
    # If the observation type is 'filler', enable only the validate button.
    def enable_buttons_by_fileinput(v, pv, obs_type):
        if v is None:
            logger.info("Buttons are disabled because no file is uploaded.")
            _toggle_widgets(
                [panel_validate_button.validate, panel_ppp_button.PPPrun],
                disabled=True,
            )
            return
        logger.info("Buttons are enabled because file upload is detected.")
        _toggle_widgets(
            [panel_validate_button.validate],
            disabled=False,
        )

        if obs_type == "queue" or obs_type == "classical":
            _toggle_widgets(
                [panel_ppp_button.PPPrun],
                disabled=False,
            )
        if obs_type == "filler":
            logger.info(
                "Simulate button is disabled because the observation type is 'filler'."
            )
            _toggle_widgets(
                [panel_ppp_button.PPPrun],
                disabled=True,
            )
        if (v is not None) and (v != pv):
            _toggle_widgets(
                [panel_submit_button.submit],
                disabled=True,
            )

    # if the observation type is 'classical', enable the exposure time widget.
    # if the observation type is 'queue' or 'filler', disable the exposure time widget and reset the file input widget.
    def toggle_classical_mode(obs_type):
        if obs_type == "classical":
            panel_obs_type.single_exptime.disabled = False
            panel_ppcinput.file_input.disabled = False
        else:
            panel_obs_type.single_exptime.disabled = True
            panel_obs_type.single_exptime.value = 900
            panel_ppcinput.file_input.disabled = True
            panel_ppcinput.file_input.filename = None
            panel_ppcinput.file_input.value = None

    fileinput_watcher = pn.bind(
        enable_buttons_by_fileinput,
        panel_input.file_input,
        panel_input.previous_value,
        panel_obs_type.obs_type,
    )

    ppcinput_watcher = pn.bind(toggle_classical_mode, panel_obs_type.obs_type)

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        pn.Column(
            panel_obs_type.obstype_pane,
            margin=(10, 0, 0, 0),
        ),
        pn.Column(
            # pn.Row("<font size=4>**Select an operation**</font>", panel_timer.pane),
            pn.Row(
                "<font size=4><i class='fas fa-calculator'></i> **Execute an operation**</font>",
                panel_timer.pane,
            ),
            pn.Row(
                panel_validate_button.pane,
                panel_ppp_button.pane,
                panel_submit_button.pane,
                sizing_mode="stretch_width",
            ),
            margin=(10, 0, 0, 0),
        ),
        pn.Column(
            pn.Row(
                "<font size=4><i class='fa-solid fa-magnifying-glass-chart fa-lg'></i> **Validation status**</font>"
            ),
            panel_status.pane,
            margin=(10, 0, 0, 0),
        ),
        fileinput_watcher,
    )

    sidebar_configs = pn.Column(
        pn.Column(
            panel_dates.pane,
            margin=(10, 0, 0, 0),
        ),
        pn.Column(
            panel_obs_type.exptime_pane,
            margin=(10, 0, 0, 0),
        ),
        pn.Column(
            panel_ppcinput.pane,
            margin=(10, 0, 0, 0),
        ),
        ppcinput_watcher,
    )

    sidebar_about = pn.Column(
        pn.Column(
            pn.pane.Markdown(
                "<font size=4><i class='fas fa-tag'></i>  **Version**</font>",
                sizing_mode="stretch_width",
            ),
        ),
        pn.Column(
            pn.pane.Markdown(
                f"{pfs_target_uploader_version}",
                sizing_mode="stretch_width",
            ),
            margin=(-20, 0, 0, 0),
        ),
        margin=(10, 0, 0, 0),
    )

    tab_sidebar = pn.Tabs(
        ("Home", sidebar_column),
        ("Config", sidebar_configs),
        ("About", sidebar_about),
    )
    # tab_sidebar.active = 1

    # bundle panel(s) in the main area
    tab_panels = pn.Tabs(
        ("Input list", panel_targets.pane),
        ("Validation", panel_results.pane),
        ("Pointing Simulation", panel_ppp.pane),
    )

    sidepanel_column = pn.Column(
        panel_doc.pane,
        tab_sidebar,
    )

    main_column = pn.Column(
        placeholder_floatpanel,
        placeholder_announcement,
        tab_panels,
        margin=(30, 0, 0, 0),
    )

    # put them into the template
    # template.sidebar.append(panel_doc.pane)
    template.sidebar.append(sidepanel_column)
    template.main.append(main_column)

    tab_panels.visible = False

    # define on_click callback for the "validate" button
    async def cb_validate(event):
        # disable the buttons and input file widget while validation
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        _toggle_widgets(widget_set, disabled=True)

        placeholder_floatpanel.objects = []

        tab_panels.visible = False

        panel_status.reset()
        panel_results.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        panel_timer.timer(on=True, time_limit=False)

        # Select min_mag based on observation type
        effective_min_mag = get_min_fluxmag_for_obstype(
            panel_obs_type.obs_type.value,
            config,
        )

        validation_status, df_input, df_validated = await asyncio.to_thread(
            panel_input.validate,
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
            single_exptime=panel_obs_type.single_exptime.value,
            min_mag=effective_min_mag,
            max_mag=config.max_fluxmag,
        )

        _toggle_widgets(widget_set, disabled=False)
        _toggle_widgets(button_set, disabled=False)

        if validation_status is None:
            panel_timer.timer(on=False, time_limit=False)
            return

        if panel_obs_type.obs_type.value == "queue":
            _toggle_widgets(
                [panel_obs_type.single_exptime, panel_ppcinput.file_input],
                disabled=True,
            )
        if panel_obs_type.obs_type.value == "filler":
            _toggle_widgets(
                [
                    panel_ppp_button.PPPrun,
                    panel_obs_type.single_exptime,
                    panel_ppcinput.file_input,
                ],
                disabled=True,
            )

        panel_status.show_results(df_validated, validation_status)
        panel_targets.show_results(df_validated)
        panel_results.show_results(df_validated, validation_status)

        panel_ppp.df_input = df_validated
        try:
            panel_ppp.df_summary = panel_status.df_summary
        except AttributeError as e:
            logger.error(f"{str(e)}")
            pass

        tab_panels.active = 1
        tab_panels.visible = True

        panel_timer.timer(on=False, time_limit=False)

        if validation_status["status"]:
            ready_to_submit = (
                panel_ppp.ppp_status
                if panel_obs_type.obs_type.value in ["queue", "classical"]
                else True
            )
            # panel_submit_button.enable_button(panel_ppp.ppp_status)
            panel_submit_button.enable_button(ready_to_submit)

    # define on_click callback for the "PPP start" button
    async def cb_PPP(event):
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        _toggle_widgets(widget_set, disabled=True)

        placeholder_floatpanel.objects = []

        # reset some panels
        panel_status.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        panel_timer.timer(on=True, time_limit=False)

        # Select min_mag based on observation type
        effective_min_mag = get_min_fluxmag_for_obstype(
            panel_obs_type.obs_type.value,
            config,
        )

        validation_status, df_input_, df_validated = await asyncio.to_thread(
            panel_input.validate,
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
            single_exptime=panel_obs_type.single_exptime.value,
            min_mag=effective_min_mag,
            max_mag=config.max_fluxmag,
        )
        df_ppc = await asyncio.to_thread(panel_ppcinput.validate)

        if df_ppc is None:
            _toggle_widgets(button_set, disabled=False)
            panel_timer.timer(on=False, time_limit=False)
            return
        elif not df_ppc.empty:
            pn.state.notifications.info(
                "No automatic pointing determination will be performed as a user-defined pointing list is provided",
                duration=5000,  # 5sec
            )

        if validation_status is None:
            _toggle_widgets(button_set, disabled=False)
            _toggle_widgets(widget_set, disabled=False)
            panel_timer.timer(on=False, time_limit=False)
            return

        if not validation_status["visibility"]["status"]:
            logger.error("No visible object is found")
            pn.state.notifications.error(
                "Cannot simulate pointing for 0 visible targets",
                duration=0,
            )
            _toggle_widgets(button_set, disabled=False)
            _toggle_widgets(widget_set, disabled=False)
            panel_timer.timer(on=False, time_limit=False)
            return

        panel_status.show_results(df_validated, validation_status)
        panel_results.show_results(df_validated, validation_status)
        panel_targets.show_results(df_validated)

        panel_timer.timer(on=False, time_limit=False)

        tab_panels.active = 1
        tab_panels.visible = True

        try:
            panel_timer.timer(on=True, time_limit=True)

            panel_ppp.origname = panel_input.file_input.filename
            panel_ppp.origname_ppc = panel_ppcinput.file_input.filename
            panel_ppp.origdata = panel_input.file_input.value
            panel_ppp.origdata_ppc = panel_ppcinput.file_input.value
            panel_ppp.df_summary = panel_status.df_summary

            if not validation_status["status"]:
                logger.error("Validation failed")
                _toggle_widgets(button_set, disabled=False)
                _toggle_widgets(widget_set, disabled=False)
                panel_timer.timer(on=False, time_limit=True)
                return

            dt_now = datetime.now()
            if config.max_exetime > 0:
                pn.state.notifications.info(
                    f"Pointing simulation started at {dt_now.strftime('%H:%M:%S')} and last up for about {int(config.max_exetime/60)} minutes",
                    duration=0,
                )
            else:
                pn.state.notifications.info(
                    f"Pointing simulation started at {dt_now.strftime('%H:%M:%S')}",
                    duration=0,
                )

            await asyncio.to_thread(
                panel_ppp.run_ppp,
                df_validated,
                df_ppc,
                validation_status,
                single_exptime=panel_obs_type.single_exptime.value,
                clustering_algorithm=config.clustering_algorithm,
                quiet=config.ppp_quiet,
                max_exetime=config.max_exetime,
                logger=logger,
                solver_backend=config.solver_backend,
            )

            await asyncio.to_thread(panel_ppp.show_results)

            tab_panels.active = 2

            # enable the submit button only with the successful validation
            if validation_status["status"]:
                panel_submit_button.enable_button(panel_ppp.ppp_status)
                panel_submit_button.submit.disabled = False

            if panel_ppp.nppc is None:
                logger.error("Pointing simulation failed")
                _toggle_widgets(button_set, disabled=False)
                _toggle_widgets(widget_set, disabled=False)
                _toggle_widgets([panel_submit_button.submit], disabled=True)
                panel_timer.timer(on=False, time_limit=True)
                return

        except gurobipy.GurobiError as e:
            pn.state.notifications.error(f"{str(e)}", duration=0)
            pass

        _toggle_widgets(widget_set, disabled=False)
        _toggle_widgets(button_set, disabled=False)
        if panel_obs_type.obs_type.value == "queue":
            _toggle_widgets(
                [panel_obs_type.single_exptime, panel_ppcinput.file_input],
                disabled=True,
            )
        if panel_obs_type.obs_type.value == "filler":
            _toggle_widgets(
                [
                    panel_ppp_button.PPPrun,
                    panel_obs_type.single_exptime,
                    panel_ppcinput.file_input,
                ],
                disabled=True,
            )

        panel_timer.timer(on=False, time_limit=True)

    async def cb_submit(event):
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        _toggle_widgets(widget_set, disabled=True)

        placeholder_floatpanel.objects = []

        logger.info("Submit button clicked.")
        logger.info("Validation before actually writing to the storage")

        panel_timer.timer(on=True, time_limit=False)

        # do the validation again and again (input file can be different)
        # and I don't know how to implement to return value
        # from callback to another function (sorry)
        # Select min_mag based on observation type
        effective_min_mag = get_min_fluxmag_for_obstype(
            panel_obs_type.obs_type.value,
            config,
        )

        validation_status, df_input, df_validated = await asyncio.to_thread(
            panel_input.validate,
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
            min_mag=effective_min_mag,
            max_mag=config.max_fluxmag,
        )

        if (validation_status is None) or (not validation_status["status"]):
            logger.error("Validation failed for some reason")

            tab_panels.visible = False

            panel_status.reset()
            panel_results.reset()

            pn.state.notifications.clear()

            _toggle_widgets(widget_set, disabled=False)
            _toggle_widgets(button_set, disabled=False)

            if validation_status is None:
                panel_timer.timer(on=False, time_limit=False)
                return
            else:
                panel_status.show_results(df_validated, validation_status)
                panel_results.show_results(df_validated, validation_status)
                panel_targets.show_results(df_validated)
                tab_panels.visible = True
                panel_timer.timer(on=False, time_limit=False)
                return

        panel_ppp.df_input = df_validated
        panel_ppp.df_summary = panel_status.df_summary
        panel_ppp.origname = panel_input.file_input.filename
        panel_ppp.origname_ppc = panel_ppcinput.file_input.filename
        panel_ppp.origdata = panel_input.file_input.value
        panel_ppp.origdata_ppc = panel_ppcinput.file_input.value
        panel_ppp.upload_time = datetime.now(timezone.utc)
        panel_ppp.secret_token = panel_input.secret_token

        if panel_ppp.status_ == 2:
            ppc_status_ = "user"
        elif panel_ppp.status_ == 0:
            ppc_status_ = "skip"
        else:
            ppc_status_ = "auto"

        outdir, outfile_zip, _ = await asyncio.to_thread(
            panel_ppp.upload,
            outdir_prefix=config.output_dir,
            single_exptime=panel_obs_type.single_exptime.value,
            observation_type=panel_obs_type.obs_type.value,
            ppc_status=ppc_status_,
        )

        try:
            if (
                (
                    "EMAIL_FROM" not in config.raw_config.keys()
                    or config.raw_config["EMAIL_FROM"] == ""
                )
                or (
                    "EMAIL_TO" not in config.raw_config.keys()
                    or config.raw_config["EMAIL_TO"] == ""
                )
                or (
                    "SMTP_SERVER" not in config.raw_config.keys()
                    or config.raw_config["SMTP_SERVER"] == ""
                )
            ):
                logger.warning(
                    "Email configuration is not found. No email will be sent."
                )
            else:
                await asyncio.to_thread(
                    send_email,
                    config.raw_config,
                    outdir=outdir,
                    outfile=outfile_zip,
                    upload_id=panel_ppp.secret_token,
                    upload_time=panel_ppp.upload_time,
                    url=pn.state.location.href,
                )
        except Exception as e:
            logger.error(f"Failed to send an email: {str(e)}")

        panel_notes = UploadNoteWidgets(
            panel_ppp.secret_token,
            panel_ppp.upload_time,
            panel_ppp.ppp_status,
            outdir.replace(config.output_dir, "data/", 1).replace("//", "/"),
            outfile_zip,
        )
        placeholder_floatpanel[:] = [panel_notes.floatpanel]

        _toggle_widgets(widget_set, disabled=False)
        _toggle_widgets(button_set, disabled=False)
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        if panel_obs_type.obs_type.value == "queue":
            _toggle_widgets(
                [panel_obs_type.single_exptime, panel_ppcinput.file_input],
                disabled=True,
            )
        if panel_obs_type.obs_type.value == "filler":
            _toggle_widgets(
                [
                    panel_ppp_button.PPPrun,
                    panel_obs_type.single_exptime,
                    panel_ppcinput.file_input,
                ],
                disabled=True,
            )

        if config.use_uid_db:
            await asyncio.to_thread(
                single_insert_uid_db, panel_ppp.secret_token, config.db_path
            )

        panel_input.secret_token = await asyncio.to_thread(
            assign_secret_token,
            db_path=config.db_path,
            output_dir=config.output_dir,
            use_db=config.use_uid_db,
        )

        panel_timer.timer(on=False, time_limit=False)

    # set callback to the buttons
    panel_validate_button.validate.on_click(cb_validate)
    panel_ppp_button.PPPrun.on_click(cb_PPP)
    panel_submit_button.submit.on_click(cb_submit)

    app = template

    if use_panel_cli:
        return app.servable()
    else:
        return app
