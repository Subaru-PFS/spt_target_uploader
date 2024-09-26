#!/usr/bin/env python3

import os
from datetime import datetime, timezone

import gurobipy
import panel as pn
from dotenv import dotenv_values
from loguru import logger

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

    logger.info(f"{pn.state.headers=}")
    logger.info(f"{pn.state.location.href=}")

    config = dotenv_values(".env.shared")

    if "MAX_EXETIME" not in config.keys():
        max_exetime: int = 900
    else:
        max_exetime = int(config["MAX_EXETIME"])

    if "MAX_NPPC" not in config.keys():
        max_nppc: int = 200
    else:
        max_nppc = int(config["MAX_NPPC"])

    if "PPP_QUIET" not in config.keys():
        ppp_quiet: bool = True
    else:
        ppp_quiet: bool = bool(int(config["PPP_QUIET"]))

    if "CLUSTERING_ALGORITHM" not in config.keys():
        clustering_algorithm = "HDBSCAN"
    else:
        clustering_algorithm = config["CLUSTERING_ALGORITHM"]

    if "ANN_FILE" not in config.keys():
        ann_file = None
    elif config["ANN_FILE"] == "":
        ann_file = None
    elif not os.path.exists(config["ANN_FILE"]):
        logger.error(f"{config['ANN_FILE']} not found")
        ann_file = None
    else:
        logger.info(f"{config['ANN_FILE']} found")
        ann_file = config["ANN_FILE"]

    uid_db = config["UPLOADID_DB"] if "UPLOADID_DB" in config.keys() else None
    use_uid_db = True if uid_db is not None else False

    db_path = os.path.join(config["OUTPUT_DIR"], uid_db) if use_uid_db else None
    if use_uid_db:
        if os.path.exists(db_path):
            logger.info(f"{db_path} found")
        else:
            logger.error(f"{db_path} not found")
            raise FileNotFoundError(f"{db_path} not found")
    else:
        logger.info("No upload ID database is used. Scan output directories directly.")

    logger.info(f"Maximum execution time for the PPP is set to {max_exetime} sec.")
    logger.info(f"Maximum number of PPCs is set to {max_nppc}.")

    logger.info(f"config params from dotenv: {config}")

    if os.path.exists(config["OUTPUT_DIR"]):
        logger.info(f"{config['OUTPUT_DIR']} already exists.")
    else:
        os.makedirs(config["OUTPUT_DIR"])
        logger.info(f"{config['OUTPUT_DIR']} created.")

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

    panel_timer = TimerWidgets()

    panel_results = ValidationResultWidgets()
    panel_targets = TargetWidgets()
    panel_ppp = PppResultWidgets(exetime=max_exetime, max_nppc=max_nppc)

    panel_input.reset()
    panel_input.db_path = db_path
    panel_input.output_dir = config["OUTPUT_DIR"]
    panel_input.use_db = use_uid_db

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

    if ann_file is not None:
        panel_annoucement = AnnouncementNoteWidgets(ann_file)
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

    tab_sidebar = pn.Tabs(
        ("Home", sidebar_column),
        ("Config", sidebar_configs),
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
    def cb_validate(event):
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

        panel_timer.timer(True)

        validation_status, df_input, df_validated = panel_input.validate(
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
        )

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

        panel_timer.timer(False)

        if validation_status is None:
            return

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

        if validation_status["status"]:
            ready_to_submit = (
                panel_ppp.ppp_status
                if panel_obs_type.obs_type.value in ["queue", "classical"]
                else True
            )
            # panel_submit_button.enable_button(panel_ppp.ppp_status)
            panel_submit_button.enable_button(ready_to_submit)

    # define on_click callback for the "PPP start" button
    def cb_PPP(event):
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        _toggle_widgets(widget_set, disabled=True)

        placeholder_floatpanel.objects = []

        # reset some panels
        panel_status.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        panel_timer.timer(True)

        validation_status, df_input_, df_validated = panel_input.validate(
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
        )
        df_ppc = panel_ppcinput.validate()

        if df_ppc is None:
            _toggle_widgets(button_set, disabled=False)
            panel_timer.timer(False)
            return
        elif not df_ppc.empty:
            pn.state.notifications.info(
                "No automatic pointing determination will be performed as a user-defined pointing list is provided",
                duration=5000,  # 5sec
            )

        if validation_status is None:
            _toggle_widgets(button_set, disabled=False)
            _toggle_widgets(widget_set, disabled=False)
            panel_timer.timer(False)
            return

        if not validation_status["visibility"]["status"]:
            logger.error("No visible object is found")
            pn.state.notifications.error(
                "Cannot simulate pointing for 0 visible targets",
                duration=0,
            )
            _toggle_widgets(button_set, disabled=False)
            _toggle_widgets(widget_set, disabled=False)
            panel_timer.timer(False)
            return

        panel_status.show_results(df_validated, validation_status)
        panel_results.show_results(df_validated, validation_status)
        panel_targets.show_results(df_validated)

        tab_panels.active = 1
        tab_panels.visible = True

        try:
            panel_ppp.origname = panel_input.file_input.filename
            panel_ppp.origname_ppc = panel_ppcinput.file_input.filename
            panel_ppp.origdata = panel_input.file_input.value
            panel_ppp.origdata_ppc = panel_ppcinput.file_input.value
            panel_ppp.df_summary = panel_status.df_summary

            if not validation_status["status"]:
                logger.error("Validation failed")
                _toggle_widgets(button_set, disabled=False)
                _toggle_widgets(widget_set, disabled=False)
                panel_timer.timer(False)
                return

            panel_ppp.run_ppp(
                df_validated,
                df_ppc,
                validation_status,
                single_exptime=panel_obs_type.single_exptime.value,
                clustering_algorithm=clustering_algorithm,
                quiet=ppp_quiet,
                max_exetime=max_exetime,
            )

            panel_ppp.show_results()

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
                panel_timer.timer(False)
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

        panel_timer.timer(False)

    def cb_submit(event):
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        _toggle_widgets(widget_set, disabled=True)

        placeholder_floatpanel.objects = []

        logger.info("Submit button clicked.")
        logger.info("Validation before actually writing to the storage")

        panel_timer.timer(True)

        # do the validation again and again (input file can be different)
        # and I don't know how to implement to return value
        # from callback to another function (sorry)
        validation_status, df_input, df_validated = panel_input.validate(
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
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
                panel_timer.timer(False)
                return
            else:
                panel_status.show_results(df_validated, validation_status)
                panel_results.show_results(df_validated, validation_status)
                panel_targets.show_results(df_validated)
                tab_panels.visible = True
                panel_timer.timer(False)
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

        outdir, outfile_zip, _ = panel_ppp.upload(
            outdir_prefix=config["OUTPUT_DIR"],
            single_exptime=panel_obs_type.single_exptime.value,
            observation_type=panel_obs_type.obs_type.value,
            ppc_status=ppc_status_,
        )

        try:
            if (
                ("EMAIL_FROM" not in config.keys() or "EMAIL_FROM" == "")
                or ("EMAIL_TO" not in config.keys() or "EMAIL_TO" == "")
                or ("SMTP_SERVER" not in config.keys() or "SMTP_SERVER" == "")
            ):
                logger.warning(
                    "Email configuration is not found. No email will be sent."
                )
            else:
                send_email(
                    config,
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
            outdir.replace(config["OUTPUT_DIR"], "data/", 1).replace("//", "/"),
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

        if use_uid_db:
            single_insert_uid_db(panel_ppp.secret_token, db_path)

        panel_input.secret_token = assign_secret_token(
            db_path=db_path, output_dir=config["OUTPUT_DIR"], use_db=use_uid_db
        )

        panel_timer.timer(False)

    # set callback to the buttons
    panel_validate_button.validate.on_click(cb_validate)
    panel_ppp_button.PPPrun.on_click(cb_PPP)
    panel_submit_button.submit.on_click(cb_submit)

    app = template

    if use_panel_cli:
        return app.servable()
    else:
        return app
