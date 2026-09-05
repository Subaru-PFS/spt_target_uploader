"""Main Panel web app: wires the upload, validation, PPP and submission widgets together."""

import asyncio
import sys
import weakref
from datetime import UTC, datetime
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

_active_ppp_widgets = weakref.WeakSet()


def terminate_active_ppp_runs():
    """Stop PPP processes still running in any active web-app session."""
    for panel_ppp in list(_active_ppp_widgets):
        panel_ppp.terminate_active_ppp()


def _toggle_widgets(widgets: list, disabled: bool = True):
    for w in widgets:
        w.disabled = disabled


def run_config_matches(completed_config, current_config):
    """Return whether the editable inputs still match a completed PPP run."""
    return completed_config == current_config


def relock_run_config_widgets(
    obs_type_widgets,
    ppc_input_widgets,
    date_widgets,
    ppp_button_widgets,
):
    """Re-apply sidebar constraints that are inherent to an observation mode.

    cb_validate, cb_PPP and cb_submit re-enable the whole sidebar when they
    finish. This re-applies only the locks that must outlast that blanket
    re-enable:

    - queue / filler pin ``single_exptime`` and the user pointing list to
      their fixed values regardless of anything else; filler also keeps the
      Simulate button off.

    Only ever disables; each callback owns the matching re-enable.
    """
    obs_type = obs_type_widgets.obs_type.value
    if obs_type == "filler":
        _toggle_widgets([ppp_button_widgets.PPPrun], disabled=True)
    if obs_type in ("queue", "filler"):
        _toggle_widgets(
            [obs_type_widgets.single_exptime, ppc_input_widgets.file_input],
            disabled=True,
        )


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
        title="PFS Target Uploader",
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
    _active_ppp_widgets.add(panel_ppp)
    pn.state.on_session_destroyed(
        lambda _session_context: panel_ppp.terminate_active_ppp()
    )

    panel_input.reset()
    panel_input.db_path = config.db_path
    panel_input.output_dir = config.output_dir
    panel_input.use_db = config.use_uid_db

    # Which validation result the three panels currently display. cb_validate
    # and cb_PPP both render the same data, and that is 2.7 s of Tabulator
    # work on a 30,000-row list. None means "nothing trustworthy on screen".
    rendered_validation = {"key": None}
    pending_run = {"config": None, "stale": False}

    def reset_validation_panels():
        """The only sanctioned way to blank the validation panels.

        The render gate is correct only while "the panels were blanked"
        implies "the key was cleared". Routing every reset through here keeps
        that from being a rule spread across four call sites -- forget it once
        and the gate refuses to redraw a panel it believes is still current,
        which the user cannot recover from without reloading the page.

        It blanks the same three panels render_validation_results() writes.
        Leaving panel_targets out (as the code did before these two became a
        pair) strands the previous file's rows in the "Input list" tab on
        every path that resets and then early-returns -- cb_PPP's "0 visible
        targets", above all, which does not hide the tabs either.
        """
        panel_status.reset()
        panel_results.reset()
        panel_targets.reset()
        rendered_validation["key"] = None

    def render_validation_results(df_validated, validation_status):
        # last_validation_key describes the most recent validate() call: a key
        # on success, None on failure. Never a stale key from an earlier run.
        key = panel_input.last_validation_key
        if key is not None and key == rendered_validation["key"]:
            logger.info(
                "Validation panels already show this result; skipping the re-render."
            )
            return
        panel_status.show_results(df_validated, validation_status)
        panel_results.show_results(df_validated, validation_status)
        panel_targets.show_results(df_validated)
        rendered_validation["key"] = key

    def reset_simulation_result():
        """Discard a PPP result that can no longer be submitted."""
        pending_run["config"] = None
        pending_run["stale"] = False
        panel_ppp.reset()

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

    def relock_config_widgets():
        relock_run_config_widgets(
            panel_obs_type,
            panel_ppcinput,
            panel_dates,
            panel_ppp_button,
        )

    def restore_controls():
        _toggle_widgets(widget_set, disabled=False)
        _toggle_widgets(button_set, disabled=False)
        relock_config_widgets()

    def target_input_identity():
        return (
            panel_input.file_input.filename,
            panel_input.file_input.mime_type,
            panel_input.file_input.value,
        )

    def current_run_config():
        observation_type = panel_obs_type.obs_type.value
        return {
            "observation_type": observation_type,
            "date_begin": panel_dates.date_begin.value,
            "date_end": panel_dates.date_end.value,
            "single_exptime": panel_obs_type.single_exptime.value,
            "min_mag": get_min_fluxmag_for_obstype(observation_type, config),
            "max_mag": config.max_fluxmag,
            "target_input": target_input_identity(),
            "pointing_input": (
                panel_ppcinput.file_input.filename,
                panel_ppcinput.file_input.value,
            ),
        }

    def target_input_matches_pending_run():
        run_config = pending_run["config"]
        return (
            run_config is None or run_config["target_input"] == target_input_identity()
        )

    def pending_run_matches_current_config():
        run_config = pending_run["config"]
        return run_config is not None and run_config_matches(
            run_config, current_run_config()
        )

    def notify_config_changed_after_simulation():
        notifications = pn.state.notifications
        if notifications is not None:
            notifications.warning(
                "The simulation result uses different configuration values. "
                "Simulate again before submitting.",
                duration=5000,
            )

    def refresh_pending_run_submission_state(*_events):
        """Mark a prior result when Config no longer matches its input snapshot."""
        if pending_run["config"] is None:
            return

        is_current = pending_run_matches_current_config()
        was_stale = pending_run["stale"]
        pending_run["stale"] = not is_current
        panel_submit_button.set_warning(not is_current)

        if not is_current and not was_stale:
            notify_config_changed_after_simulation()

    # Takes *args: this watches three parameters, and a browser upload changes
    # all three in one batched param.update(), which param delivers as one call
    # carrying one Event per changed parameter. A single-argument signature
    # raises TypeError there, and the exception aborts param's watcher loop
    # before the binding that enables Validate/Simulate can run -- the file
    # arrives and every button stays disabled.
    def discard_pending_run_if_target_input_changed(*_events):
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        if target_input_matches_pending_run():
            return
        logger.info("Discarding the simulation result because the target list changed.")
        reset_simulation_result()
        notify_target_input_changed_after_simulation()
        _toggle_widgets(widget_set, disabled=False)
        relock_config_widgets()

    def notify_target_input_changed_after_simulation():
        notifications = pn.state.notifications
        if notifications is not None:
            notifications.warning(
                "The target list changed after the simulation. "
                "Validate and simulate it again before submitting.",
                duration=0,
            )

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
    def enable_buttons_by_fileinput(v, obs_type):
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
        panel_obs_type.obs_type,
    )

    ppcinput_watcher = pn.bind(toggle_classical_mode, panel_obs_type.obs_type)
    panel_input.file_input.param.watch(
        discard_pending_run_if_target_input_changed,
        ["filename", "mime_type", "value"],
    )
    panel_obs_type.obs_type.param.watch(refresh_pending_run_submission_state, "value")
    panel_obs_type.single_exptime.param.watch(
        refresh_pending_run_submission_state, "value"
    )
    panel_dates.date_begin.param.watch(refresh_pending_run_submission_state, "value")
    panel_dates.date_end.param.watch(refresh_pending_run_submission_state, "value")
    panel_ppcinput.file_input.param.watch(
        refresh_pending_run_submission_state, ["filename", "value"]
    )

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        pn.Column(
            panel_obs_type.obstype_pane,
            margin=(10, 0, 0, 0),
        ),
        pn.Column(
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

        observation_type = panel_obs_type.obs_type.value
        effective_min_mag = get_min_fluxmag_for_obstype(
            observation_type,
            config,
        )
        validation_args = {
            "date_begin": panel_dates.date_begin.value,
            "date_end": panel_dates.date_end.value,
            "single_exptime": panel_obs_type.single_exptime.value,
            "min_mag": effective_min_mag,
            "max_mag": config.max_fluxmag,
        }

        # Blank the panels only when a real validation is about to run: a
        # cached one returns instantly, so there is nothing stale to hide.
        if not panel_input.has_cached_validation(
            **validation_args,
        ):
            reset_validation_panels()
        reset_simulation_result()

        pn.state.notifications.clear()

        panel_timer.timer(on=True, time_limit=False)

        validation_status, df_validated = await asyncio.to_thread(
            panel_input.validate,
            **validation_args,
        )

        restore_controls()

        if validation_status is None:
            panel_timer.timer(on=False, time_limit=False)
            return

        render_validation_results(df_validated, validation_status)

        panel_ppp.df_input = df_validated
        try:
            panel_ppp.df_summary = panel_status.df_summary
        except AttributeError as e:
            logger.error(f"{e!s}")

        tab_panels.active = 1
        tab_panels.visible = True

        panel_timer.timer(on=False, time_limit=False)

        if validation_status["status"]:
            ready_to_submit = (
                panel_ppp.ppp_status
                if observation_type in ["queue", "classical"]
                else True
            )
            panel_submit_button.enable_button(ready_to_submit)

    # define on_click callback for the "PPP start" button
    async def cb_PPP(event):
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        _toggle_widgets(widget_set, disabled=True)

        placeholder_floatpanel.objects = []

        run_config = current_run_config()
        validation_args = {
            key: run_config[key]
            for key in (
                "date_begin",
                "date_end",
                "single_exptime",
                "min_mag",
                "max_mag",
            )
        }

        # reset some panels -- but only when a real validation is about to
        # run, otherwise the status pane would be blanked and then skipped
        # by the render gate below.
        if not panel_input.has_cached_validation(
            **validation_args,
        ):
            reset_validation_panels()
        reset_simulation_result()

        pn.state.notifications.clear()

        panel_timer.timer(on=True, time_limit=False)

        validation_status, df_validated = await asyncio.to_thread(
            panel_input.validate,
            **validation_args,
        )
        df_ppc = await asyncio.to_thread(panel_ppcinput.validate)

        if df_ppc is None:
            restore_controls()
            panel_timer.timer(on=False, time_limit=False)
            return
        elif not df_ppc.empty:
            pn.state.notifications.info(
                "No automatic pointing determination will be performed as a user-defined pointing list is provided",
                duration=5000,  # 5sec
            )

        if validation_status is None:
            restore_controls()
            panel_timer.timer(on=False, time_limit=False)
            return

        # Validation can stop before the visibility step is ever reached
        # (missing required columns, a header-only file, out-of-range values).
        # Render that error in the panels the way cb_validate does, instead of
        # falling into the visibility check below whose "0 visible targets"
        # message would be misleading here.
        if validation_status["visibility"]["status"] is None:
            logger.error("Validation failed before the visibility check")
            render_validation_results(df_validated, validation_status)
            tab_panels.active = 1
            tab_panels.visible = True
            restore_controls()
            panel_timer.timer(on=False, time_limit=False)
            return

        if not validation_status["visibility"]["status"]:
            logger.error("No visible object is found")
            pn.state.notifications.error(
                "Cannot simulate pointing for 0 visible targets",
                duration=0,
            )
            restore_controls()
            panel_timer.timer(on=False, time_limit=False)
            return

        render_validation_results(df_validated, validation_status)

        panel_timer.timer(on=False, time_limit=False)

        tab_panels.active = 1
        tab_panels.visible = True

        try:
            panel_timer.timer(on=True, time_limit=True)

            panel_ppp.origname = run_config["target_input"][0]
            panel_ppp.origname_ppc = run_config["pointing_input"][0]
            panel_ppp.origdata = run_config["target_input"][2]
            panel_ppp.origdata_ppc = run_config["pointing_input"][1]
            panel_ppp.df_summary = panel_status.df_summary

            if not validation_status["status"]:
                logger.error("Validation failed")
                restore_controls()
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
                single_exptime=run_config["single_exptime"],
                clustering_algorithm=config.clustering_algorithm,
                quiet=config.ppp_quiet,
                max_exetime=config.max_exetime,
                logger=logger,
                solver_backend=config.solver_backend,
                timing_verbose=config.ppp_timing_verbose,
            )

            await asyncio.to_thread(panel_ppp.show_results)

            tab_panels.active = 2

            if panel_ppp.nppc is None:
                logger.error("Pointing simulation failed")
                restore_controls()
                _toggle_widgets([panel_submit_button.submit], disabled=True)
                panel_timer.timer(on=False, time_limit=True)
                return

            pending_run["config"] = run_config
            pending_run["stale"] = False
            panel_submit_button.enable_button(panel_ppp.ppp_status)
            panel_submit_button.submit.disabled = False

        except gurobipy.GurobiError as e:
            pn.state.notifications.error(f"{e!s}", duration=0)

        restore_controls()

        panel_timer.timer(on=False, time_limit=True)

    async def cb_submit(event):
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)
        _toggle_widgets(widget_set, disabled=True)

        placeholder_floatpanel.objects = []

        logger.info("Submit button clicked.")
        logger.info("Validation before actually writing to the storage")

        submission_config = pending_run["config"]
        has_pending_run = submission_config is not None
        if has_pending_run and not target_input_matches_pending_run():
            logger.error("Target list changed after the simulation result was created")
            notify_target_input_changed_after_simulation()
            reset_simulation_result()
            restore_controls()
            panel_timer.timer(on=False, time_limit=False)
            return
        # Clear stale notifications up front, as cb_validate() and cb_PPP() do.
        # Clearing after panel_input.validate() has run would wipe the sticky
        # notifications it raises to report a failure.
        pn.state.notifications.clear()

        panel_timer.timer(on=True, time_limit=False)

        if submission_config is None:
            observation_type = panel_obs_type.obs_type.value
            effective_min_mag = get_min_fluxmag_for_obstype(observation_type, config)
            submission_config = {
                "observation_type": observation_type,
                "date_begin": panel_dates.date_begin.value,
                "date_end": panel_dates.date_end.value,
                "single_exptime": panel_obs_type.single_exptime.value,
                "min_mag": effective_min_mag,
                "max_mag": config.max_fluxmag,
            }
        validation_args = {
            key: submission_config[key]
            for key in (
                "date_begin",
                "date_end",
                "single_exptime",
                "min_mag",
                "max_mag",
            )
        }

        validation_status, df_validated = await asyncio.to_thread(
            panel_input.validate,
            **validation_args,
        )

        if has_pending_run and pending_run["config"] is not submission_config:
            logger.error("Simulation result changed while it was validated")
            notify_target_input_changed_after_simulation()
            reset_simulation_result()
            restore_controls()
            panel_timer.timer(on=False, time_limit=False)
            return

        if (validation_status is None) or (not validation_status["status"]):
            logger.error("Validation failed for some reason")

            tab_panels.visible = False

            reset_validation_panels()
            reset_simulation_result()

            restore_controls()

            if validation_status is None:
                panel_timer.timer(on=False, time_limit=False)
                return
            else:
                render_validation_results(df_validated, validation_status)
                tab_panels.visible = True
                panel_timer.timer(on=False, time_limit=False)
                return

        panel_ppp.df_input = df_validated
        panel_ppp.df_summary = panel_status.df_summary
        panel_ppp.origname = panel_input.file_input.filename
        panel_ppp.origdata = panel_input.file_input.value
        if pending_run["config"] is None:
            panel_ppp.origname_ppc = panel_ppcinput.file_input.filename
            panel_ppp.origdata_ppc = panel_ppcinput.file_input.value
        panel_ppp.upload_time = datetime.now(UTC)
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
            single_exptime=submission_config["single_exptime"],
            observation_type=submission_config["observation_type"],
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
            logger.error(f"Failed to send an email: {e!s}")

        panel_notes = UploadNoteWidgets(
            panel_ppp.secret_token,
            panel_ppp.upload_time,
            panel_ppp.ppp_status,
            outdir.replace(config.output_dir, "data/", 1).replace("//", "/"),
            outfile_zip,
        )
        placeholder_floatpanel[:] = [panel_notes.floatpanel]

        pending_run["config"] = None
        restore_controls()
        _toggle_widgets([panel_submit_button.submit], disabled=True)

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
        # The token is part of the cache key, so that entry can never be hit
        # again -- but it would keep its frames alive until the next
        # validation, which for an idle session is never.
        panel_input.invalidate_cache()

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
