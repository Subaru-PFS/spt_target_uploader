#!/usr/bin/env python3

import os
from datetime import datetime, timezone

import gurobipy
import numpy as np
import panel as pn
from astropy.table import Table
from dotenv import dotenv_values
from logzero import logger

from .utils.io import load_file_properties, upload_file
from .widgets import (
    DatePickerWidgets,
    DocLinkWidgets,
    FileInputWidgets,
    PppResultWidgets,
    RunPppButtonWidgets,
    StatusWidgets,
    SubmitButtonWidgets,
    TargetWidgets,
    UploadNoteWidgets,
    ValidateButtonWidgets,
    ValidationResultWidgets,
)


def _toggle_buttons(buttons: list, disabled: bool = True):
    for b in buttons:
        b.disabled = disabled


def target_uploader_app():
    config = dotenv_values(".env.shared")

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
        favicon="docs/site/assets/images/favicon.png",
    )

    # setup panel components
    panel_doc = DocLinkWidgets()

    panel_input = FileInputWidgets()
    panel_validate_button = ValidateButtonWidgets()
    panel_status = StatusWidgets()
    panel_ppp_button = RunPppButtonWidgets()
    panel_submit_button = SubmitButtonWidgets()

    panel_dates = DatePickerWidgets()

    panel_results = ValidationResultWidgets()
    panel_targets = TargetWidgets()
    panel_ppp = PppResultWidgets()

    panel_input.reset()

    button_set = [
        panel_input.file_input,
        panel_validate_button.validate,
        panel_ppp_button.PPPrun,
    ]

    placeholder_floatpanel = pn.Column(height=0, width=0)

    loading_spinner = pn.indicators.LoadingSpinner(
        value=False, size=50, margin=(10, 0, 0, 0), color="secondary"
    )

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        pn.Column(
            pn.pane.Markdown(
                "<font size=5>**Select an operation**</font>",
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
            pn.Row("<font size=5>**Validation status**</font>", loading_spinner),
            panel_status.pane,
            margin=(10, 0, 0, 0),
        ),
    )

    sidebar_configs = pn.Column(panel_dates.pane)

    tab_sidebar = pn.Tabs(
        ("Home", sidebar_column),
        ("Config", sidebar_configs),
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
        _toggle_buttons(button_set, disabled=True)

        placeholder_floatpanel.objects = []

        tab_panels.visible = False

        panel_status.reset()
        panel_results.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        loading_spinner.value = True
        validation_status, df_input, df_output = panel_input.validate(
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
        )

        _toggle_buttons(button_set, disabled=False)
        loading_spinner.value = False

        if validation_status is None:
            return

        panel_status.show_results(df_output, validation_status)
        panel_targets.show_results(df_output)
        panel_results.show_results(df_output, validation_status)

        tab_panels.active = 1
        tab_panels.visible = True

        if validation_status["status"]:
            panel_submit_button.enable_button(panel_ppp.ppp_status)
            # panel_submit_button.submit.disabled = False

    # define on_click callback for the "PPP start" button
    def cb_PPP(event):
        _toggle_buttons(button_set, disabled=True)
        panel_submit_button.submit.disabled = True

        placeholder_floatpanel.objects = []

        # reset some panels
        panel_status.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        loading_spinner.value = True

        validation_status, df_input_, df_validated = panel_input.validate(
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
        )

        if validation_status is None:
            _toggle_buttons(button_set, disabled=False)
            loading_spinner.value = False
            return

        if not validation_status["visibility"]["status"]:
            logger.error("No visible object is found")
            pn.state.notifications.error(
                "Cannot simulate pointing for 0 visible targets",
                duration=0,
            )
            _toggle_buttons(button_set, disabled=False)
            loading_spinner.value = False
            return

        panel_status.show_results(df_validated, validation_status)
        panel_results.show_results(df_validated, validation_status)
        panel_targets.show_results(df_validated)

        tab_panels.active = 1
        tab_panels.visible = True

        # start progress icon
        panel_ppp_button.start()

        try:
            panel_ppp.origname = panel_input.file_input.filename
            panel_ppp.origdata = panel_input.file_input.value
            panel_ppp.df_summary = panel_status.df_summary

            panel_ppp.run_ppp(df_validated, validation_status)
            panel_ppp.show_results()

            tab_panels.active = 2

            # enable the submit button only with the successful validation
            if validation_status["status"]:
                panel_submit_button.enable_button(panel_ppp.ppp_status)
                # panel_submit_button.submit.disabled = False

        except gurobipy.GurobiError as e:
            pn.state.notifications.error(f"{str(e)}", duration=0)
            pass

        panel_ppp_button.stop()

        _toggle_buttons(button_set, disabled=False)
        loading_spinner.value = False

    def cb_submit(event):
        panel_submit_button.submit.disabled = True

        placeholder_floatpanel.objects = []

        logger.info("Submit button clicked.")
        logger.info("Validation before actually writing to the storage")

        loading_spinner.value = True

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

            if validation_status is None:
                loading_spinner.value = False
                return
            else:
                panel_status.show_results(df_validated, validation_status)
                panel_results.show_results(df_validated, validation_status)
                panel_targets.show_results(df_validated)
                tab_panels.visible = True
                loading_spinner.value = False
                return

        panel_ppp.origname = panel_input.file_input.filename
        panel_ppp.origdata = panel_input.file_input.value
        panel_ppp.df_summary = panel_status.df_summary

        upload_time = datetime.now(timezone.utc)
        secret_token = panel_input.secret_token

        try:
            df_psl = panel_ppp.p_result_tab.value
            df_ppc = panel_ppp.p_result_ppc.value
            ppp_fig = panel_ppp.p_result_fig
            # ppp_fig = panel_ppp.ppp_figure
        except AttributeError:
            df_psl = None
            df_ppc = None
            ppp_fig = None

        outdir, outfile_zip, _ = upload_file(
            df_validated,
            df_psl,
            df_ppc,
            panel_status.df_summary,
            ppp_fig,
            outdir_prefix=config["OUTPUT_DIR"],
            origname=panel_input.file_input.filename,
            origdata=panel_input.file_input.value,
            secret_token=secret_token,
            upload_time=upload_time,
            ppp_status=panel_ppp.ppp_status,
        )
        panel_notes = UploadNoteWidgets(
            secret_token,
            upload_time,
            panel_ppp.ppp_status,
            outdir.replace(config["OUTPUT_DIR"], "data", 1),
            outfile_zip,
        )
        placeholder_floatpanel[:] = [panel_notes.floatpanel]

        panel_submit_button.submit.disabled = True
        loading_spinner.value = False

    # set callback to the buttons
    panel_validate_button.validate.on_click(cb_validate)
    panel_ppp_button.PPPrun.on_click(cb_PPP)
    panel_submit_button.submit.on_click(cb_submit)

    app = template.servable()

    return app


#
# admin app
#
def list_files_app():
    config = dotenv_values(".env.shared")

    logger.info(f"config params from dotenv: {config}")

    if not os.path.exists(config["OUTPUT_DIR"]):
        logger.error(f"{config['OUTPUT_DIR']} not found")
        raise ValueError

    template = pn.template.VanillaTemplate(
        title="PFS Target & Proposal Lists",
        # collapsed_sidebar=True,
        # header_background="#3A7D7E",
        # header_background="#C71585",  # mediumvioletred
        header_background="#dc143c",  # crimson
        busy_indicator=None,
        favicon="docs/site/assets/images/favicon.png",
        # sidebar_width=400,
    )

    _df_files_tgt, _df_files_psl = load_file_properties(
        config["OUTPUT_DIR"],
        ext="ecsv",
    )

    # join two dataframes for filtering
    df_files_tgt = _df_files_tgt.merge(
        _df_files_psl.loc[:, ["Upload ID", "Time_tot_L (h)", "Time_tot_M (h)"]],
        how="left",
        right_on="Upload ID",
        left_on="upload_id",
    )
    df_files_tgt.drop(columns=["Upload ID"], inplace=True)

    df_files_psl = _df_files_psl.merge(
        _df_files_tgt.loc[:, ["upload_id", "n_obj", "t_exp", "timestamp"]],
        how="left",
        right_on="upload_id",
        left_on="Upload ID",
    )
    df_files_psl.drop(columns=["upload_id"], inplace=True)
    df_files_psl.sort_values(
        "timestamp", ascending=False, ignore_index=True, inplace=True
    )

    # range sliders for filtering
    slider_nobj = pn.widgets.EditableRangeSlider(
        name="N (ob_code)",
        start=np.floor(df_files_tgt["n_obj"].min() / 10) * 10,
        end=np.ceil(df_files_tgt["n_obj"].max() / 10) * 10,
        step=10,
    )
    slider_fiberhour = pn.widgets.EditableRangeSlider(
        name="Fiberhour (h)",
        start=np.floor(df_files_tgt["t_exp"].min()),
        end=np.ceil(df_files_tgt["t_exp"].max()),
        step=1,
    )

    slider_rot_l = pn.widgets.EditableRangeSlider(
        name="ROT (low, h)",
        start=np.floor(df_files_psl["Time_tot_L (h)"].min()),
        end=np.ceil(df_files_psl["Time_tot_L (h)"].max()),
        step=1,
    )
    slider_rot_m = pn.widgets.EditableRangeSlider(
        name="ROT (medium, h)",
        start=np.floor(df_files_psl["Time_tot_M (h)"].min()),
        end=np.ceil(df_files_psl["Time_tot_M (h)"].max()),
        step=1,
    )

    # setup panel components

    # Target summary table
    table_files_tgt = pn.widgets.Tabulator(
        df_files_tgt,
        page_size=500,
        theme="bootstrap",
        # theme_classes=["table-striped", "table-sm"],
        theme_classes=["table-striped"],
        frozen_columns=["index"],
        pagination="remote",
        header_filters=True,
        titles={
            "upload_id": "Upload ID",
            "filenames": "File",
            "n_obj": "N (ob_code)",
            "t_exp": "Fiberhour (h)",
            "origname": "Original filename",
            "filesize": "Size (kB)",
            "timestamp": "Timestamp",
        },
        hidden_columns=[
            "index",
            "fullpath",
            "link",
            "Time_tot_L (h)",
            "Time_tot_M (h)",
        ],
        buttons={"download": "<i class='fa-solid fa-download'></i>"},
        layout="fit_data_table",
        disabled=True,
    )
    table_files_tgt.add_filter(slider_nobj, "n_obj")
    table_files_tgt.add_filter(slider_fiberhour, "t_exp")
    table_files_tgt.add_filter(slider_rot_l, "Time_tot_L (h)")
    table_files_tgt.add_filter(slider_rot_m, "Time_tot_M (h)")

    # PPP summary table
    table_files_psl = pn.widgets.Tabulator(
        df_files_psl,
        page_size=500,
        theme="bootstrap",
        theme_classes=["table-striped"],
        pagination="remote",
        header_filters=True,
        layout="fit_data_table",
        disabled=True,
        buttons={
            "magnify": "<i class='fa-solid fa-magnifying-glass'></i>",
            "download": "<i class='fa-solid fa-download'></i>",
        },
        hidden_columns=["index", "n_obj", "t_exp", "timestamp", "fullpath"],
        width=1400,
    )
    table_files_psl.add_filter(slider_nobj, "n_obj")
    table_files_psl.add_filter(slider_fiberhour, "t_exp")
    table_files_psl.add_filter(slider_rot_l, "Time_tot_L (h)")
    table_files_psl.add_filter(slider_rot_m, "Time_tot_M (h)")

    table_files_ppc = pn.widgets.Tabulator(
        page_size=20,
        theme="bootstrap",
        theme_classes=["table-striped"],
        pagination="remote",
        header_filters=True,
        layout="fit_data_table",
        disabled=True,
        hidden_columns=["index", "Fiber usage fraction (%)", "link"],
        width=550,
        visible=False,
    )

    # Open a file by clicking the download buttons
    # https://discourse.holoviz.org/t/how-to-make-a-dynamic-link-in-panel/2137
    js_panel = pn.pane.HTML(width=0, height=0, margin=0, sizing_mode="fixed")

    def execute_javascript(script):
        script = f'<script type="text/javascript">{script}</script>'
        js_panel.object = script
        js_panel.object = ""

    def open_panel_download(event):
        if event.column == "download":
            p_href = df_files_tgt["fullpath"][event.row].replace(
                config["OUTPUT_DIR"], "data", 1
            )
            # c.f. https://www.w3schools.com/jsref/met_win_open.asp
            script = f"window.open('{p_href}', '_blank')"
            execute_javascript(script)

    def open_panel_magnify(event):
        if event.column == "magnify":
            p_ppc = os.path.split(df_files_psl["fullpath"][event.row])[0]
            table_ppc_t = Table.read(
                os.path.join(p_ppc, f"ppc_{df_files_psl['Upload ID'][event.row]}.ecsv")
            )
            table_files_ppc.value = Table.to_pandas(table_ppc_t).sort_values(
                "ppc_priority", ascending=True, ignore_index=True
            )
            table_files_ppc.visible = True

        if event.column == "download":
            p_href = df_files_psl["fullpath"][event.row].replace(
                config["OUTPUT_DIR"], "data", 1
            )
            script = f"window.open('{p_href}', '_blank')"
            execute_javascript(script)

    table_files_tgt.on_click(open_panel_download)
    table_files_psl.on_click(open_panel_magnify)

    sidebar_column = pn.Column(
        slider_nobj, slider_fiberhour, slider_rot_l, slider_rot_m
    )

    tab_panels = pn.Tabs(
        ("Target info", pn.Column(table_files_tgt, js_panel)),
        ("Program info", pn.Row(table_files_psl, table_files_ppc)),
    )

    # put them into the template
    template.sidebar.append(sidebar_column)
    template.main.append(tab_panels)

    app = template.servable()

    return app
