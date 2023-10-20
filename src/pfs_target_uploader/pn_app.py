#!/usr/bin/env python3

import os
import time
from datetime import datetime, timezone
from io import BytesIO

import gurobipy
import numpy as np
import panel as pn
from astropy.table import Table
from dotenv import dotenv_values
from logzero import logger

from .utils.checker import validate_input, visibility_checker
from .utils.io import load_file_properties, load_input, upload_file
from .utils.ppp import PPPrunStart, ppp_result
from .widgets import (
    DatePickerWidgets,
    DocLinkWidgets,
    FileInputWidgets,
    PPPresultWidgets,
    ResultWidgets,
    RunPppButtonWidgets,
    StatusWidgets,
    SubmitButtonWidgets,
    TargetWidgets,
    UploadNoteWidgets,
    ValidateButtonWidgets,
)


def _validate_file(panel_input):
    if panel_input.file_input.filename is not None:
        logger.info(f"{panel_input.file_input.filename} is selected.")
        file_format = os.path.splitext(panel_input.file_input.filename)[-1].replace(
            ".", ""
        )
        df_input, dict_load = load_input(
            BytesIO(panel_input.file_input.value),
            format=file_format,
        )
        # if the input file cannot be read, raise a sticky error notifications
        if not dict_load["status"]:
            pn.state.notifications.error(
                f"Cannot load the input file. Please check the content. Error: {dict_load['error']}",
                duration=0,
            )
            return None, None
    else:
        logger.info("No file selected.")
        pn.state.notifications.error("Please select a CSV file.")
        return None, None

    validation_status = validate_input(df_input)

    return df_input, validation_status


def _toggle_buttons(buttons: list, disabled: bool = True):
    for b in buttons:
        b.disabled = disabled


def target_uploader_app():
    config = dotenv_values(".env.shared")

    logger.info(f"config params from dotenv: {config}")

    if os.path.exists(
        os.path.join(config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_data"])
    ):
        logger.info(
            f"{os.path.join(config['OUTPUT_DIR_PREFIX'], config['OUTPUT_DIR_data'])} already exists."
        )
    else:
        os.makedirs(
            os.path.join(config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_data"])
        )
        logger.info(
            f"{os.path.join(config['OUTPUT_DIR_PREFIX'], config['OUTPUT_DIR_data'])} created."
        )

    template = pn.template.VanillaTemplate(
        title="PFS Target Uploader",
        sidebar_width=400,
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

    panel_results = ResultWidgets()
    panel_targets = TargetWidgets()
    panel_ppp = PPPresultWidgets()

    panel_input.reset()

    button_set = [
        panel_input.file_input,
        panel_validate_button.validate,
        panel_ppp_button.PPPrun,
    ]

    placeholder_floatpanel = pn.Column(height=0, width=0)

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        panel_validate_button.pane,
        panel_status.pane,
        panel_ppp_button.pane,
        panel_submit_button.pane,
    )

    sidebar_configs = pn.Column(panel_dates.pane)

    tab_sidebar = pn.Tabs(
        ("Home", sidebar_column),
        ("Config", sidebar_configs),
    )

    # bundle panel(s) in the main area
    tab_panels = pn.Tabs(
        ("Input list", panel_targets.pane),
        ("Results of validation", panel_results.pane),
        ("Results of PPP", panel_ppp.pane),
    )

    main_column = pn.Column(
        placeholder_floatpanel,
        panel_doc.pane,
        tab_panels,
    )

    # put them into the template
    template.sidebar.append(tab_sidebar)
    template.main.append(main_column)

    tab_panels.visible = False

    # define on_click callback for the "validate" button
    def cb_validate(event):
        # disable the buttons and input file widget while validation
        _toggle_buttons(button_set, disabled=True)

        placeholder_floatpanel.objects = []
        # tab_panels.active = 0
        tab_panels.visible = False

        panel_status.reset()
        panel_results.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        df_input, validation_status = _validate_file(panel_input)

        if validation_status is None:
            _toggle_buttons(button_set, disabled=False)
            return

        panel_status.show_results(df_input, validation_status)
        panel_targets.show_results(df_input)
        panel_results.show_results(df_input, validation_status)

        _toggle_buttons(button_set, disabled=False)

        tab_panels.active = 1
        tab_panels.visible = True

    # define on_click callback for the "PPP start" button
    def cb_PPP(event):
        _toggle_buttons(button_set, disabled=True)
        panel_submit_button.submit.disabled = True

        placeholder_floatpanel.objects = []
        tab_panels.active = 1
        # tab_panels.visible = False

        panel_status.reset()
        panel_results.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        gif_pane = pn.pane.GIF(
            "https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif",
            width=20,
        )

        df_input_, validation_status = _validate_file(panel_input)

        if validation_status is None:
            _toggle_buttons(button_set, disabled=False)
            return

        panel_status.show_results(df_input_, validation_status)
        panel_results.show_results(df_input_, validation_status)
        panel_targets.show_results(df_input_)
        tab_panels.visible = True

        panel_ppp_button.PPPrunStats.append(gif_pane)

        tb_input = Table.from_pandas(df_input_)

        # tgt_obs_ok = visibility_checker(tb_input, "B")
        logger.info(f"Observation period start at {panel_dates.date_begin.value}")
        logger.info(f"Observation period end at {panel_dates.date_end.value}")

        tgt_obs_ok = visibility_checker(
            tb_input,
            panel_dates.date_begin.value,
            panel_dates.date_end.value,
        )

        # NOTE: It seems boolean comparison for a numpy array must not be done with "is"
        # https://beta.ruff.rs/docs/rules/true-false-comparison/
        tgt_obs_no = np.where(~tgt_obs_ok)[0]
        tgt_obs_yes = np.where(tgt_obs_ok)[0]

        tb_input_ = tb_input[tgt_obs_yes]

        weight_para = [4.02, 0.01, 0.01]

        try:
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
            ) = PPPrunStart(tb_input_, weight_para)

            res_mode_, nppc, p_result_fig, p_result_ppc, p_result_tab_ = ppp_result(
                cR_L_, sub_l, obj_allo_L_fin, uS_L2, cR_M_, sub_m, obj_allo_M_fin, uS_M2
            )

            if p_result_tab_.value.iloc[-1]["Request time (h)"] > 10 * 5:
                ppp_Alert = pn.pane.Alert(
                    """### Warnings
The total requested time exceeds the 5-night upper limit of normal program. Please reduce the time.
            """,
                    alert_type="warning",
                )
                if len(tgt_obs_no) > 0:
                    tgt_obs_no_id = " ".join(tb_input[tgt_obs_no]["ob_code"])
                    ppp_Alert.object += f"""

The following targets are not observable during the semester. Please remove them.
    {tgt_obs_no_id}
                """

            else:
                if len(tgt_obs_no) > 0:
                    tgt_obs_no_id = " ".join(tb_input[tgt_obs_no]["ob_code"])
                    ppp_Alert = pn.pane.Alert(
                        f"""### Warnings
The following targets are not observable during the semester. Please remove them.
    {tgt_obs_no_id}
                """,
                        alert_type="warning",
                    )

                else:
                    ppp_Alert = pn.pane.Alert(
                        """### Success
The total requested time is reasonable for normal program. All the input targets are observable in the semester.
                """,
                        alert_type="success",
                    )

            panel_ppp.show_results(
                res_mode_, nppc, p_result_fig, p_result_tab_, ppp_Alert
            )

            # tab_panels.visible = True
            tab_panels.active = 2

            panel_submit_button.submit.disabled = False

        except gurobipy.GurobiError as e:
            pn.state.notifications.error(f"{str(e)}", duration=0)
            pass

        panel_ppp_button.PPPrunStats.remove(gif_pane)

        _toggle_buttons(button_set, disabled=False)

        def cb_submit(event):
            panel_submit_button.submit.disabled = True

            placeholder_floatpanel.objects = []

            logger.info("Submit button clicked.")
            logger.info("Validation before actually writing to the storage")

            # do the validation again (input file can be different)
            # and I don't know how to implement to return value
            # from callback to another function (sorry)
            df_input, validation_status = _validate_file(panel_input)

            if (validation_status is None) or (not validation_status["status"]):
                logger.error("Validation failed for some reason")

                tab_panels.visible = False

                panel_status.reset()
                panel_results.reset()

                pn.state.notifications.clear()

                if validation_status is None:
                    return
                else:
                    logger.error("Validation failed for some reason")
                    panel_status.show_results(df_input, validation_status)
                    panel_results.show_results(df_input, validation_status)
                    panel_targets.show_results(df_input)
                    tab_panels.visible = True
                    return

            upload_time = datetime.now(timezone.utc)
            secret_token = panel_input.secret_token

            _, _, _ = upload_file(
                df_input,
                outdir=os.path.join(
                    config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_data"]
                ),
                origname=panel_input.file_input.filename,
                secret_token=secret_token,
                upload_time=upload_time,
            )
            _, _, _ = upload_file(
                p_result_tab_.value,
                outdir=os.path.join(
                    config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_ppp"]
                ),
                origname=panel_input.file_input.filename,
                secret_token=secret_token,
                upload_time=upload_time,
            )
            _, _, _ = upload_file(
                p_result_ppc.value,
                outdir=os.path.join(
                    config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_ppc"]
                ),
                origname=panel_input.file_input.filename,
                secret_token=secret_token,
                upload_time=upload_time,
            )
            panel_notes = UploadNoteWidgets(secret_token, upload_time)
            placeholder_floatpanel[:] = [panel_notes.floatpanel]

            panel_submit_button.submit.disabled = True

        panel_submit_button.submit.on_click(cb_submit)

    # set callback to the "validate" click
    panel_validate_button.validate.on_click(cb_validate)
    panel_ppp_button.PPPrun.on_click(cb_PPP)

    app = template.servable()

    return app


def list_files_app():
    config = dotenv_values(".env.shared")

    logger.info(f"config params from dotenv: {config}")

    if not os.path.exists(
        os.path.join(config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_data"])
    ):
        logger.error(
            f"{os.path.join(config['OUTPUT_DIR_PREFIX'], config['OUTPUT_DIR_data'])} does not exist."
        )
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
    # template = pn.template.BootstrapTemplate(
    #     title="PFS Target Lists",
    #     collapsed_sidebar=True,
    #     # header_background="#3A7D7E",
    #     # header_background="#C71585",  # mediumvioletred
    #     header_background="#dc143c",  # crimson
    #     busy_indicator=None,
    #     favicon="docs/site/assets/images/favicon.png",
    # )

    _df_files_tgt = load_file_properties(
        os.path.join(config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_data"]), ext="ecsv"
    )

    _df_files_psl = load_file_properties(
        os.path.join(config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_ppp"]), ext="ecsv"
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
        name="N(ob_code)",
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
            "n_obj": "N(ob_code)",
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
        hidden_columns=["index", "n_obj", "t_exp", "timestamp"],
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
        # print("js executed")
        script = f'<script type="text/javascript">{script}</script>'
        js_panel.object = script
        js_panel.object = ""

    def open_panel_download(event):
        if event.column == "download":
            # href = f"data/target_lists/{df_files_tgt['filename'][event.row]}"
            href = f"data/{config['OUTPUT_DIR_data']}/{df_files_tgt['filename'][event.row]}"
            # c.f. https://www.w3schools.com/jsref/met_win_open.asp
            script = f"window.open('{href}', '_blank')"
            # print(href)
            execute_javascript(script)

    def open_panel_magnify(event):
        if event.column == "magnify":
            table_ppc_t = Table.read(
                os.path.join(
                    config["OUTPUT_DIR_PREFIX"],
                    config["OUTPUT_DIR_ppc"],
                    f"targets_{df_files_psl['Upload ID'][event.row]}.ecsv",
                )
            )
            table_files_ppc.value = Table.to_pandas(table_ppc_t).sort_values(
                "ppc_priority", ascending=True, ignore_index=True
            )
            table_files_ppc.visible = True

        if event.column == "download":
            href = f"data/{config['OUTPUT_DIR_ppc']}/targets_{df_files_psl['Upload ID'][event.row]}.ecsv"
            # c.f. https://www.w3schools.com/jsref/met_win_open.asp
            script = f"window.open('{href}', '_blank')"
            # print(href)
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
