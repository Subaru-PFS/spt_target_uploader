#!/usr/bin/env python3

import os
import time
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import panel as pn
from astropy.table import Table
from dotenv import dotenv_values
from logzero import logger

from .utils import (
    PPPrunStart,
    load_file_properties,
    load_input,
    ppp_result,
    upload_file,
    validate_input,
    visibility_checker,
)
from .widgets import (
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
        df_input, dict_load = load_input(
            BytesIO(panel_input.file_input.value), format="csv"
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


def target_uploader_app():
    config = dotenv_values(".env.shared")

    logger.info(f"config params from dotenv: {config}")

    if os.path.exists(config["OUTPUT_DIR_data"]):
        logger.info(f"{config['OUTPUT_DIR_data']} already exists.")
    else:
        os.makedirs(config["OUTPUT_DIR_data"])
        logger.info(f"{config['OUTPUT_DIR_data']} created.")

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
    panel_ppp_button = RunPppButtonWidgets()
    panel_status = StatusWidgets()
    panel_results = ResultWidgets()
    panel_targets = TargetWidgets()
    panel_ppp = PPPresultWidgets()
    panel_submit_button = SubmitButtonWidgets()

    placeholder_floatpanel = pn.Column(height=0, width=0)

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        panel_validate_button.pane,
        panel_status.pane,
        panel_ppp_button.pane,
        panel_submit_button.pane,
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
    template.sidebar.append(sidebar_column)
    template.main.append(main_column)

    tab_panels.visible = False

    # define on_click callback for the "validate" button
    def cb_validate(event):
        # try:
        #     del placeholder_floatpanel.objects[-1]
        # except:
        #     pass
        placeholder_floatpanel.objects = []
        # panel_validate_button.submit.disabled = True
        tab_panels.active = 0
        tab_panels.visible = False
        panel_status.reset()
        panel_results.reset()
        time.sleep(0.1)  # may be removed
        pn.state.notifications.clear()

        df_input, validation_status = _validate_file(panel_input)

        if validation_status is None:
            return

        panel_status.show_results(df_input, validation_status)
        panel_results.show_results(df_input, validation_status)
        panel_targets.show_results(df_input)

        tab_panels.visible = True

    # define on_click callback for the "PPP start" button
    def cb_PPP(event):
        placeholder_floatpanel.objects = []
        tab_panels.active = 0
        tab_panels.visible = False
        # panel_status.reset()
        panel_results.reset()
        panel_ppp.reset()
        time.sleep(0.1)  # may be removed
        pn.state.notifications.clear()

        gif_pane = pn.pane.GIF(
            "https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif",
            width=20,
        )
        panel_ppp_button.PPPrunStats.append(gif_pane)

        df_input_, validation_status = _validate_file(panel_input)

        if validation_status is None:
            time.sleep(0.1)
            panel_ppp_button.PPPrunStats.remove(gif_pane)
            return

        tb_input = Table.from_pandas(df_input_)

        tgt_obs_ok = visibility_checker(tb_input, "B")

        # NOTE: It seems boolean comparison for a numpy array must not be done with "is"
        # https://beta.ruff.rs/docs/rules/true-false-comparison/
        tgt_obs_no = np.where(~tgt_obs_ok)[0]
        tgt_obs_yes = np.where(tgt_obs_ok)[0]

        tb_input_ = tb_input[tgt_obs_yes]

        weight_para = [4.02, 0.01, 0.01]
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

        if p_result_tab_.value.iloc[-1]["Request time 1 (h)"] > 10 * 5:
            ppp_Alert = pn.pane.Alert(
                """### Warnings
The total requested time exceeds the 5-night upper limit of normal program. Please reduce the time.
            """,
                alert_type="danger",
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
                    alert_type="danger",
                )

            else:
                ppp_Alert = pn.pane.Alert(
                    """### Success
The total requested time is reasonable for normal program. All the input targets are observable in the semester.
                """,
                    alert_type="success",
                )

        panel_ppp_button.PPPrunStats.remove(gif_pane)

        panel_status.show_results(df_input_, validation_status)
        panel_results.show_results(df_input_, validation_status)
        panel_targets.show_results(df_input_)

        panel_ppp.show_results(res_mode_, nppc, p_result_fig, p_result_tab_, ppp_Alert)

        tab_panels.visible = True
        tab_panels.active = 2

        panel_submit_button.submit.disabled = False

        def cb_submit(event):
            # try:
            #     del placeholder_floatpanel.objects[-1]
            # except:
            #     pass
            placeholder_floatpanel.objects = []
            # placeholder_floatpanel = pn.Column(height=0, width=0)
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
                time.sleep(0.1)  # may be removed
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
                outdir=config["OUTPUT_DIR_data"],
                origname=panel_input.file_input.filename,
                secret_token=secret_token,
                upload_time=upload_time,
            )
            _, _, _ = upload_file(
                p_result_tab_.value,
                outdir=config["OUTPUT_DIR_ppp"],
                origname=panel_input.file_input.filename,
                secret_token=secret_token,
                upload_time=upload_time,
            )
            _, _, _ = upload_file(
                p_result_ppc.value,
                outdir=config["OUTPUT_DIR_ppc"],
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

    if not os.path.exists(config["OUTPUT_DIR_data"]):
        logger.error(f"{config['OUTPUT_DIR_data']} does not exist.")
        raise ValueError

    template = pn.template.VanillaTemplate(
        title="PFS Target & Proposal Lists",
        collapsed_sidebar=True,
        # header_background="#3A7D7E",
        # header_background="#C71585",  # mediumvioletred
        header_background="#dc143c",  # crimson
        busy_indicator=None,
        favicon="docs/site/assets/images/favicon.png",
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

    df_files_tgt = load_file_properties(config["OUTPUT_DIR_data"], ext="ecsv")
    df_files_psl = load_file_properties(config["OUTPUT_DIR_ppp"], ext="ecsv")

    editors = {}
    for c in df_files_tgt.columns:
        editors[c] = None

    # setup panel components
    table_files_tgt = pn.widgets.Tabulator(
        df_files_tgt,
        page_size=500,
        theme="bootstrap",
        # theme_classes=["table-striped", "table-sm"],
        theme_classes=["table-striped"],
        frozen_columns=["index"],
        pagination="remote",
        header_filters=True,
        editors=editors,
        titles={
            "upload_id": "Upload ID",
            "filenames": "File",
            "n_obj": "N(object)",
            "t_exp": "Fiberhour (h)",
            "origname": "Original filename",
            "filesize": "Size (kB)",
            "timestamp": "Timestamp",
        },
        hidden_columns=["index", "fullpath", "link"],
        buttons={"download": "<i class='fa-solid fa-download'></i>"},
        layout="fit_data_table",
    )

    table_files_psl = pn.widgets.Tabulator(
        df_files_psl,
        page_size=500,
        theme="bootstrap",
        theme_classes=["table-striped"],
        pagination="remote",
        header_filters=True,
        editors=editors,
        layout="fit_data_table",
        disabled=True,
        buttons={
            "magnify": "<i class='fa-solid fa-magnifying-glass'></i>",
            "download": "<i class='fa-solid fa-download'></i>",
        },
        hidden_columns=["index"],
        width=1400,
    )

    table_files_ppc = pn.widgets.Tabulator(
        page_size=20,
        theme="bootstrap",
        theme_classes=["table-striped"],
        pagination="remote",
        header_filters=True,
        editors=editors,
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
            href = f"/data/target_lists/{df_files_tgt['filename'][event.row]}"
            # c.f. https://www.w3schools.com/jsref/met_win_open.asp
            script = f"window.open('{href}', '_blank')"
            # print(href)
            execute_javascript(script)

    def open_panel_magnify(event):
        if event.column == "magnify":
            table_ppc_t = Table.read(
                config["OUTPUT_DIR_ppc"]
                + "targets_"
                + df_files_psl["Upload ID"][event.row]
                + ".ecsv"
            )
            table_files_ppc.value = Table.to_pandas(table_ppc_t).sort_values(
                "ppc_priority", ascending=True, ignore_index=True
            )
            table_files_ppc.visible = True

        if event.column == "download":
            href = (
                f"/data/ppc_lists/targets_{df_files_psl['Upload ID'][event.row]}.ecsv"
            )
            # c.f. https://www.w3schools.com/jsref/met_win_open.asp
            script = f"window.open('{href}', '_blank')"
            # print(href)
            execute_javascript(script)

    table_files_tgt.on_click(open_panel_download)
    table_files_psl.on_click(open_panel_magnify)

    tab_panels = pn.Tabs(
        ("Target info", pn.Column(table_files_tgt, js_panel)),
        ("Program info", pn.Row(table_files_psl, table_files_ppc)),
    )

    # put them into the template
    # template.sidebar.append(sidebar_column)
    template.main.append(tab_panels)

    app = template.servable()

    return app
