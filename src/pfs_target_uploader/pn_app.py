#!/usr/bin/env python3

import glob
import os
import secrets
import time
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import panel as pn
from astropy.table import Table
from bokeh.models.widgets.tables import HTMLTemplateFormatter
from dotenv import dotenv_values
from logzero import logger

from .utils import load_file_properties, load_input, upload_file, validate_input, PPPrunStart, ppp_result, visibility_checker
from .widgets import (
    ButtonWidgets1,
    ButtonWidgets2,
    DocLinkWidgets,
    FileInputWidgets,
    ResultWidgets,
    StatusWidgets,
    TargetWidgets,
    UploadNoteWidgets,
    PPPresultWidgets,
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
    secret_token_t = secrets.token_hex(8)

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
    panel_buttons1 = ButtonWidgets1()
    panel_buttons2 = ButtonWidgets2()
    panel_status = StatusWidgets()
    panel_results = ResultWidgets()
    panel_targets = TargetWidgets()
    panel_ppp = PPPresultWidgets()

    placeholder_floatpanel = pn.Column(height=0, width=0)

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        panel_buttons1.pane,
        panel_status.pane,
        panel_buttons2.pane,
    )

    # bundle panel(s) in the main area
    tab_panels = pn.Tabs(
        ("Input list", panel_targets.pane),
        ("Results of validating", panel_results.pane),
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
        panel_buttons1.submit.disabled = True
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

        # activate submit button when no error is detected
        if validation_status["status"]:
            panel_buttons1.submit.disabled = False

        tab_panels.visible = True

    def cb_submit(event):
        # try:
        #     del placeholder_floatpanel.objects[-1]
        # except:
        #     pass
        placeholder_floatpanel.objects = []
        # placeholder_floatpanel = pn.Column(height=0, width=0)
        logger.info("Submit button clicked.")
        logger.info("Validation before actually writing to the storage")

        # do the validatin again (input file can be different)
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
                panel_buttons1.submit.disabled = True
                tab_panels.visible = True
                return

        outfile, uploaded_time, secret_token = upload_file(
            df_input,
            outdir=config["OUTPUT_DIR_data"],
            origname=panel_input.file_input.filename,
            secret_token=secret_token_t,
        )
        panel_notes = UploadNoteWidgets(
            f"""<i class='fa-regular fa-thumbs-up fa-2xl'></i><font size='4'>  The target list has been uploaded successfully!</font>

<font size='4'>Upload ID:  </font><font size='6'><span style='color: darkcyan;'>**{secret_token}**</span></font>

<font size='4'>Uploaded at {uploaded_time.isoformat(timespec='seconds')}</font>

Please keep the Upload ID for the observation planning.
            """
        )
        placeholder_floatpanel[:] = [panel_notes.floatpanel]

    # define on_click callback for the "PPP start" button
    def cb_PPP(event):
        placeholder_floatpanel.objects = []
        panel_ppp.reset()
        time.sleep(0.1)  # may be removed
        pn.state.notifications.clear()

        gif_pane = pn.pane.GIF('https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif', width=20)
        panel_buttons2.PPPrunStats.append(gif_pane)

        df_input_ = _validate_file(panel_input)[0]
        df_input = Table.from_pandas(df_input_)

        tgt_obs_ok = visibility_checker(df_input, 'B')
        tgt_obs_no = np.where(tgt_obs_ok == False)[0]
        tgt_obs_yes = np.where(tgt_obs_ok == True)[0]
        
        df_input_ = df_input[tgt_obs_yes]

        weight_para = [4.02,0.01,0.01]
        uS_L2, cR_L, cR_L_, sub_l, obj_allo_L_fin, uS_M2, cR_M, cR_M_, sub_m, obj_allo_M_fin = PPPrunStart( df_input_, weight_para)
        res_mode_, nppc, p_result_fig, p_result_ppc, p_result_tab_ = ppp_result(cR_L_, sub_l, obj_allo_L_fin, uS_L2, cR_M_, sub_m, obj_allo_M_fin, uS_M2)

        if p_result_tab_.value.iloc[-1]['Request time 1 (h)'] > 10 * 5:
            ppp_Alert=pn.pane.Alert(
            """### Warnings
The total requested time exceeds the 5-night upper limit of normal program. Please reduce the time.
            """,
            alert_type="danger",
            )
            if len(tgt_obs_no) > 0:
                tgt_obs_no_id = ' '.join(df_input[tgt_obs_no]['ob_code'])
                ppp_Alert.object += f"""
                
The following targets are not observable during the semester. Please remove them.
    {tgt_obs_no_id}
                """

        else:
            if len(tgt_obs_no) > 0:
                tgt_obs_no_id = ' '.join(df_input[tgt_obs_no]['ob_code'])
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
      
        panel_buttons2.PPPrunStats.remove(gif_pane)   
        panel_ppp.show_results(res_mode_, nppc, p_result_fig, p_result_tab_, ppp_Alert)   
        
        panel_buttons2.PPPsubmit.disabled = False

        def cb_PPP_submit(event):
            placeholder_floatpanel.objects = []
            logger.info("PPP submit button clicked.")

            outfile, uploaded_time, secret_token = upload_file(
                p_result_tab_.value,
                outdir=config["OUTPUT_DIR_ppp"],
                origname=panel_input.file_input.filename,
                secret_token=secret_token_t,
            )
            outfile, uploaded_time, secret_token = upload_file(
                p_result_ppc.value,
                outdir=config["OUTPUT_DIR_ppc"],
                origname=panel_input.file_input.filename,
                secret_token=secret_token_t,
            )
            panel_notes = UploadNoteWidgets(
                f"""<i class='fa-regular fa-thumbs-up fa-2xl'></i><font size='4'>  The PPP output has been uploaded successfully!</font>

<font size='4'>Upload ID:  </font><font size='6'><span style='color: darkcyan;'>**{secret_token}**</span></font>

<font size='4'>Uploaded at {uploaded_time.isoformat(timespec='seconds')}</font>
                """
            )
            placeholder_floatpanel[:] = [panel_notes.floatpanel]

        panel_buttons2.PPPsubmit.on_click(cb_PPP_submit)

    # set callback to the "validate" click
    panel_buttons1.validate.on_click(cb_validate)
    panel_buttons1.submit.on_click(cb_submit)
    panel_buttons2.PPPrun.on_click(cb_PPP)

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
        buttons={"magnify":"<i class='fa-solid fa-magnifying-glass'></i>"},
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
            href = f"/data/{df_files_tgt['filename'][event.row]}"
            # c.f. https://www.w3schools.com/jsref/met_win_open.asp
            script = f"window.open('{href}', '_blank')"
            # print(href)
            execute_javascript(script)
    
    def open_panel_magnify(event):
        if event.column == "magnify":
            table_ppc_t = Table.read(config["OUTPUT_DIR_ppc"]+'targets_'+df_files_psl.iloc[event.row]['Upload ID']+'.ecsv')
            table_files_ppc.value = Table.to_pandas(table_ppc_t).sort_values("ppc_priority", ascending=True, ignore_index=True)
            table_files_ppc.visible = True

    table_files_tgt.on_click(open_panel_download)
    table_files_psl.on_click(open_panel_magnify)

    tab_panels = pn.Tabs(
        ("Target list", pn.Column(table_files_tgt, js_panel)),
        ("Proposal list", pn.Row(table_files_psl,table_files_ppc)),
    )

    # put them into the template
    # template.sidebar.append(sidebar_column)
    template.main.append(tab_panels)

    app = template.servable()

    return app
