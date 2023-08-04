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

from .utils import load_file_properties, load_input, upload_file, validate_input
from .widgets import (
    ButtonWidgets,
    DocLinkWidgets,
    FileInputWidgets,
    ResultWidgets,
    StatusWidgets,
    TargetWidgets,
    UploadNoteWidgets,
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

    if os.path.exists(config["OUTPUT_DIR"]):
        logger.info(f"{config['OUTPUT_DIR']} already exists.")
    else:
        os.makedirs(config["OUTPUT_DIR"])
        logger.info(f"{config['OUTPUT_DIR']} created.")

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
    panel_buttons = ButtonWidgets()
    panel_status = StatusWidgets()
    panel_results = ResultWidgets()
    panel_targets = TargetWidgets()

    placeholder_floatpanel = pn.Column(height=0, width=0)

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        panel_buttons.pane,
        panel_status.pane,
    )

    # bundle panel(s) in the main area
    tab_panels = pn.Tabs(
        ("Results", panel_results.pane),
        ("Inputs", panel_targets.pane),
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
        panel_buttons.submit.disabled = True
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
            panel_buttons.submit.disabled = False

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
                panel_buttons.submit.disabled = True
                tab_panels.visible = True
                return

        outfile, uploaded_time, secret_token = upload_file(
            df_input,
            outdir=config["OUTPUT_DIR"],
            origname=panel_input.file_input.filename,
        )
        panel_notes = UploadNoteWidgets(
            uploaded_time,
            secret_token,
        )
        #         panel_notes = UploadNoteWidgets(
        #             f"""<i class='fa-regular fa-thumbs-up fa-2xl'></i><font size='4'>  The target list has been uploaded successfully!</font>

        # <font size='4'>Upload ID:  </font><font size='6'><span style='color: darkcyan;'>**{secret_token}**</span></font>

        # <font size='4'>Uploaded at {uploaded_time.isoformat(timespec='seconds')}</font>

        # Please keep the Upload ID for the observation planning.
        #             """,
        #             secret_token,
        #         )
        placeholder_floatpanel[:] = [panel_notes.floatpanel]

    # set callback to the "validate" click
    panel_buttons.validate.on_click(cb_validate)
    panel_buttons.submit.on_click(cb_submit)

    app = template.servable()

    return app


def list_files_app():
    config = dotenv_values(".env.shared")

    logger.info(f"config params from dotenv: {config}")

    if not os.path.exists(config["OUTPUT_DIR"]):
        logger.error(f"{config['OUTPUT_DIR']} does not exist.")
        raise ValueError

    template = pn.template.VanillaTemplate(
        title="PFS Target Lists",
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

    df_files = load_file_properties(config["OUTPUT_DIR"], ext="ecsv")

    editors = {}
    for c in df_files.columns:
        editors[c] = None

    # setup panel components
    table_files = pn.widgets.Tabulator(
        df_files,
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
            href = f"/data/{df_files['filename'][event.row]}"
            # c.f. https://www.w3schools.com/jsref/met_win_open.asp
            script = f"window.open('{href}', '_blank')"
            # print(href)
            execute_javascript(script)

    table_files.on_click(open_panel_download)

    main_column = pn.Column(
        # pn.pane.Markdown(
        #     "<font size='5' style='text-color: red;'>`PFS`</font>",
        #     renderer="markdown-it",
        # ),
        table_files,
        js_panel,
    )

    # put them into the template
    # template.sidebar.append(sidebar_column)
    template.main.append(main_column)

    app = template.servable()

    return app
