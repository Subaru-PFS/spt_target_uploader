#!/usr/bin/env python3

import os
import secrets
import time
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import panel as pn
from astropy.table import Table
from dotenv import dotenv_values
from logzero import logger

from .utils import load_input, validate_input
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


def _upload_file(df, origname=None, outdir="."):
    # convert pandas.DataFrame to astropy.Table
    tb = Table.from_pandas(df)

    # use the current UTC time and random hash string to construct an output filename
    uploaded_time = datetime.now(timezone.utc)
    secret_token = secrets.token_hex(8)

    # add metadata
    tb.meta["original_filename"] = origname
    tb.meta["upload_at"] = uploaded_time.isoformat(timespec="seconds")

    # filename = f"{uploaded_time.strftime('%Y%m%d-%H%M%S')}_{secret_token}.ecsv"
    filename = (
        f"targets_{uploaded_time.isoformat(timespec='seconds')}_{secret_token}.ecsv"
    )

    logger.info(f"File `{filename}` was saved under `{outdir}`")

    # save the table in the output directory as an ECSV file
    tb.write(os.path.join(outdir, filename), delimiter=",", format="ascii.ecsv")

    return filename, uploaded_time, secret_token


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

    # main_column.visible = False
    # panel_results.pane.visible = False
    # panel_targets.pane.visible = False

    tab_panels.visible = False

    # define on_click callback for the "validate" button
    def cb_validate(event):
        try:
            del placeholder_floatpanel.objects[-1]
        except:
            pass
        panel_buttons.submit.disabled = True
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
        try:
            del placeholder_floatpanel.objects[-1]
        except:
            pass
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

        outfile, uploaded_time, secret_token = _upload_file(
            df_input,
            outdir=config["OUTPUT_DIR"],
            origname=panel_input.file_input.filename,
        )
        panel_notes = UploadNoteWidgets(
            f"""<font size='4'>The target list has been uploaded successfully!</font>

<font size='4'>Upload ID: </font><font size='6'>**{secret_token}**</font>


Please keep the `Upload ID`.
            """
        )
        placeholder_floatpanel[:] = [panel_notes.floatpanel]

    # set callback to the "validate" click
    panel_buttons.validate.on_click(cb_validate)
    panel_buttons.submit.on_click(cb_submit)

    app = template.servable()

    # @pn.depends(tab_panels.param.active, watch=True)
    # def set_height(index):
    #     tab = tab_panels[index]
    #     tab_panels.height = tab.height

    return app
