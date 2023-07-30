#!/usr/bin/env python3

import time
from io import BytesIO

import pandas as pd
import panel as pn
from logzero import logger

from .utils import load_input, validate_input
from .widgets import ButtonWidgets, FileInputWidgets, ResultWidgets, StatusWidgets


def target_uploader_app():
    template = pn.template.VanillaTemplate(
        title="PFS Target Uploader",
        sidebar_width=400,
        header_background="#3A7D7E",
        busy_indicator=None,
        favicon="docs/site/assets/images/favicon.png",
        # logo="docs/site/assets/images/favicon.png",
        # logo="src/pfs_etc_web/assets/logo-pfs.png",
    )

    # setup panel components
    panel_input = FileInputWidgets()
    panel_buttons = ButtonWidgets()
    panel_status = StatusWidgets()
    panel_results = ResultWidgets()

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        panel_buttons.pane,
        panel_status.pane,
    )

    # bundle panel(s) in the main area
    main_column = pn.Column(panel_results.pane)

    panel_results.pane.visible = False

    # put them into the template
    template.sidebar.append(sidebar_column)
    template.main.append(main_column)

    # define on_click callback for the "validate" button
    def cb_validate(event):
        panel_status.reset()
        panel_results.reset()
        time.sleep(0.25)
        pn.state.notifications.clear()
        if panel_input.file_input.filename is not None:
            logger.info(f"{panel_input.file_input.filename} is selected.")
            df_input, dict_load = load_input(
                BytesIO(panel_input.file_input.value), format="csv"
            )
            # panel_input.file_input.filename = None
            # if the input file cannot be read, raise a sticky error notifications
            if not dict_load["status"]:
                pn.state.notifications.error(
                    f"Cannot load the input file. Please check the content. Error: {dict_load['error']}",
                    duration=0,
                )
                return None
        else:
            logger.info("No file selected.")
            pn.state.notifications.error("Please select a CSV file.")
            return None

        df, validation_status = validate_input(df_input)
        panel_status.show_results(df, validation_status)
        panel_results.show_results(df, validation_status)
        panel_results.pane.visible = True

    # def on_click_submit(event):
    #     pass

    # set callback to the "validate" click
    panel_buttons.validate.on_click(cb_validate)
    # panel_buttons.submit.on_click(on_click_submit)

    return template.servable()
