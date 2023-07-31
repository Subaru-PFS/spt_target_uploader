#!/usr/bin/env python3

import time
from io import BytesIO

import pandas as pd
import panel as pn
from logzero import logger

from .utils import load_input, validate_input
from .widgets import (
    ButtonWidgets,
    DocLinkWidgets,
    FileInputWidgets,
    ResultWidgets,
    StatusWidgets,
    TargetWidgets,
)


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
    panel_doc = DocLinkWidgets()
    panel_input = FileInputWidgets()
    panel_buttons = ButtonWidgets()
    panel_status = StatusWidgets()
    panel_results = ResultWidgets()
    panel_targets = TargetWidgets()

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
    main_column = pn.Column(panel_doc.pane, tab_panels)

    # put them into the template
    template.sidebar.append(sidebar_column)
    template.main.append(main_column)

    # main_column.visible = False
    # panel_results.pane.visible = False
    # panel_targets.pane.visible = False

    tab_panels.visible = False

    # define on_click callback for the "validate" button
    def cb_validate(event):
        # panel_results.pane.visible = False
        panel_buttons.submit.disabled = True
        tab_panels.visible = False
        panel_status.reset()
        panel_results.reset()
        time.sleep(0.1)  # may be removed
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

        validation_status = validate_input(df_input)

        panel_status.show_results(df_input, validation_status)
        panel_results.show_results(df_input, validation_status)
        panel_targets.show_results(df_input)

        # activate submit button when no error is detected
        if validation_status["status"]:
            panel_buttons.submit.disabled = False

        # panel_results.pane.visible = True
        # panel_targets.pane.visible = True
        tab_panels.visible = True

    def cb_submit(event):
        # logger.info(f"""\n{df_input}\n{panel_input.file_input.filename}""")
        logger.info("Submit button clicked. (did nothing.)")
        pass

    # set callback to the "validate" click
    panel_buttons.validate.on_click(cb_validate)
    panel_buttons.submit.on_click(cb_submit)

    return template.servable()
