#!/usr/bin/env python3

import os
import time
from io import BytesIO

import panel as pn
from loguru import logger

from ..utils.io import load_input


class PPCInputWidgets:
    def __init__(self):
        self.file_input = pn.widgets.FileInput(
            value=None,
            filename=None,
            accept=".csv,.ecsv",
            multiple=False,
            height=30,
        )

        self.pane = pn.Column(
            pn.pane.Markdown(
                "<font size=3>**Select an input pointing list**</font> "
                "<font size=3>(<a href='doc/examples/example_ppclist.csv' target='_blank'>example</a>)</font>",
            ),
            self.file_input,
            margin=(-30, 0, 0, 0),
        )

    def validate(self):
        if self.file_input.filename is not None:
            file_format = os.path.splitext(self.file_input.filename)[-1].replace(
                ".", ""
            )
            df_input, dict_load = load_input(
                BytesIO(self.file_input.value),
                format=file_format,
            )
            # if the input file cannot be read, raise a sticky error notifications
            if not dict_load["status"]:
                pn.state.notifications.error(
                    f"Cannot load the input file. Please check the content. Error: {dict_load['error']}",
                    duration=0,
                )
                return None
            for col_t in ["ppc_ra", "ppc_dec", "ppc_pa", "ppc_resolution"]:
                if col_t not in df_input.columns:
                    pn.state.notifications.error(
                        f"Cannot load the input pointing list due to the missing columns: {col_t}",
                        duration=0,
                    )
                    return None
        else:
            logger.info("No pointing list selected.")
            pn.state.notifications.info(
                "No pointing list input, the automatic pointing simulation will run instead",
                duration=0,
            )
            return []

        return df_input
