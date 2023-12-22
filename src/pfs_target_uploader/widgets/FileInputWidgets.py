#!/usr/bin/env python3

import os
import secrets
import time
from io import BytesIO

import panel as pn
import param
from loguru import logger

from ..utils.checker import validate_input
from ..utils.io import load_input


class FileInputWidgets(param.Parameterized):
    def __init__(self):
        self.file_input = pn.widgets.FileInput(
            value=None,
            filename=None,
            accept=".csv,.ecsv",
            multiple=False,
            height=40,
        )

        # store previous information for input comparison
        self.previous_filename = None
        self.previous_value = None
        self.previous_mime_type = None

        # hex string to be used as an upload ID
        self.secret_token = None

        self.pane = pn.Column(
            pn.pane.Markdown(
                "<font size=5>**Select an input CSV file**</font> "
                "<font size=4>(<a href='doc/examples/example_targetlist_random10.csv' target='_blank'>example</a>)</font>",
                # styles={
                #     "border-left": "10px solid #3A7D7E",
                #     "border-bottom": "1px solid #3A7D7E",
                #     "padding-left": "0.5em",
                # },
            ),
            #             """# Step 1:
            # ## Select a target list (<a href='doc/examples/example_targetlist_random100.csv' target='_blank'>example</a>)""",
            self.file_input,
            margin=(10, 0, 0, 0),
        )

    def reset(self):
        self.file_input.filename = None
        self.file_input.mime_type = None
        self.file_input.value = None

    def validate(self, date_begin=None, date_end=None):
        t_start = time.time()
        if date_begin >= date_end:
            pn.state.notifications.error(
                "Date Begin must be before Date End.", duration=0
            )
            return None, None, None

        # update the upload ID when the input file is different from previous validation.
        if (
            (self.file_input.filename != self.previous_filename)
            or (self.file_input.value != self.previous_value)
            or (self.file_input.mime_type != self.previous_mime_type)
        ):
            st = secrets.token_hex(8)

            logger.info("New file detected.")
            logger.info(f"    Upload ID updated: {st}")
            logger.info(f"    Filename: {self.file_input.filename}")
            logger.info(f"    MIME Type: {self.file_input.mime_type}")

            self.secret_token = st

            self.previous_filename = self.file_input.filename
            self.previous_value = self.file_input.value
            self.previous_mime_type = self.file_input.mime_type
        else:
            logger.info("Identical to the previous validation.")
            logger.info(
                "    Upload ID not updated: one or more of the filename, content, "
                "and mime type are identical to the previous validation."
            )

        logger.info(f"Upload ID: {self.secret_token}")

        if self.file_input.filename is not None:
            logger.info(f"{self.file_input.filename} is selected.")
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
                return None, None, None
        else:
            logger.info("No file selected.")
            pn.state.notifications.error("Please select a CSV file.")
            return None, None, None

        validation_status, df_output = validate_input(
            df_input.copy(deep=True), date_begin=date_begin, date_end=date_end
        )
        t_stop = time.time()
        logger.info(f"Validation finished in {t_stop - t_start:.2f} [s]")

        return validation_status, df_input, df_output
