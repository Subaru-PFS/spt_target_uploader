#!/usr/bin/env python3

import os
from io import BytesIO

import pandas as pd
import panel as pn
from loguru import logger

from ..utils import ppc_datatype
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
                dtype=ppc_datatype,
                format=file_format,
            )
            # if the input file cannot be read, raise a sticky error notifications
            if not dict_load["status"]:
                logger.error(f"Cannot load the input file {self.file_input.filename}.")
                pn.state.notifications.error(
                    f"Cannot load the input file {self.file_input.filename}. Please check the content.\n"
                    f"\nError: {dict_load['error']}",
                    duration=0,
                )
                return None

            is_required_columns = True
            for col_t in ["ppc_ra", "ppc_dec", "ppc_resolution"]:
                if col_t not in df_input.columns:
                    is_required_columns = False
                    logger.error(f"Missing mandatory column: {col_t}")
                    pn.state.notifications.error(
                        f"Missing mandatory column: {col_t}", duration=0
                    )

            if not is_required_columns:
                return None

        else:
            logger.info("No pointing list selected.")
            return pd.DataFrame()  # return an empty DataFrame

        if "ppc_pa" not in df_input.columns:
            df_input["ppc_pa"] = 0.0

        if "ppc_priority" not in df_input.columns:
            df_input["ppc_priority"] = 0.0

        if "ppc_code" not in df_input.columns:
            df_input["ppc_code"] = ["Point_" + str(tt) for tt in range(len(df_input))]

        return df_input
