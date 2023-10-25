#!/usr/bin/env python3

import os
import secrets
import sys
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import panel as pn
import param
from astropy import units as u
from astropy.table import Table
from logzero import logger
from zoneinfo import ZoneInfo

from .utils.checker import get_semester_daterange, validate_input
from .utils.io import load_input
from .utils.ppp import PPPrunStart, ppp_result


class UploadNoteWidgets:
    # TODO: perhaps I can refactor to make it simple...
    def __init__(self, secret_token, uploaded_time, ppp_status):
        if ppp_status:
            self.floatpanel = pn.layout.FloatPanel(
                None,
                # pn.pane.Markdown(message),
                name="Info",
                # config={"headerLogo": "<i class='fa-regular fa-thumbs-up fa-lg'></i>"},
                contained=False,
                position="center",
                # theme="none",
                theme="#3A7D7E",
                margin=20,
                width=720,
            )

            # JS on-click actions
            # https://github.com/awesome-panel/awesome-panel/blob/master/examples/js_actions.py
            # so far not working...
            stylesheet = """
        :host {
            --font-size: 2.5em;
            --color: darkcyan;
        }
        .bk-btn-light {
            color: darkcyan;
        }
        """

            self.copy_source_button = pn.widgets.Button(
                name=f"{secret_token}",
                icon="copy",
                # width=500,
                height=96,
                icon_size="1.5em",
                # button_style="outline",
                button_type="light",
                stylesheets=[stylesheet],
            )

            copy_source_code = "navigator.clipboard.writeText(source);"

            self.copy_source_button.js_on_click(
                args={"source": f"{secret_token}"},
                code=copy_source_code,
            )

            messages = [
                pn.pane.Markdown(
                    "<i class='fa-regular fa-thumbs-up fa-2xl'></i><font size='4'>  Upload successful! Your **Upload ID** is the following.</font>"
                ),
                self.copy_source_button,
                pn.pane.Markdown(
                    f"<font size='4'>Uploaded at {uploaded_time.isoformat(timespec='seconds')}</font>"
                ),
                pn.pane.Markdown(
                    """
                - Please keep the Upload ID for the observation planning.
                - You can copy the Upload ID to the clipboard by clicking it.
                """
                ),
            ]

        if not ppp_status:
            self.floatpanel = pn.layout.FloatPanel(
                None,
                # pn.pane.Markdown(message),
                name="⚠️ Warning",
                # config={"headerLogo": "<i class='fa-regular fa-thumbs-up fa-lg'></i>"},
                contained=False,
                position="center",
                # theme="none",
                # theme="#FFF1C2",
                theme="#866208",
                margin=20,
                width=720,
            )

            # JS on-click actions
            # https://github.com/awesome-panel/awesome-panel/blob/master/examples/js_actions.py
            # so far not working...
            stylesheet = """
        :host {
            --font-size: 2.5em;
            --color: #866208;
        }
        .bk-btn-light {
            color: #866208;
        }
        """

            self.copy_source_button = pn.widgets.Button(
                name=f"{secret_token}",
                icon="copy",
                # width=500,
                height=96,
                icon_size="1.5em",
                # button_style="outline",
                button_type="light",
                stylesheets=[stylesheet],
            )

            copy_source_code = "navigator.clipboard.writeText(source);"

            self.copy_source_button.js_on_click(
                args={"source": f"{secret_token}"},
                code=copy_source_code,
            )

            messages = [
                pn.pane.Markdown(
                    "<i class='fa-regular fa-thumbs-up fa-2xl'></i><font size='4'>  Upload successful **_WITHOUT_** pointing simulation! Your **Upload ID** is the following.</font>"
                ),
                self.copy_source_button,
                pn.pane.Markdown(
                    f"<font size='4'>Uploaded at {uploaded_time.isoformat(timespec='seconds')}</font>"
                ),
                pn.pane.Markdown(
                    """
                - Please keep the Upload ID for the observation planning.
                - You can copy the Upload ID to the clipboard by clicking it.
                - **It is not recommended to submit a target list without simulating pointings.**
                """
                ),
            ]

        self.floatpanel.objects = []
        for m in messages:
            self.floatpanel.objects.append(m)


class DocLinkWidgets:
    def __init__(self):
        self.doc = pn.pane.Markdown(
            "<font size='4'><i class='fa-solid fa-circle-info fa-lg' style='color: #3A7D7E;'></i> <a href='doc/index.html' target='_blank'>Documentation</a></font>",
            styles={"text-align": "right"},
        )
        self.pane = pn.Column(self.doc)


class FileInputWidgets(param.Parameterized):
    def __init__(self):
        self.file_input = pn.widgets.FileInput(
            value=None,
            filename=None,
            accept=".csv,.ecsv",
            multiple=False,
            sizing_mode="stretch_width",
        )

        # store previous information for input comparison
        self.previous_filename = None
        self.previous_value = None
        self.previous_mime_type = None

        # hex string to be used as an upload ID
        self.secret_token = None

        self.pane = pn.Column(
            """# Step 1:
## Select a target list (<a href='doc/examples/example_targetlist_random100.csv' target='_blank'>example</a>)""",
            self.file_input,
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


class StatusWidgets:
    def __init__(self, size=20):
        self.status = pn.pane.Alert("", alert_type="light", height=80)
        self.status.visible = False

        self.summary_table = pn.widgets.Tabulator(
            None,
            theme="bootstrap",
            theme_classes=["table-sm"],
            visible=False,
            layout="fit_data_table",
            hidden_columns=["index"],
            selectable=False,
            width=350,
            header_align="right",
            configuration={"columnDefaults": {"headerSort": False}},
            disabled=True,
        )

        self.table_footnote = pn.pane.Markdown(
            "- <font size=2>`N` is the number of `ob_code`s for each priority.</font>\n"
            "- <font size=2>`T` is the total fiberhours of `ob_code`s for each priority.</font>\n"
            "- <font size=2>`L` and `M` correspond to the low- and medium-resolution modes, respectively.</font>",
        )
        self.table_footnote.visible = False

        self.pane = pn.Column(self.status, self.summary_table, self.table_footnote)

    def reset(self):
        self.status.alert_type = "light"
        self.status.visible = False

        self.summary_table.visible = False
        self.table_footnote.visible = False

    def show_results(self, df, validation_status):
        if validation_status["status"]:
            if validation_status["optional_keys"]["status"] and np.any(
                validation_status["visibility"]["success"]
            ):
                self.status.object = "<font size=5>✅ Success</font>"
                self.status.alert_type = "success"
            else:
                # self.status.object = "<i class='fa-regular fa-thumbs-up fa-2xl'></i><font size=5>  Success</font> <font size=3>with warnings</font>"
                # self.status.object = "<i class='fa-solid fa-circle-exclamation fa-2xl'></i><font size=5>  Success</font> <font size=3>with warnings</font>"
                self.status.object = (
                    "<font size=5>⚠️ Success</font><font size=3>  with warnings</font>"
                )
                self.status.alert_type = "success"
                # self.status.alert_type = "warning"
        elif not validation_status["status"]:
            self.status.object = "<font size=5>🚫 Failed</font>"
            self.status.alert_type = "danger"

        self.status.visible = True

        try:
            unique_priority = np.arange(0, 10, 1, dtype=object)
            number_priority_L = np.zeros_like(unique_priority, dtype=int)
            number_priority_M = np.zeros_like(unique_priority, dtype=int)
            exptime_priority_L = np.zeros_like(unique_priority, dtype=float)
            exptime_priority_M = np.zeros_like(unique_priority, dtype=float)

            idx_l = df["resolution"] == "L"
            idx_m = df["resolution"] == "M"

            for i, p in enumerate(unique_priority):
                idx_p = df["priority"] == p
                number_priority_L[i] = df.loc[idx_p & idx_l, :].index.size
                number_priority_M[i] = df.loc[idx_p & idx_m, :].index.size
                exptime_priority_L[i] = df.loc[idx_p & idx_l, "exptime"].sum()
                exptime_priority_M[i] = df.loc[idx_p & idx_m, "exptime"].sum()

            df_summary = pd.DataFrame(
                {
                    "Priority": unique_priority,
                    "N (L)": number_priority_L,
                    "Texp (L)": exptime_priority_L / 3600,
                    "N (M)": number_priority_M,
                    "Texp (M)": exptime_priority_M / 3600,
                }
            )

            df_summary.loc[len(df_summary.index)] = [
                "Other",
                df.loc[idx_l, :].index.size - sum(number_priority_L),
                df.loc[idx_l, "exptime"].sum() - sum(exptime_priority_L),
                df.loc[idx_m, :].index.size - sum(number_priority_M),
                df.loc[idx_m, "exptime"].sum() - sum(exptime_priority_M),
            ]

            df_summary.loc[len(df_summary.index)] = [
                "Total",
                sum(number_priority_L),
                sum(exptime_priority_L / 3600),
                sum(number_priority_M),
                sum(exptime_priority_M / 3600),
            ]

            logger.info(f"Summary Table:\n{df_summary}")

            self.summary_table.value = pd.DataFrame()
            self.summary_table.value = df_summary
            self.summary_table.visible = True
            self.table_footnote.visible = True

        except:
            logger.error("failed to show the summary table in the side bar.")
            pass


class TargetWidgets:
    def __init__(self):
        self.table_all = pn.widgets.Tabulator(
            None,
            page_size=50,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=[],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
            disabled=True,
        )

        self.pane = pn.Column(self.table_all)

    def show_results(self, df):
        # it seems that frozen_columns must be empty when replacing its value
        self.table_all.frozen_columns = []
        # self.table_all.value = pd.DataFrame()
        self.table_all.value = df
        self.table_all.frozen_columns = ["index"]
        self.table_all.visible = True

    def reset(self):
        self.table_all.frozen_columns = []
        self.table_all.value = pd.DataFrame()
        self.table_all.visible = False


class ResultWidgets:
    box_width = 1200
    tabulator_kwargs = dict(
        page_size=50,
        theme="bootstrap",
        theme_classes=["table-striped", "table-sm"],
        frozen_columns=[],
        pagination="remote",
        header_filters=True,
        visible=False,
        layout="fit_data_table",
        disabled=True,
        max_width=box_width,
    )

    def __init__(self):
        # grand title of the main pane
        self.title = pn.pane.Markdown(
            """# Results on the validation of the input list
<font size='3'>Please check the validation results carefully and fix the input list accordingly before proceeding to the submission.</font>
""",
            dedent=True,
        )

        # subsection titles
        self.error_title = pn.pane.Alert(
            """<font size=5>🚫 **Errors**</font>\n\n
<font size=3>Detected errors are listed below. Please fix them.</font>
            """,
            alert_type="danger",
            max_width=self.box_width,
        )
        self.warning_title = pn.pane.Alert(
            """<font size=5>⚠️ **Warnings**</font>\n\n
<font size=3>Detected warnings listed below. Please take a look and fix them if possible and necessary.</font>""",
            alert_type="warning",
            max_width=self.box_width,
        )
        self.info_title = pn.pane.Alert(
            """<font size=5>✅ Info</font>\n\n
<font size=3>The following items are successfully passed the validation.</font>""",
            alert_type="success",
            max_width=self.box_width,
        )

        # subsection texts
        self.error_text_success = pn.pane.Markdown("", max_width=self.box_width)
        self.error_text_keys = pn.pane.Markdown("", max_width=self.box_width)
        self.error_text_str = pn.pane.Markdown("", max_width=self.box_width)
        self.error_text_vals = pn.pane.Markdown("", max_width=self.box_width)
        self.error_text_flux = pn.pane.Markdown("", max_width=self.box_width)
        self.error_text_visibility = pn.pane.Markdown("", max_width=self.box_width)
        self.error_text_dups = pn.pane.Markdown("", max_width=self.box_width)

        # self.warning_text_keys = pn.pane.Markdown(
        #     "<font size=5>Missing optional keys</font>\n", max_width=self.box_width
        # )
        self.warning_text_keys = pn.pane.Markdown("", max_width=self.box_width)
        self.warning_text_str = pn.pane.Markdown("", max_width=self.box_width)
        self.warning_text_vals = pn.pane.Markdown("", max_width=self.box_width)
        self.warning_text_visibility = pn.pane.Markdown("", max_width=self.box_width)

        self.info_text_keys = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_str = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_vals = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_flux = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_visibility = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_dups = pn.pane.Markdown("", max_width=self.box_width)

        self.error_table_str = pn.widgets.Tabulator(None, **self.tabulator_kwargs)
        self.warning_table_str = pn.widgets.Tabulator(None, **self.tabulator_kwargs)

        self.error_table_vals = pn.widgets.Tabulator(None, **self.tabulator_kwargs)
        self.warning_table_vals = pn.widgets.Tabulator(None, **self.tabulator_kwargs)

        self.error_table_flux = pn.widgets.Tabulator(None, **self.tabulator_kwargs)

        self.error_table_visibility = pn.widgets.Tabulator(
            None, **self.tabulator_kwargs
        )
        self.warning_table_visibility = pn.widgets.Tabulator(
            None, **self.tabulator_kwargs
        )

        self.error_table_dups = pn.widgets.Tabulator(None, **self.tabulator_kwargs)

        self.error_pane = pn.Column()
        self.warning_pane = pn.Column()
        self.info_pane = pn.Column()

        self.pane = pn.Column(
            self.title, self.error_pane, self.warning_pane, self.info_pane
        )

    def reset(self):
        for t in [
            self.error_text_success,
            self.error_text_keys,
            self.error_text_str,
            self.error_text_vals,
            self.error_text_flux,
            self.error_text_visibility,
            self.error_text_dups,
            self.warning_text_keys,
            self.warning_text_str,
            self.warning_text_vals,
            self.warning_text_visibility,
            self.info_text_keys,
            self.info_text_str,
            self.info_text_vals,
            self.info_text_flux,
            self.info_text_visibility,
            self.info_text_dups,
        ]:
            t.object = ""

        for t in [
            self.error_table_str,
            self.warning_table_str,
            self.error_table_vals,
            self.warning_table_vals,
            self.error_table_flux,
            self.error_table_visibility,
            self.warning_table_visibility,
            self.error_table_dups,
        ]:
            t.value = pd.DataFrame()
            t.visible = False

        self.error_pane.objects.clear()
        self.warning_pane.objects.clear()
        self.info_pane.objects.clear()

        # self.pane.objects.clear()

    def append_title(self, flag, status_str):
        if status_str == "error":
            if not flag:
                self.error_pane.append(self.error_title)
                flag = True
        if status_str == "warning":
            if not flag:
                self.warning_pane.append(self.warning_title)
                flag = True
        if status_str == "info":
            if not flag:
                self.info_pane.append(self.info_title)
                flag = True
        return flag

    def show_results(self, df, validation_status):
        # self.pane.append(self.title)

        is_error = False
        is_warning = False
        is_info = False

        if validation_status["status"]:
            self.error_title.visible = False

        # Errors on missing required keys
        if not validation_status["required_keys"]["status"]:
            self.error_text_keys.object = (
                "<font size=4><u>Missing required columns</u></font>\n"
            )
            for desc in validation_status["required_keys"]["desc_error"]:
                self.error_text_keys.object += f"- <font size='3'>{desc}</font>\n"
            is_error = self.append_title(is_error, "error")
            self.error_pane.append(self.error_text_keys)

        # Warnings on missing optional keys
        if not validation_status["optional_keys"]["status"]:
            self.warning_text_keys.object = (
                "<font size=4><u>Missing optional columns</u></font>\n"
            )
            for desc in validation_status["optional_keys"]["desc_warning"]:
                self.warning_text_keys.object += f"- <font size='3'>{desc}</font>\n"
            is_warning = self.append_title(is_warning, "warning")
            self.warning_pane.append(self.warning_text_keys)

        # Info on discovered keys
        n_req_success = len(validation_status["required_keys"]["desc_success"])
        n_opt_success = len(validation_status["optional_keys"]["desc_success"])
        if n_req_success + n_opt_success > 0:
            is_info = self.append_title(is_info, "info")
            self.info_text_keys.object = (
                "<font size=4><u>Discovered columns</u></font>\n"
            )
            for desc in validation_status["required_keys"]["desc_success"]:
                self.info_text_keys.object += f"- <font size='3'>{desc}</font>\n"
            for desc in validation_status["optional_keys"]["desc_success"]:
                self.info_text_keys.object += f"- <font size='3'>{desc}</font>\n"
            self.info_pane.append(self.info_text_keys)

        # if there are missing required columns, return immediately
        if not validation_status["required_keys"]["status"]:
            return

        # String values
        if validation_status["str"]["status"] is None:
            pass
        elif validation_status["str"]["status"]:
            is_info = self.append_title(is_info, "info")
            self.info_text_str.object = """<font size=4><u>String values</u></font>

<font size=3>All string values consist of `[A-Za-z0-9_-+.]` </font>"""
            self.info_pane.append(self.info_text_str)
        elif not validation_status["str"]["status"]:
            is_error = self.append_title(is_error, "error")
            self.error_text_str.object = """<font size=4><u>Invalid characters in string values</u></font>

<font size=3>String values must consist of `[A-Za-z0-9_-+.]`. The following entries must be fixed.</font>"""

            is_invalid_str = np.logical_or(
                ~validation_status["str"]["success_required"],
                ~validation_status["str"]["success_optional"],
            )
            self.error_table_str.frozen_columns = []
            # self.error_table_str.value = pd.DataFrame()
            self.error_table_str.value = df.loc[is_invalid_str, :]
            self.error_table_str.frozen_columns = ["index"]
            self.error_pane.append(self.error_text_str)
            self.error_pane.append(self.error_table_str)
            self.error_table_str.visible = True

        # If string validation failed, retrun immediately
        if not validation_status["str"]["status"]:
            return

        # Data range
        if validation_status["values"]["status"] is None:
            pass
        elif validation_status["values"]["status"]:
            is_info = self.append_title(is_info, "info")
            self.info_text_vals.object = """<font size=4><u>Data ranges</u></font>

<font size=3>All values of `ra`, `dec`, `priority`, `exptime`, and `resolution` satisfy the allowed ranges (see [documentation](doc/validation.html)).</font>
"""
            self.info_pane.append(self.info_text_vals)
        elif not validation_status["values"]["status"]:
            is_error = self.append_title(is_error, "error")
            self.error_text_vals.object = """<font size=4><u>Value errors</u></font>

<font size=3>Invalid values are detected for the following columns in the following entries (see [documentation](doc/validation.html)).</font>
"""
            for k, v in zip(
                ["ra", "dec", "priority", "exptime", "resolution"],
                [
                    "0 < `ra` < 360",
                    "-90 < `dec` < 90",
                    "[0, 9]",
                    "positive `float` value",
                    "`L` or `M`",
                ],
            ):
                if not validation_status["values"][f"status_{k}"]:
                    self.error_text_vals.object += (
                        f"- <font size=3>`{k}` ({v})</font>\n"
                    )
            self.error_table_vals.frozen_columns = []
            # self.error_table_vals.value = pd.DataFrame()
            self.error_table_vals.value = df.loc[
                ~validation_status["values"]["success"], :
            ]
            self.error_table_vals.frozen_columns = ["index"]
            self.error_pane.append(self.error_text_vals)
            self.error_pane.append(self.error_table_vals)
            self.error_table_vals.visible = True

        # If invalid values are detected, return immediately
        if not validation_status["values"]["status"]:
            return

        # flux columns
        # TODO: show a list of detected/undetected flux columns
        if validation_status["flux"]["status"]:
            is_info = self.append_title(is_info, "info")
            self.info_text_flux.object = "<font size=4><u>Flux information</u></font>\n\n<font size=3>All `ob_code`s have at least one flux information. The detected filters are the following: </font>"
            for f in validation_status["flux"]["filters"]:
                self.info_text_flux.object += f"<font size=3>`{f}`</font>, "
            self.info_text_flux.object = self.info_text_flux.object[:-2]

            self.info_pane.append(self.info_text_flux)
            self.error_table_flux.visible = False
        else:
            is_error = self.append_title(is_error, "error")
            # add an error message and data table for duplicates
            self.error_text_flux.object = "<font size=4><u>Missing flux information</u></font>\n\n<font size=3>No flux information found in the following `ob_code`s. Detected filters are the following: </font>"
            for f in validation_status["flux"]["filters"]:
                self.error_text_flux.object += f"<font size=3>`{f}`</font>, "
            if len(validation_status["flux"]["filters"]) > 0:
                self.error_text_flux.object = self.error_text_flux.object[:-2]

            self.error_table_flux.frozen_columns = []
            # self.error_table_flux.value = pd.DataFrame()
            self.error_table_flux.value = df.loc[
                ~validation_status["flux"]["success"], :
            ]
            self.error_table_flux.frozen_columns = ["index"]
            self.error_pane.append(self.error_text_flux)
            self.error_pane.append(self.error_table_flux)
            self.error_table_flux.visible = True

        # Visibility
        # TODO: add begin_date and end_date in the message
        if validation_status["visibility"]["status"]:
            if np.all(validation_status["visibility"]["success"]):
                is_info = self.append_title(is_info, "info")
                self.info_text_visibility.object = "<font size=4><u>Visibility</u></font>\n\n<font size=3>All `ob_code`s are visible in the input observing period.</font>"
                self.info_pane.append(self.info_text_visibility)
            elif np.any(validation_status["visibility"]["success"]):
                is_warning = self.append_title(is_warning, "warning")
                n_invisible = np.count_nonzero(
                    ~validation_status["visibility"]["success"]
                )
                self.warning_text_visibility.object = (
                    "<font size=4><u>Visibility</u></font>\n\n"
                )
                if n_invisible == 1:
                    self.warning_text_visibility.object += f"<font size=3>{n_invisible} `ob_code` is not visible in the input observing period</font>"
                else:
                    self.warning_text_visibility.object += f"<font size=3>{n_invisible} `ob_code`s are not visible in the input observing period</font>"
                self.warning_text_visibility.object += (
                    "<font size=3> (see the following table).</font>"
                )
                # self.warning_text_visibility.value = pd.DataFrame()
                self.warning_table_visibility.frozen_columns = []
                dfout = df.loc[~validation_status["visibility"]["success"], :]
                self.warning_table_visibility.value = dfout
                self.warning_table_visibility.frozen_columns = ["index"]
                self.warning_pane.append(self.warning_text_visibility)
                self.warning_pane.append(self.warning_table_visibility)
                self.warning_table_visibility.visible = True
            self.error_table_visibility.visible = False
        else:
            is_error = self.append_title(is_error, "error")
            # add an error message and data table for duplicates
            self.error_text_visibility.object = "<font size=4><u>Visibility</u></font>\n\n<font size='3'>None of `ob_code`s in the list is visible in the input observing period.</font>"
            self.error_pane.append(self.error_text_visibility)

        # Duplication
        if validation_status["unique"]["status"]:
            is_info = self.append_title(is_info, "info")
            self.info_text_dups.object = "<font size=4><u>Uniqueness of `ob_code`s</u></font>\n\n<font size=3>All `ob_code` are unique.</font>"
            self.info_pane.append(self.info_text_dups)
            self.error_table_dups.visible = False
        else:
            is_error = self.append_title(is_error, "error")
            # add an error message and data table for duplicates
            self.error_text_dups.object = "<font size=4><u>Duplication of `ob_code`s </u></font>\n\n<font size=3>Each `ob_code` must be unique within a proposal, but duplicate `ob_code` detected in the following targets</font>"
            self.error_table_dups.frozen_columns = []
            # self.error_table_dups.value = pd.DataFrame()
            self.error_table_dups.value = df.loc[
                validation_status["unique"]["flags"], :
            ]
            self.error_pane.append(self.error_text_dups)
            self.error_pane.append(self.error_table_dups)
            self.error_table_dups.frozen_columns = ["index"]
            self.error_table_dups.visible = True

        if (
            validation_status["required_keys"]["status"]
            and validation_status["str"]["status"]
            and validation_status["values"]["status"]
            and validation_status["flux"]["status"]
            and validation_status["visibility"]["status"]
            and validation_status["unique"]["status"]
        ):
            self.error_text_success.visible = False


class PPPresultWidgets:
    box_width = 1200

    def __init__(self):
        # PPP status
        # True if PPP has been run
        # False if PPP has not been run
        self.ppp_status = True

        self.ppp_title = pn.pane.Markdown(
            """# Results of PPP""",
            dedent=True,
        )

        self.ppp_warning = pn.pane.Alert(
            "<font size=5>⚠️ **Warnings**</font>\n\n"
            "<font size=3>The total requested time exceeds the 5-night (max. request for a normal program). "
            "Please make sure to adjust it to your requirement before proceeding to the submission. "
            "Note that targets observable in the input observing period are considered.</font>",
            alert_type="warning",
            max_width=self.box_width,
        )

        self.ppp_success = pn.pane.Alert(
            "<font size=5>✅ **Success**</font>\n\n"
            "<font size=3>The total requested time is reasonable for normal program. "
            "Note that targets observable in the input period are considered.</font>",
            alert_type="success",
            max_width=self.box_width,
        )

        self.ppp_figure = pn.Column()

        self.ppp_alert = pn.Column()

        self.pane = pn.Column(
            self.ppp_title,
            self.ppp_figure,
        )

    def reset(self):
        self.ppp_alert.clear()
        self.ppp_figure.clear()
        self.ppp_figure.visible = False
        self.ppp_status = False

    def show_results(self):  # , mode, nppc, p_result_fig, p_result_tab, ppp_Alert):
        logger.info("showing PPP results")
        self.ppp_figure.append(self.ppp_alert)
        self.ppp_figure.append(
            pn.pane.Markdown(
                f"""## For the {self.res_mode:s} resolution mode:""",
                dedent=True,
            )
        )
        self.ppp_figure.append(self.nppc)
        self.ppp_figure.append(self.p_result_tab)
        self.ppp_figure.append(self.p_result_fig)
        self.ppp_figure.visible = True

        size_of_ppp_figure = sys.getsizeof(self.p_result_fig) * u.byte
        logger.info(
            f"size of the ppp_figure object is {size_of_ppp_figure.to(u.kilobyte)}"
        )
        logger.info("showing PPP results done")

    def run_ppp(self, df, validation_status, weights=None):
        if weights is None:
            weights = [4.02, 0.01, 0.01]

        tb_input = Table.from_pandas(df)
        tb_visible = tb_input[validation_status["visibility"]["success"]]

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
        ) = PPPrunStart(tb_visible, weights)

        (
            self.res_mode,
            self.nppc,
            self.p_result_fig,
            self.p_result_ppc,
            self.p_result_tab,
        ) = ppp_result(
            cR_L_, sub_l, obj_allo_L_fin, uS_L2, cR_M_, sub_m, obj_allo_M_fin, uS_M2
        )

        if self.p_result_tab.value.iloc[-1]["Request time (h)"] > 10 * 5:
            self.ppp_alert.append(self.ppp_warning)
        else:
            self.ppp_alert.append(self.ppp_success)

        self.ppp_status = True


class ValidateButtonWidgets:
    def __init__(self):
        self.validate = pn.widgets.Button(
            name="Validate",
            button_style="outline",
            button_type="primary",
            icon="stethoscope",
        )
        self.pane = pn.Column(
            """# Step 2:
## Check the uploaded list""",
            pn.Row(self.validate),
        )


class RunPppButtonWidgets:
    def __init__(self):
        self.PPPrun = pn.widgets.Button(
            name="Start (takes a few minutes ~ half hour)",
            button_style="outline",
            button_type="primary",
            icon="player-play-filled",
        )
        # loading spinner
        self.gif_pane = pn.pane.GIF(
            "https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif",
            width=20,
        )

        # placeholder for loading spinner
        self.PPPrunStats = pn.Column("", width=100)

        self.pane = pn.Column(
            """# Step 3:
## Estimate the total required time""",
            self.PPPrun,
            self.PPPrunStats,
        )

    def start(self):
        self.PPPrunStats.append(self.gif_pane)

    def stop(self):
        self.PPPrunStats.remove(self.gif_pane)


class SubmitButtonWidgets:
    def __init__(self):
        self.submit = pn.widgets.Button(
            name="Submit Results",
            button_type="primary",
            icon="send",
            disabled=True,
        )
        self.pane = pn.Column(
            """# Step 4:
## Submit results""",
            self.submit,
        )


class DatePickerWidgets(param.Parameterized):
    def __init__(self):
        today = datetime.now(ZoneInfo("US/Hawaii"))

        semester_begin, semester_end = get_semester_daterange(
            today.date(),
            next=True,
        )

        self.date_begin = pn.widgets.DatePicker(
            name="Date Begin (HST)", value=semester_begin.date()
        )
        self.date_end = pn.widgets.DatePicker(
            name="Date End (HST)", value=semester_end.date()
        )

        self.pane = pn.Column("### Observation Period", self.date_begin, self.date_end)
