#!/usr/bin/env python3

import os
import secrets
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import panel as pn
import param
from logzero import logger
from zoneinfo import ZoneInfo

from .utils.checker import get_semester_daterange, validate_input
from .utils.io import load_input


class UploadNoteWidgets:
    # def __init__(self, message):
    def __init__(self, secret_token, uploaded_time):
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
## Select a target list (<a href='doc/examples/example_targetlist.csv' target='_blank'>example</a>)""",
            self.file_input,
        )

    def reset(self):
        self.file_input.filename = None
        self.file_input.mime_type = None
        self.file_input.value = None

    def validate(self, date_begin=None, date_end=None):
        if date_begin >= date_end:
            pn.state.notifications.error(
                "Date Begin must be before Date End.", duration=0
            )
            return None, None

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
                return None, None
        else:
            logger.info("No file selected.")
            pn.state.notifications.error("Please select a CSV file.")
            return None, None

        validation_status = validate_input(
            df_input, date_begin=date_begin, date_end=date_end
        )

        return df_input, validation_status


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

        self.pane = pn.Column(
            self.status,
            self.summary_table,
        )

    def reset(self):
        self.status.alert_type = "light"
        self.status.visible = False

        self.summary_table.visible = False

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
            unique_priority = np.arange(0, 10, 1)  # np.unique(df["priority"])
            number_priority_L = np.zeros_like(unique_priority, dtype=int)
            number_priority_M = np.zeros_like(unique_priority, dtype=int)
            exptime_priority_L = np.zeros_like(unique_priority, dtype=float)
            exptime_priority_M = np.zeros_like(unique_priority, dtype=float)

            for i, p in enumerate(unique_priority):
                number_priority_L[i] = df.loc[
                    (df["priority"] == p) & (df["resolution"] == "L"), :
                ].index.size
                number_priority_M[i] = df.loc[
                    (df["priority"] == p) & (df["resolution"] == "M"), :
                ].index.size
                exptime_priority_L[i] = df.loc[
                    (df["priority"] == p) & (df["resolution"] == "L"), "exptime"
                ].sum()
                exptime_priority_M[i] = df.loc[
                    (df["priority"] == p) & (df["resolution"] == "M"), "exptime"
                ].sum()

            df_summary = pd.DataFrame(
                {
                    "priority": unique_priority,
                    "N_L": number_priority_L,
                    "Texp_L (FH)": exptime_priority_L / 3600,
                    "N_M": number_priority_M,
                    "Texp_M (FH)": exptime_priority_M / 3600,
                }
            )
            df_summary.loc[len(df_summary.index)] = [
                "total",
                sum(number_priority_L),
                sum(exptime_priority_L / 3600),
                sum(number_priority_M),
                sum(exptime_priority_M / 3600),
            ]

            self.summary_table.value = pd.DataFrame()
            self.summary_table.value = df_summary
            self.summary_table.visible = True

        except:
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
    )

    def __init__(self):
        # grand title of the main pane
        self.title = pn.pane.Markdown(
            """# Validation results
<font size='3'>Please check the validation results carefully and fix any errors to proceed to the submission.</font>
""",
            # renderer="myst",
            dedent=True,
        )

        # subsection titles
        self.error_title = pn.pane.Alert(
            """### Errors
Detected errors are listed below. Please fix them.
            """,
            alert_type="danger",
        )
        self.warning_title = pn.pane.Alert(
            """### Warnings
Detected warnings detected. Please take a look and fix them if possible and necessary.""",
            alert_type="warning",
        )
        self.info_title = pn.pane.Alert(
            """### Info""",
            alert_type="success",
        )

        # subsection texts
        self.error_text_success = pn.pane.Markdown("")
        self.error_text_keys = pn.pane.Markdown("")
        self.error_text_str = pn.pane.Markdown("")
        self.error_text_vals = pn.pane.Markdown("")
        self.error_text_flux = pn.pane.Markdown("")
        self.error_text_visibility = pn.pane.Markdown("")
        self.error_text_dups = pn.pane.Markdown("")

        self.warning_text_keys = pn.pane.Markdown("")
        self.warning_text_str = pn.pane.Markdown("")
        self.warning_text_vals = pn.pane.Markdown("")
        self.warning_text_visibility = pn.pane.Markdown("")

        self.info_text_keys = pn.pane.Markdown("")
        self.info_text_str = pn.pane.Markdown("")
        self.info_text_vals = pn.pane.Markdown("")
        self.info_text_flux = pn.pane.Markdown("")
        self.info_text_visibility = pn.pane.Markdown("")
        self.info_text_dups = pn.pane.Markdown("")

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

        self.pane = pn.Column(
            self.title,
            # put errors first
            self.error_title,
            # show success message
            self.error_text_success,
            # errors on required keys (no need to put a table)
            self.error_text_keys,
            # errors on characters in string values
            self.error_text_str,
            self.error_table_str,
            # errors on out-of-range values
            self.error_text_vals,
            self.error_table_vals,
            # errors on flux columns
            self.error_text_flux,
            self.error_table_flux,
            # errors on visibility
            self.error_text_visibility,
            self.error_table_visibility,
            # errors on duplicate ob_codes
            self.error_text_dups,
            self.error_table_dups,
            # warnings next
            self.warning_title,
            # warnings on optional keys
            self.warning_text_keys,
            # warnings on characters in string values (for optional keywords)
            self.warning_text_str,
            self.warning_table_str,
            # warnings on out-of-range values
            self.warning_text_vals,
            self.warning_table_vals,
            # warnings on visibility
            self.warning_text_visibility,
            self.warning_table_visibility,
            # successful vaildations last
            self.info_title,
            self.info_text_keys,
            self.info_text_str,
            self.info_text_vals,
            self.info_text_flux,
            self.info_text_visibility,
            self.info_text_dups,
            # height=200,
        )

    def reset(self):
        self.error_text_success.object = "\n####"
        self.error_text_keys.object = "\n####"
        self.error_text_str.object = "\n####"
        self.error_text_vals.object = "\n####"
        self.error_text_flux.object = "\n####"
        self.error_text_visibility.object = "\n####"
        self.error_text_dups.object = "\n####"

        self.warning_text_keys.object = "\n####"
        self.warning_text_str.object = "\n####"
        self.warning_text_vals.object = "\n####"
        self.warning_text_visibility.object = "\n####"

        self.info_text_keys.object = "\n####"
        self.info_text_str.object = "\n####"
        self.info_text_vals.object = "\n####"
        self.info_text_flux.object = "\n####"
        self.info_text_visibility.object = "\n####"
        self.info_text_dups.object = "\n####"

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

    def show_results(self, df, validation_status):
        if validation_status["status"]:
            self.error_title.visible = False

        # Stage 1 results
        for desc in validation_status["required_keys"]["desc_error"]:
            self.error_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["optional_keys"]["desc_warning"]:
            self.warning_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["required_keys"]["desc_success"]:
            self.info_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["optional_keys"]["desc_success"]:
            self.info_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        # Stage 2 results
        if not validation_status["required_keys"]["status"]:
            return

        if validation_status["str"]["status"] is None:
            pass
        elif validation_status["str"]["status"]:
            pass
        elif not validation_status["str"]["status"]:
            self.error_text_str.object += """\n
<font size='3'>String values must consist of `[A-Za-z0-9_-+.]`.</font>
<font size='3'>The following entries must be fixed.</font>
                """
            self.error_table_str.frozen_columns = []
            # self.error_table_str.value = pd.DataFrame()
            self.error_table_str.value = df.loc[
                ~validation_status["str"]["success_required"], :
            ]
            self.error_table_str.frozen_columns = ["index"]
            self.error_table_str.visible = True

        if not validation_status["str"]["status_optional"]:
            self.warning_text_str.object += """\n
<font size='3'>String values must consist of `[A-Za-z0-9_-+.]`.</font>
<font size='3'>The following entries must be fixed.</font>
                """
            self.warning_table_str.frozen_columns = []
            # self.warning_table_str.value = pd.DataFrame()
            self.warning_table_str.value = df.loc[
                ~validation_status["str"]["success_optional"], :
            ]
            self.warning_table_str.frozen_columns = ["index"]
            self.warning_table_str.visible = True

        # Stage 3 results
        if not validation_status["str"]["status"]:
            return

        if validation_status["values"]["status"] is None:
            pass
        elif validation_status["values"]["status"]:
            pass
        elif not validation_status["values"]["status"]:
            self.error_text_vals.object += """\n<font size='3'>Errors in values are detected for the following entries (See [documentation](doc/validation.html#stage-3)). </font>"""
            self.error_table_vals.frozen_columns = []
            # self.error_table_vals.value = pd.DataFrame()
            self.error_table_vals.value = df.loc[
                ~validation_status["values"]["success"], :
            ]
            self.error_table_vals.frozen_columns = ["index"]
            self.error_table_vals.visible = True

        # Stage 3' results
        if not validation_status["values"]["status"]:
            return

        if validation_status["flux"]["status"]:
            self.info_text_flux.object += "\n<font size='3'>All `ob_code`s have at least one flux information</font>\n"
            self.error_table_flux.visible = False
        else:
            # add an error message and data table for duplicates
            self.error_text_flux.object += "\n<font size='3'>No flux info found in the following `ob_code`s:</font>\n"
            self.error_table_flux.frozen_columns = []
            # self.error_table_flux.value = pd.DataFrame()
            self.error_table_flux.value = df.loc[
                ~validation_status["flux"]["success"], :
            ]
            self.error_table_flux.frozen_columns = ["index"]
            self.error_table_flux.visible = True

        # Stage 3'' results
        if validation_status["visibility"]["status"]:
            if np.all(validation_status["visibility"]["success"]):
                self.info_text_flux.object += "\n<font size='3'>All `ob_code`s are visible in the input observing period.</font>\n"
            elif np.any(validation_status["visibility"]["success"]):
                self.warning_text_visibility.object += "\n<font size='3'>Some `ob_code`s are not visible in the input observing period.</font>\n"
                # self.warning_text_visibility.value = pd.DataFrame()
                self.error_table_flux.frozen_columns = []
                dfout = df.loc[~validation_status["visibility"]["success"], :]
                self.warning_table_visibility.value = dfout
                self.error_table_flux.frozen_columns = ["index"]
                self.warning_table_visibility.visible = True
            self.error_table_visibility.visible = False
        else:
            # add an error message and data table for duplicates
            self.error_text_visibility.object += "\n<font size='3'>None of `ob_code`s are visible in the input observing period.</font>\n"

        # Stage 4 results
        if validation_status["unique"]["status"]:
            self.info_text_dups.object += (
                "\n<font size='3'>All `ob_code` are unique</font>\n"
            )
            # tweak for text update (I don't know the cause)
            # self.error_text_dups.object += "\n###"
            self.error_table_dups.visible = False
        else:
            # add an error message and data table for duplicates
            self.error_text_dups.object += "\n<font size='3'>`ob_code` must be unique within a proposal, but duplicate `ob_code` detected in the following targets:</font>\n"
            self.error_table_dups.frozen_columns = []
            # self.error_table_dups.value = pd.DataFrame()
            self.error_table_dups.value = df.loc[
                validation_status["unique"]["flags"], :
            ]
            self.error_table_dups.frozen_columns = ["index"]

            # BUG: it seems that the pandas-like styling does not work for panel>1.0.4 or so.
            # def _set_column_color(x, c="red"):
            #     print("setting background for the ob_code column")
            #     return [f"background-color: {c}" for _ in x]
            # self.table_duplicate.style.apply(
            #     _set_column_color,
            #     axis=0,
            #     subset=["ob_code"],
            #     # c="green",
            # )
            self.error_table_dups.visible = True

        if (
            validation_status["required_keys"]["status"]
            and validation_status["str"]["status"]
            and validation_status["values"]["status"]
            and validation_status["flux"]["status"]
            and validation_status["visibility"]["status"]
            and validation_status["unique"]["status"]
        ):
            # self.error_text_success.object += "\n<font size='3'>No error is found. Congratulations. You can proceed to the submission.</font>\n"
            self.error_text_success.visible = False


class PPPresultWidgets:
    def __init__(self):
        self.ppp_title = pn.pane.Markdown(
            """# Results of PPP""",
            dedent=True,
        )

        self.ppp_figure = pn.Column("")

        self.pane = pn.Column(
            # "# Status",
            self.ppp_title,
            self.ppp_figure,
        )

    def reset(self):
        self.ppp_figure.clear()
        self.ppp_figure.visible = False

    def show_results(self, mode, nppc, p_result_fig, p_result_tab, ppp_Alert):
        self.ppp_figure.append(ppp_Alert)
        self.ppp_figure.append(
            pn.pane.Markdown(
                f"""## For the {mode:s} resolution mode:""",
                dedent=True,
            )
        )
        self.ppp_figure.append(nppc)
        self.ppp_figure.append(p_result_tab)
        self.ppp_figure.append(p_result_fig)
        self.ppp_figure.visible = True


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
        self.PPPrunStats = pn.Column("", width=100)

        self.pane = pn.Column(
            """# Step 3:
## Estimate the total required time""",
            self.PPPrun,
            self.PPPrunStats,
        )


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
