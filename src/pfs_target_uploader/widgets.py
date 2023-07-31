#!/usr/bin/env python3

import numpy as np
import pandas as pd
import panel as pn
from logzero import logger


class UploadNoteWidgets:
    def __init__(self, message):
        self.floatpanel = pn.layout.FloatPanel(
            pn.pane.Markdown(message),
            name="Info",
            # config={"headerLogo": "<i class='fa-regular fa-thumbs-up fa-lg'></i>"},
            contained=False,
            position="center",
            # theme="none",
            theme="#3A7D7E",
            margin=20,
            width=720,
        )


class DocLinkWidgets:
    def __init__(self):
        self.doc = pn.pane.Markdown(
            "<font size='4'><i class='fa-solid fa-circle-info fa-lg' style='color: #3A7D7E;'></i> <a href='/docs/index.html' target='_blank'>Documentation</a></font>",
            styles={"text-align": "right"},
        )
        self.pane = pn.Column(self.doc)


class FileInputWidgets:
    def __init__(self):
        self.file_input = pn.widgets.FileInput(accept=".csv", multiple=False)
        self.pane = pn.Column("# Input target list", self.file_input)


class ButtonWidgets:
    def __init__(self):
        self.validate = pn.widgets.Button(
            name="Validate",
            button_style="outline",
            # button_type="success",
            button_type="primary",
            icon="stethoscope",
        )
        self.submit = pn.widgets.Button(
            name="Submit",
            # button_style="outline",
            button_type="primary",
            icon="send",
            disabled=True,
            # stylesheets=[":host { --design-background-color: red; }"],
        )
        self.pane = pn.Row(self.validate, self.submit, height=40)


class StatusWidgets:
    def __init__(self):
        self.status_keys = pn.indicators.BooleanStatus(
            width=54, height=54, value=False, color="danger"
        )
        self.status_str = pn.indicators.BooleanStatus(
            width=54, height=54, value=False, color="danger"
        )
        self.status_vals = pn.indicators.BooleanStatus(
            width=54, height=54, value=False, color="danger"
        )
        self.status_dups = pn.indicators.BooleanStatus(
            width=54, height=54, value=False, color="danger"
        )

        self.status_grid = pn.GridBox(
            "<font size='3'>Stage 1</font>",
            "<font size='3'>Stage 2</font>",
            "<font size='3'>Stage 3</font>",
            "<font size='3'>Stage 4</font>",
            self.status_keys,
            self.status_str,
            self.status_vals,
            self.status_dups,
            ncols=4,
            nrows=1,
            # width=300,
        )

        self.summary_nobj = pn.indicators.Number(
            name="Number of objects",
            value=0,
            format="{value:d}",
            title_size="24pt",
            default_color="teal",
        )
        self.summary_fh = pn.indicators.Number(
            name="Fiberhours",
            value=0,
            format="{value} h",
            title_size="24pt",
            default_color="teal",
        )

        self.summary_text = pn.pane.Markdown("")

        self.summary_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            page_size=10,
            theme="bootstrap",
            theme_classes=["table-sm"],
            pagination="remote",
            visible=False,
            layout="fit_columns",
            hidden_columns=["index"],
            selectable=False,
            width=300,
            header_align="right",
            configuration={"columnDefaults": {"headerSort": False}},
        )

        self.pane = pn.Column(
            # "# Status",
            self.status_grid,
            self.summary_nobj,
            self.summary_fh,
            self.summary_text,
            self.summary_table,
        )

    def reset(self):
        self.status_keys.value = False
        self.status_str.value = False
        self.status_vals.value = False
        self.status_dups.value = False
        self.summary_nobj.value = 0
        self.summary_fh.value = 0
        self.summary_text.object = ""
        self.summary_table.value = pd.DataFrame()
        self.summary_table.visible = False

    def show_results(self, df, validation_status):
        # self.reset()

        # logger.info(validation_status)

        if validation_status["required_keys"]["status"] is None:
            pass
        elif validation_status["required_keys"]["status"]:
            self.status_keys.value = True
            if validation_status["optional_keys"]["status"]:
                self.status_keys.color = "success"
            else:
                self.status_keys.color = "warning"
        elif not validation_status["required_keys"]["status"]:
            self.status_keys.value = True
            self.status_keys.color = "danger"

        if validation_status["required_keys"]["status"]:
            if validation_status["str"]["status"] is None:
                pass
            elif validation_status["str"]["status"]:
                self.status_str.value = True
                self.status_str.color = "success"
            elif not validation_status["str"]["status"]:
                self.status_str.value = True
                self.status_str.color = "danger"

        if validation_status["str"]["status"]:
            if validation_status["values"]["status"] is None:
                pass
            elif validation_status["values"]["status"]:
                self.status_vals.value = True
                self.status_vals.color = "success"
            elif not validation_status["values"]["status"]:
                self.status_vals.value = True
                self.status_vals.color = "danger"

        if validation_status["values"]["status"]:
            if validation_status["unique"]["status"] is None:
                pass
            elif validation_status["unique"]["status"]:
                self.status_dups.value = True
                self.status_dups.color = "success"
            elif not validation_status["unique"]["status"]:
                self.status_dups.value = True
                self.status_dups.color = "danger"

        try:
            self.summary_nobj.value = df.index.size
            self.summary_fh.value = df["exptime"].sum() / 3600.0

            self.summary_text.object += """\n
<font size='3'>Breakdown of fiberhours for each priority is listed below.</font>
"""

            unique_priority = np.unique(df["priority"])
            number_priority = np.zeros_like(unique_priority, dtype=int)
            exptime_priority = np.zeros_like(unique_priority, dtype=float)

            for i, p in enumerate(unique_priority):
                number_priority[i] = df.loc[df["priority"] == p, :].index.size
                exptime_priority[i] = df.loc[df["priority"] == p, "exptime"].sum()

            df_summary = pd.DataFrame(
                {
                    "priority": unique_priority,
                    "N": number_priority,
                    "Texp (h)": exptime_priority / 3600,
                }
            )

            self.summary_table.value = df_summary
            self.summary_table.editors = {"priority": None, "N": None, "Texp (h)": None}
            self.summary_table.visible = True
        except:
            pass


class TargetWidgets:
    def __init__(self):
        self.table_all = pn.widgets.Tabulator(
            pd.DataFrame(),
            page_size=500,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=["index"],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
        )

        self.pane = pn.Column(self.table_all)

    def show_results(self, df):
        tabulator_editors = {}
        for c in df.columns:
            tabulator_editors[c] = None
        self.table_all.value = df
        self.table_all.editors = tabulator_editors
        self.table_all.visible = True


class ResultWidgets:
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
        self.error_text_dups = pn.pane.Markdown("")

        self.warning_text_keys = pn.pane.Markdown("")
        self.warning_text_str = pn.pane.Markdown("")
        self.warning_text_vals = pn.pane.Markdown("")

        self.info_text_keys = pn.pane.Markdown("")
        self.info_text_str = pn.pane.Markdown("")
        self.info_text_vals = pn.pane.Markdown("")
        self.info_text_dups = pn.pane.Markdown("")

        self.error_table_str = pn.widgets.Tabulator(
            pd.DataFrame(),
            page_size=500,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=["index"],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
        )
        self.warning_table_str = pn.widgets.Tabulator(
            pd.DataFrame(),
            page_size=500,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=["index"],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
        )

        self.error_table_vals = pn.widgets.Tabulator(
            pd.DataFrame(),
            page_size=500,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=["index"],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
        )

        self.warning_table_vals = pn.widgets.Tabulator(
            pd.DataFrame(),
            page_size=500,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=["index"],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
        )

        self.error_table_dups = pn.widgets.Tabulator(
            pd.DataFrame(),
            page_size=500,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=["index"],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
        )

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
            # successful vaildations last
            self.info_title,
            self.info_text_keys,
            self.info_text_str,
            self.info_text_vals,
            self.info_text_dups,
            # height=200,
        )

    def reset(self):
        self.error_text_success.object = "\n####"
        self.error_text_keys.object = "\n####"
        self.error_text_str.object = "\n####"
        self.error_text_vals.object = "\n####"
        self.error_text_dups.object = "\n####"

        self.warning_text_keys.object = "\n####"
        self.warning_text_str.object = "\n####"
        self.warning_text_vals.object = "\n####"

        self.info_text_keys.object = "\n####"
        self.info_text_str.object = "\n####"
        self.info_text_vals.object = "\n####"
        self.info_text_dups.object = "\n####"

        self.error_table_str.visible = False
        self.error_table_dups.visible = False
        self.error_table_vals.visible = False

        self.warning_table_str.visible = False
        self.warning_table_vals.visible = False

    def show_results(self, df, validation_status, tabulator_editors=None):
        # self.reset()

        if tabulator_editors is None:
            tabulator_editors = {}
            for c in df.columns:
                tabulator_editors[c] = None

        # Stage 1 results
        for desc in validation_status["required_keys"]["desc_error"]:
            self.error_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["optional_keys"]["desc_warning"]:
            self.warning_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["required_keys"]["desc_success"]:
            self.info_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["optional_keys"]["desc_success"]:
            self.info_text_keys.object += f"\n<font size='4'>{desc}</font>\n"

        # Stage 2 results
        if validation_status["required_keys"]["status"]:
            if validation_status["str"]["status"] is None:
                pass
            elif validation_status["str"]["status"]:
                pass
            elif not validation_status["str"]["status"]:
                self.error_text_str.object += """\n
<font size='3'>String values must consist of `[A-Za-z0-9_-+.]`.</font>
<font size='3'>The following entries must be fixed.</font>
                """
                self.error_table_str.value = df.loc[
                    ~validation_status["str"]["success_required"], :
                ]
                self.error_table_str.visible = True

            if not validation_status["str"]["status_optional"]:
                self.warning_text_str.object += """\n
<font size='3'>String values must consist of `[A-Za-z0-9_-+.]`.</font>
<font size='3'>The following entries must be fixed.</font>
                """
                self.warning_table_str.value = df.loc[
                    ~validation_status["str"]["success_optional"], :
                ]
                self.warning_table_str.visible = True
        else:
            return

        # Stage 3 results
        if validation_status["str"]["status"]:
            if validation_status["values"]["status"] is None:
                pass
            elif validation_status["values"]["status"]:
                pass
            elif not validation_status["values"]["status"]:
                self.error_text_vals.object += """\n
<font size='3'>Errors in values are detected for the following entries.</font>
                """
                self.error_table_vals.value = df.loc[
                    ~validation_status["values"]["success"], :
                ]
                self.error_table_vals.visible = True
        else:
            return

        # Stage 4 results
        if validation_status["values"]["status"]:
            # logger.info(
            #     f"Status for duplicate: {validation_status['unique']['status']}"
            # )
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
                self.error_table_dups.value = df.loc[
                    validation_status["unique"]["flags"], :
                ]

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

                self.error_table_dups.editors = tabulator_editors
                self.error_table_dups.visible = True
        else:
            return

        if (
            validation_status["required_keys"]["status"]
            and validation_status["str"]["status"]
            and validation_status["values"]["status"]
            and validation_status["unique"]["status"]
        ):
            self.error_text_success.object += "\n<font size='3'>No error is found. Congratulations. You can proceed to the submission.</font>\n"
