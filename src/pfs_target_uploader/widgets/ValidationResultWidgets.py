#!/usr/bin/env python3

import numpy as np
import pandas as pd
import panel as pn


class ValidationResultWidgets:
    box_width = 1200

    stylesheet = """
    .tabulator-row-even { background-color: #f9f9f9 !important; }

    .tabulator-row-odd:hover {color: #000000!important; }
    .tabulator-row-even:hover {color: #000000!important;}
    """

    tabulator_kwargs = dict(
        page_size=50,
        theme="bootstrap",
        # theme="simple",
        # theme_classes=["table-striped", "table-sm"],
        frozen_columns=[],
        pagination="remote",
        header_filters=True,
        visible=False,
        layout="fit_data_table",
        disabled=True,
        max_width=box_width,
        stylesheets=[stylesheet],
        titles={"obj_id_str": "obj_id"},
        hidden_columns=["obj_id"],
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
            """<font size=5>✅ **Info**</font>\n\n
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

        self.error_table_str = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )
        self.warning_table_str = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )

        self.error_table_vals = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )
        self.warning_table_vals = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )

        self.error_table_flux = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )

        self.error_table_visibility = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )
        self.warning_table_visibility = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )

        self.error_table_dups = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )

        self.error_pane = pn.Column()
        self.warning_pane = pn.Column()
        self.info_pane = pn.Column()

        self.pane = pn.Column(
            self.title,
            self.error_pane,
            self.warning_pane,
            self.info_pane,
        )

        self.is_error = False
        self.is_warning = False
        self.is_info = False

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
            if t.value is not None:
                t.value[0:0]
            t.visible = False

        self.error_pane.objects = []
        self.warning_pane.objects = []
        self.info_pane.objects = []

        self.is_error = False
        self.is_warning = False
        self.is_info = False

    def append_title(self, status_str):
        if status_str == "error":
            if not self.is_error:
                self.error_pane.append(self.error_title)
                self.is_error = True
                self.error_title.visible = True
        if status_str == "warning":
            if not self.is_warning:
                self.warning_pane.append(self.warning_title)
                self.is_warning = True
                self.warning_title.visible = True
        if status_str == "info":
            if not self.is_info:
                self.info_pane.append(self.info_title)
                self.is_info = True
                self.info_title.visible = True

    def show_results(self, df, validation_status):
        # reset title panes
        self.reset()

        # df_orig = df.copy()  # create a back up just in case

        # df["obj_id_str"] = df["obj_id"].astype(str)
        df.insert(1, "obj_id_str", df["obj_id"].astype(str))

        if validation_status["status"]:
            self.error_title.visible = False

        # Errors on missing required keys
        if not validation_status["required_keys"]["status"]:
            self.append_title("error")
            self.error_text_keys.object = (
                "<font size=4><u>Missing required columns</u></font>\n"
            )
            for desc in validation_status["required_keys"]["desc_error"]:
                self.error_text_keys.object += f"- <font size='3'>{desc}</font>\n"
            self.error_pane.append(self.error_text_keys)

        # Warnings on missing optional keys
        if not validation_status["optional_keys"]["status"]:
            self.append_title("warning")
            self.warning_text_keys.object = (
                "<font size=4><u>Missing optional columns</u></font>\n"
            )
            for desc in validation_status["optional_keys"]["desc_warning"]:
                self.warning_text_keys.object += f"- <font size='3'>{desc}</font>\n"
            self.warning_pane.append(self.warning_text_keys)

        # Info on discovered keys
        n_req_success = len(validation_status["required_keys"]["desc_success"])
        n_opt_success = len(validation_status["optional_keys"]["desc_success"])
        if n_req_success + n_opt_success > 0:
            self.append_title("info")
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
            self.append_title("info")
            self.info_text_str.object = """<font size=4><u>String values</u></font>

<font size=3>All string values consist of `[A-Za-z0-9_-+.]` </font>"""
            self.info_pane.append(self.info_text_str)
        elif not validation_status["str"]["status"]:
            self.append_title("error")
            self.error_text_str.object = """<font size=4><u>Invalid characters in string values</u></font>

<font size=3>String values must consist of `[A-Za-z0-9_-+.]`. The following entries must be fixed.</font>"""

            is_invalid_str = np.logical_or(
                ~validation_status["str"]["success_required"],
                ~validation_status["str"]["success_optional"],
            )
            self.error_table_str.frozen_columns = []
            if self.error_table_str.value is not None:
                self.error_table_str.value[0:0]
            self.error_table_str.value = df.loc[is_invalid_str, :]
            self.error_table_str.frozen_columns = ["index"]
            self.error_pane.append(self.error_text_str)
            self.error_pane.append(self.error_table_str)
            self.error_table_str.visible = True

        # If string validation failed, return immediately
        if not validation_status["str"]["status"]:
            return

        # Data range
        if validation_status["values"]["status"] is None:
            pass
        elif validation_status["values"]["status"]:
            self.append_title("info")
            self.info_text_vals.object = """<font size=4><u>Data ranges</u></font>

<font size=3>All values of `ra`, `dec`, `priority`, `exptime`, and `resolution` satisfy the allowed ranges (see [documentation](doc/validation.html)).</font>
"""
            self.info_pane.append(self.info_text_vals)
        elif not validation_status["values"]["status"]:
            self.append_title("error")
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
            if self.error_table_vals.value is not None:
                self.error_table_vals.value[0:0]
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
            self.append_title("info")
            self.info_text_flux.object = "<font size=4><u>Flux information</u></font>\n\n<font size=3>All `ob_code`s have at least one flux information. The detected filters are the following: </font>"
            for f in validation_status["flux"]["filters"]:
                self.info_text_flux.object += f"<font size=3>`{f}`</font>, "
            self.info_text_flux.object = self.info_text_flux.object[:-2]

            self.info_pane.append(self.info_text_flux)
            self.error_table_flux.visible = False
        else:
            self.append_title("error")
            # add an error message and data table for duplicates
            self.error_text_flux.object = "<font size=4><u>Missing flux information</u></font>\n\n<font size=3>No flux information found in the following `ob_code`s. Detected filters are the following: </font>"
            for f in validation_status["flux"]["filters"]:
                self.error_text_flux.object += f"<font size=3>`{f}`</font>, "
            if len(validation_status["flux"]["filters"]) > 0:
                self.error_text_flux.object = self.error_text_flux.object[:-2]

            self.error_table_flux.frozen_columns = []
            if self.error_table_flux.value is not None:
                self.error_table_flux.value[0:0]
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
                self.append_title("info")
                self.info_text_visibility.object = "<font size=4><u>Visibility</u></font>\n\n<font size=3>All `ob_code`s are visible in the input observing period.</font>"
                self.info_pane.append(self.info_text_visibility)
            elif np.any(validation_status["visibility"]["success"]):
                self.append_title("warning")
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
                if self.warning_table_visibility.value is not None:
                    self.warning_table_visibility.value[0:0]
                self.warning_table_visibility.frozen_columns = []
                dfout = df.loc[~validation_status["visibility"]["success"], :]
                self.warning_table_visibility.value = dfout
                self.warning_table_visibility.frozen_columns = ["index"]
                self.warning_pane.append(self.warning_text_visibility)
                self.warning_pane.append(self.warning_table_visibility)
                self.warning_table_visibility.visible = True
            self.error_table_visibility.visible = False
        else:
            self.append_title("error")
            # add an error message and data table for duplicates
            self.error_text_visibility.object = "<font size=4><u>Visibility</u></font>\n\n<font size='3'>None of `ob_code`s in the list is visible in the input observing period.</font>"
            self.error_pane.append(self.error_text_visibility)

        # Duplication
        if validation_status["unique"]["status"]:
            self.append_title("info")
            self.info_text_dups.object = "<font size=4><u>Uniqueness of `ob_code`</u></font>\n\n<font size=3>All `ob_code` are unique.</font>"
            self.info_pane.append(self.info_text_dups)
            self.error_table_dups.visible = False
        else:
            self.append_title("error")
            # add an error message and data table for duplicates
            self.error_text_dups.object = "<font size=4><u>Duplication of `ob_code` and `obj_id` </u></font>\n\n<font size=3>Each `ob_code` and `obj_id` must be unique within a proposal, but duplicate `ob_code` and/or `obj_id` are detected in the following targets</font>"
            self.error_table_dups.frozen_columns = []
            if self.error_table_dups.value is not None:
                self.error_table_dups.value[0:0]
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
