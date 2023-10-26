#!/usr/bin/env python3

import numpy as np
import pandas as pd
import panel as pn
from logzero import logger


class StatusWidgets:
    def __init__(self, size=20):
        self.status = pn.pane.Alert(
            "<font size=4>Waiting...</font>", alert_type="secondary", height=80
        )
        # self.status.visible = False

        stylesheet = """
            .tabulator-row-odd { background-color: #ffffff !important; }
            .tabulator-row-even { background-color: #ffffff !important; }
            .tabulator-row-odd:hover { color: #000000 !important; background-color: #ffffff !important; }
            .tabulator-row-even:hover { color: #000000 !important; background-color: #ffffff !important; }
            """

        n_priority = 10

        self.df_summary_init = pd.DataFrame(
            {
                "Priority": np.arange(0, n_priority, 1, dtype=object),
                "N (L)": np.zeros(n_priority, dtype=int),
                "Texp (L)": np.zeros(n_priority, dtype=float),
                "N (M)": np.zeros(n_priority, dtype=int),
                "Texp (M)": np.zeros(n_priority, dtype=float),
            }
        )

        self.df_summary_init.loc[n_priority] = [
            "Other",
            0,
            0,
            0,
            0,
        ]

        self.df_summary_init.loc[n_priority + 1] = [
            "Total",
            0,
            0,
            0,
            0,
        ]

        self.summary_table = pn.widgets.Tabulator(
            self.df_summary_init,
            theme="bootstrap",
            # theme_classes=["table-sm"],
            # visible=False,
            layout="fit_data_stretch",
            # layout="fit_data_table",
            hidden_columns=["index"],
            selectable=False,
            width=350,
            header_align="right",
            configuration={"columnDefaults": {"headerSort": False}},
            disabled=True,
            stylesheets=[stylesheet],
            # min_width=480,
        )

        self.table_footnote = pn.pane.Markdown(
            "- <font size=2>`N` is the number of `ob_code`s for each priority.</font>\n"
            "- <font size=2>`T` is the total fiberhours of `ob_code`s for each priority.</font>\n"
            "- <font size=2>`L` and `M` correspond to the low- and medium-resolution modes, respectively.</font>",
        )
        # self.table_footnote.visible = False

        self.pane = pn.Column(self.status, self.summary_table, self.table_footnote)

    def reset(self):
        self.status.alert_type = "light"
        # self.status.visible = False

        self.summary_table.value = self.df_summary_init
        # self.summary_table.visible = False
        # self.table_footnote.visible = False

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

        # self.status.visible = True

        try:
            unique_priority = np.arange(0, 10, 1, dtype=object)
            number_priority_L = np.zeros_like(unique_priority, dtype=int)
            number_priority_M = np.zeros_like(unique_priority, dtype=int)
            exptime_priority_L = np.zeros_like(unique_priority, dtype=float)
            exptime_priority_M = np.zeros_like(unique_priority, dtype=float)

            idx_l = df["resolution"] == "L"
            idx_m = df["resolution"] == "M"

            s_to_h = 1.0 / 3600.0

            for i, p in enumerate(unique_priority):
                idx_p = df["priority"] == p
                number_priority_L[i] = df.loc[idx_p & idx_l, :].index.size
                number_priority_M[i] = df.loc[idx_p & idx_m, :].index.size
                exptime_priority_L[i] = df.loc[idx_p & idx_l, "exptime"].sum()
                exptime_priority_M[i] = df.loc[idx_p & idx_m, "exptime"].sum()

            self.df_summary = pd.DataFrame(
                {
                    "Priority": unique_priority,
                    "N (L)": number_priority_L,
                    "Texp (L)": exptime_priority_L * s_to_h,
                    "N (M)": number_priority_M,
                    "Texp (M)": exptime_priority_M * s_to_h,
                }
            )

            self.df_summary.loc[len(self.df_summary.index)] = [
                "Other",
                df.loc[idx_l, :].index.size - sum(number_priority_L),
                (df.loc[idx_l, "exptime"].sum() - sum(exptime_priority_L)) * s_to_h,
                df.loc[idx_m, :].index.size - sum(number_priority_M),
                (df.loc[idx_m, "exptime"].sum() - sum(exptime_priority_M)) * s_to_h,
            ]

            self.df_summary.loc[len(self.df_summary.index)] = [
                "Total",
                df.loc[idx_l, :].index.size,
                df.loc[idx_l, "exptime"].sum() * s_to_h,
                df.loc[idx_m, :].index.size,
                df.loc[idx_m, "exptime"].sum() * s_to_h,
            ]

            logger.info(f"Summary Table:\n{self.df_summary}")

            self.summary_table.value = pd.DataFrame()
            self.summary_table.value = self.df_summary
            # self.summary_table.visible = True
            # self.table_footnote.visible = True

        except Exception as e:
            logger.warning(
                f"failed to show the summary table in the side bar: {e=}, {type(e)=}"
            )
