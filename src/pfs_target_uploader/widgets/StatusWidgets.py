#!/usr/bin/env python3

import numpy as np
import pandas as pd
import panel as pn
from logzero import logger


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
