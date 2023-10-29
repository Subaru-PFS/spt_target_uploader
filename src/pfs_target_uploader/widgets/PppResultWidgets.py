#!/usr/bin/env python3

import sys

import numpy as np
import panel as pn
from astropy import units as u
from astropy.table import Table
from bokeh.models.widgets.tables import NumberFormatter
from logzero import logger

from ..utils.io import upload_file
from ..utils.ppp import PPPrunStart, ppp_result


class PppResultWidgets:
    box_width = 1200

    # Maximum ROT can be requested for an openuse normal program
    max_reqtime_normal = 35.0

    def __init__(self):
        # PPP status
        # True if PPP has been run
        # False if PPP has not been run
        self.ppp_status = True
        self.origname = None
        self.origdata = None
        self.df_input = None
        self.df_summary = None

        self.ppp_title = pn.pane.Markdown(
            """# Results of PFS pointing simulation""",
            dedent=True,
            max_width=self.box_width,
        )
        self.ppp_warning_text = (
            "<font size=5>⚠️ **Warnings**</font>\n\n"
            "<font size=3>The total requested time exceeds 35 hours (maximum for a normal program). "
            "Please make sure to adjust it to your requirement before proceeding to the submission. "
            "Note that targets observable in the input observing period are considered.</font>"
        )

        self.ppp_success_text = (
            "<font size=5>✅ **Success**</font>\n\n"
            "<font size=3>The total requested time is reasonable for normal program. "
            "Note that targets observable in the input period are considered.</font>"
        )

        self.ppp_figure = pn.Column()

        self.ppp_alert = pn.Column()

        self.pane = pn.Column(
            self.ppp_title,
            self.ppp_figure,
            max_width=self.box_width,
        )

    def reset(self):
        self.ppp_figure.clear()
        self.ppp_figure.visible = False
        self.ppp_status = False
        self.df_input = None
        self.df_summary = None
        self.origname = None
        self.origdata = None

    def show_results(self):
        logger.info("showing PPP results")

        print(self.df_summary)

        def update_alert(df):
            rot = np.ceil(df.iloc[-1]["Request time (h)"] * 10.0) / 10.0
            if rot > self.max_reqtime_normal:
                text = self.ppp_warning_text
                type = "warning"
            else:
                text = self.ppp_success_text
                type = "success"
            return {"object": text, "alert_type": type}

        def update_reqtime(df):
            rot = np.ceil(df.iloc[-1]["Request time (h)"] * 10.0) / 10.0
            if rot > self.max_reqtime_normal:
                c = "crimson"
            else:
                # c = "#007b43"
                c = "#3A7D7E"
            return {"value": rot, "default_color": c}

        def update_summary_text(df):
            rot = np.ceil(df.iloc[-1]["Request time (h)"] * 10.0) / 10.0
            n_ppc = df.iloc[-1]["N_ppc"]
            t_exp = df.iloc[-1]["Texp (h)"]
            t_fh = df.iloc[-1]["Texp (fiberhour)"]
            try:
                comp_all_low = df.iloc[0]["P_all"]
                text_comp_low = f"- <font size=3>The expected **completion rate** for **low-resolution** mode is **{comp_all_low:.0f}%**.</font>\n"
            except Exception:
                comp_all_low = None
                text_comp_low = ""
            try:
                comp_all_med = df.iloc[1]["P_all"]
                text_comp_med = f"- <font size=3>The expected **completion rate** for **medium-resolution** mode is **{comp_all_med:.0f}%**.</font>"
            except Exception:
                comp_all_med = None
                text_comp_med = ""
            text = (
                f"- <font size=3>You have requested **{int(n_ppc)}** **PFS pointing centers (PPCs)**.</font>\n"
                f"- <font size=3>The optimized PPCs correspond to **{t_fh:.1f} fiber hours**.</font>\n"
                f"- <font size=3>The **exposure time** to complete {int(n_ppc)} PPCs (without overhead) is **{t_exp:.1f} hours** ({int(n_ppc)} x 15 minutes).</font>\n"
                f"- <font size=3>The **requested observing time (ROT)** including overhead is **{rot:.1f} hours**.</font>\n"
                f"{text_comp_low}"
                f"{text_comp_med}"
            )
            return {"object": text}

        def stream_export_files(df_psl, df_ppc, p_fig):
            _, outfile_zip, sio = upload_file(
                self.df_input,
                df_psl,
                df_ppc,
                self.df_summary,
                p_fig,
                origname=self.origname,
                origdata=self.origdata,
                export=True,
            )
            return sio

        # A number indicator showing the current total ROT
        self.reqtime = pn.indicators.Number(
            name="Your total request is",
            format="{value:.1f} <font size=18>h</font>",
            max_width=300,
            refs=pn.bind(update_reqtime, self.p_result_tab),
        )

        # alert panel is bind to the total request
        self.ppp_alert = pn.pane.Alert(
            refs=pn.bind(update_alert, self.p_result_tab),
            max_width=self.box_width,
            height=150,
        )

        # summary text
        self.summary_text = pn.pane.Markdown(
            refs=pn.bind(update_summary_text, self.p_result_tab),
            max_width=self.box_width,
        )

        # set export files
        stylesheet = """
        .bk-btn a {
            border-color: #3A7D7E !important;
            text-align: left !important;
        }"""
        self.export_button = pn.widgets.FileDownload(
            name="Export the results",
            callback=pn.bind(
                stream_export_files,
                self.p_result_tab.value,
                self.p_result_ppc.value,
                self.p_result_fig,
            ),
            filename="pfs_target.zip",
            button_style="outline",
            button_type="primary",
            icon="download",
            icon_size="1.5em",
            label="",
            max_width=150,
            stylesheets=[stylesheet],
        )

        # add styling/formatting to the table
        self.p_result_tab.formatters = {
            "N_ppc": NumberFormatter(format="0", text_align="right"),
            "Texp (h)": NumberFormatter(format="0.00", text_align="right"),
            "Texp (fiberhour)": NumberFormatter(format="0.00", text_align="right"),
            "Request time (h)": NumberFormatter(format="0.00", text_align="right"),
            "Used fiber fraction (%)": NumberFormatter(
                format="0.000", text_align="right"
            ),
            "Fraction of PPC < 30% (%)": NumberFormatter(
                format="0.0", text_align="right"
            ),
        }
        for p in ["all", np.arange(10)]:
            self.p_result_tab.formatters[f"P_{p}"] = NumberFormatter(
                format="0.0", text_align="left"
            )

        # compose the pane
        self.ppp_figure.append(self.ppp_alert)
        self.ppp_figure.append(pn.Row(self.reqtime, self.summary_text))
        self.ppp_figure.append(
            pn.Column(
                "<font size=4><u>Number of PFS pointing centers (adjustable with the sliders)</u></font>",
                pn.Row(self.export_button, self.nppc),
                self.p_result_tab,
            )
        )
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

        self.df_input = df

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
            cR_L_,
            sub_l,
            obj_allo_L_fin,
            uS_L2,
            cR_M_,
            sub_m,
            obj_allo_M_fin,
            uS_M2,
            box_width=self.box_width,
        )

        self.ppp_status = True
