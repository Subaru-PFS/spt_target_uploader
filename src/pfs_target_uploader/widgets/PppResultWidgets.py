#!/usr/bin/env python3

import multiprocessing as mp
import sys

import numpy as np
import panel as pn
from astropy import units as u
from astropy.table import Table
from loguru import logger

from ..utils.io import upload_file
from ..utils.ppp import PPPrunStart, ppp_result


class PppResultWidgets:
    box_width = 1200

    # Maximum ROT can be requested for an openuse normal program
    max_reqtime_normal = 35.0

    def __init__(
        self,
    ):
        # PPP status
        # True if PPP has been run
        # False if PPP has not been run
        self.ppp_status = True
        self.df_input = None
        self.df_summary = None
        self.origname = None
        self.origname_ppc = None
        self.origdata = None
        self.origdata_ppc = None
        self.upload_time = None
        self.secret_token = None
        self.status_ = 0

        self.ppp_title = pn.pane.Markdown(
            """# Results of PFS pointing simulation""",
            dedent=True,
            max_width=self.box_width,
        )
        self.ppp_warning_text_1 = (
            "<font size=5>⚠️ **Warnings**</font>\n\n"
            f"<font size=3>The total requested time exceeds {int(self.max_reqtime_normal)} hours (maximum for a normal program). "
            "Please make sure to adjust it to your requirement before proceeding to the submission. "
            "Note that targets observable in the input observing period are considered.</font>"
        )

        self.ppp_error_text_1 = (
            "<font size=5>⚠️ **Error**</font>\n\n"
            "<font size=3>No fiber can be assigned. Please check the input pointing list, or the single exposure time.</font>"
        )

        self.ppp_error_text_2 = (
            "<font size=5>⚠️ **Error**</font>\n\n"
            "<font size=3>Calculation stops because time is running out. "
            "If you would get the complete outputs, please modify the input list or consult with the observatory. </font>"
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
        self.origname_ppc = None
        self.origdata = None
        self.origdata_ppc = None
        self.upload_time = None
        self.secret_token = None
        self.status_ = 0

    def show_results(self):
        logger.info("showing PPP results")

        # print(self.df_summary)

        # @pn.io.profile("update_alert")
        def update_alert(df):
            if df is None:
                rot = 0
            else:
                rot = np.ceil(df.iloc[-1]["Request time (h)"] * 10.0) / 10.0

            if self.status_ == 999 and rot > self.max_reqtime_normal:
                text = self.ppp_warning_text_1
                type = "warning"
            elif self.status_ == 1:
                text = self.ppp_error_text_2
                type = "danger"
            elif self.status_ == 2 and df is None:
                text = self.ppp_error_text_1
                type = "danger"
            else:
                text = self.ppp_success_text
                type = "success"
            return {"object": text, "alert_type": type}

        # @pn.io.profile("update_reqtime")
        def update_reqtime(df):
            if df is None:
                return {"value": 0, "default_color": "#3A7D7E"}

            rot = np.ceil(df.iloc[-1]["Request time (h)"] * 10.0) / 10.0
            if rot > self.max_reqtime_normal:
                c = "crimson"
            else:
                # c = "#007b43"
                c = "#3A7D7E"
            return {"value": rot, "default_color": c}

        # @pn.io.profile("update_summary_text")
        def update_summary_text(df):
            if df is None:
                return {"object": " "}

            rot = np.ceil(df.iloc[-1]["Request time (h)"] * 10.0) / 10.0
            n_ppc = df.iloc[-1]["N_ppc"]
            t_exp = df.iloc[-1]["Texp (h)"]
            t_fh = df.iloc[-1]["Texp (fiberhour)"]

            text_comp_low, text_comp_med = "", ""

            for i in range(2):
                try:
                    res = df.iloc[i]["resolution"]
                    comp_all_ = df.iloc[i]["P_all"]
                    text_comp_ = (
                        "- <font size=3>The expected **completion rate** "
                        f"for **{res}-resolution** mode is **{comp_all_:.0f}%**.</font>\n"
                    )
                    if res == "low":
                        text_comp_low = text_comp_
                    elif res == "medium":
                        text_comp_med = text_comp_
                except Exception:
                    continue

            text = (
                f"- <font size=3>You have requested **{int(n_ppc)}** **PFS pointing centers (PPCs)**.</font>\n"
                f"- <font size=3>The optimized PPCs correspond to **{t_fh:.1f} fiber hours**.</font>\n"
                f"- <font size=3>The **exposure time** to complete {int(n_ppc)} PPCs (without overhead) is **{t_exp:.1f} hours** "
                f"({int(n_ppc)} x {self.single_exptime/60:.2f} minutes).</font>\n"
                f"- <font size=3>The **requested observing time (ROT)** including overhead is **{rot:.1f} hours**.</font>\n"
                f"{text_comp_low}"
                f"{text_comp_med}"
            )
            return {"object": text}

        # @pn.io.profile("stream_export_files")
        def stream_export_files(df_psl, df_ppc, p_fig):
            _, outfile_zip, sio = upload_file(
                self.df_input,
                df_psl,
                df_ppc,
                self.df_summary,
                p_fig,
                origname=self.origname,
                origname_ppc=self.origname_ppc,
                origdata=self.origdata,
                origdata_ppc=self.origdata_ppc,
                export=True,
            )

            self.export_button.filename = outfile_zip
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
        .bk-btn {
            // border-color: #3A7D7E !important;
            // border-color: var(--success-border-subtle) !important;
            display: inline !important;
            // vertical-align: bottom !important;
        }
        .bk-btn a {
            vertical-align: middle !important;
            text-align: left !important;
            padding-left: 0px !important;
            padding-right: 12px !important;
        }
        """

        # compose the pane
        self.ppp_figure.append(self.ppp_alert)
        self.ppp_figure.append(pn.Row(self.reqtime, self.summary_text))

        if self.p_result_tab is not None:
            self.export_button = pn.widgets.FileDownload(
                name="Export the results",
                callback=pn.bind(
                    stream_export_files,
                    self.p_result_tab.value,
                    self.p_result_ppc.value,
                    self.p_result_fig,
                ),
                # filename="pfs_target.zip",
                filename="",
                button_style="outline",
                button_type="primary",
                icon="download",
                icon_size="1.2em",
                label="",
                max_width=150,
                height=60,
                stylesheets=[stylesheet],
            )

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

    def run_ppp(
        self,
        df,
        df_ppc,
        validation_status,
        single_exptime=900,
        weights=None,
        clustering_algorithm="HDBSCAN",
        quiet=True,
        max_exetime=900,
        logger=None,
    ):
        if logger is None:
            logger.remove()
            logger.add(sys.stderr, level="INFO", enqueue=True)

        if weights is None:
            weights = [2.02, 0.01, 0.01]

        self.df_input = df
        self.df_input["single_exptime"] = single_exptime

        self.single_exptime = single_exptime

        tb_input = Table.from_pandas(df)
        tb_visible = tb_input[validation_status["visibility"]["success"]]

        if len(df_ppc) > 0:
            tb_ppc = Table.from_pandas(df_ppc)
            logger.info(f"PPCs are input by users: {tb_ppc}")
        else:
            tb_ppc = []

        ppp_run_results = mp.Manager().Queue()

        ppp_run = mp.Process(
            target=PPPrunStart,
            name="PPP",
            args=(
                tb_visible,
                tb_ppc,
                weights,
                self.single_exptime,
                1.38,
                quiet,
                clustering_algorithm,
                max_exetime,
                ppp_run_results,
                logger,
            ),
        )

        # start run PPP
        ppp_run.start()

        # Wait max_exetime for PPP
        ppp_run.join(max_exetime)

        if ppp_run.is_alive():
            # if ppp is still running after max_exetime, kill it
            logger.error("Pointing simulation failed (runout time)")
            pn.state.notifications.error(
                f"Simulation stops because time ({int(max_exetime):d} sec) is running out.",
                duration=0,  # ever
            )

            # Terminate PPP
            ppp_run.terminate()

            # Cleanup
            ppp_run.join()

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
                self.status_,
            ) = (
                Table(),
                [],
                [],
                [],
                Table(),
                Table(),
                [],
                [],
                [],
                Table(),
                1,
            )  # ppc_status=1 in case of runout time
        else:
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
                self.status_,
            ) = ppp_run_results.get()

        (
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
            single_exptime=self.single_exptime,
            box_width=self.box_width,
        )

        self.ppp_status = True

    def upload(
        self,
        outdir_prefix=".",
        export=False,
        single_exptime=None,
        observation_type="queue",
        ppc_status="auto",
    ):
        if single_exptime is None:
            single_exptime = self.single_exptime

        self.df_input["single_exptime"] = single_exptime

        try:
            df_psl = self.p_result_tab.value
            df_ppc = self.p_result_ppc.value
            ppp_fig = self.p_result_fig
        except AttributeError:
            df_psl = None
            df_ppc = None
            ppp_fig = None

        outdir, outfile_zip, _ = upload_file(
            self.df_input,
            df_psl,
            df_ppc,
            self.df_summary,
            ppp_fig,
            outdir_prefix=outdir_prefix,
            origname=self.origname,
            origname_ppc=self.origname_ppc,
            origdata=self.origdata,
            origdata_ppc=self.origdata_ppc,
            secret_token=self.secret_token,
            upload_time=self.upload_time,
            ppp_status=self.ppp_status,
            single_exptime=single_exptime,
            observation_type=observation_type,
            ppc_status=ppc_status,
        )

        return outdir, outfile_zip, None
