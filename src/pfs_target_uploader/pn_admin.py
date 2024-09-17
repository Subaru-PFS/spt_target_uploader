#!/usr/bin/env python3

import glob
import os
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import panel as pn
from astropy.table import Table
from dotenv import dotenv_values
from loguru import logger

from .utils.io import load_file_properties, load_input
from .utils.ppp import ppp_result_reproduce
from .widgets import TargetWidgets


#
# admin app
#
def list_files_app(use_panel_cli=False):
    pn.state.notifications.position = "bottom-left"

    config = dotenv_values(".env.shared")

    logger.info(f"config params from dotenv: {config}")

    panel_targets = TargetWidgets()

    if not os.path.exists(config["OUTPUT_DIR"]):
        logger.error(f"{config['OUTPUT_DIR']} not found")
        raise ValueError

    template = pn.template.VanillaTemplate(
        title="PFS Target & Proposal Lists",
        # collapsed_sidebar=True,
        # header_background="#3A7D7E",
        # header_background="#C71585",  # mediumvioletred
        header_background="#dc143c",  # crimson
        busy_indicator=None,
        favicon="doc/assets/images/favicon.png",
        # sidebar_width=400,
    )

    df_files_tgt_psl = load_file_properties(
        config["OUTPUT_DIR"],
        ext="ecsv",
    )

    psl_info_input = pn.widgets.FileInput(
        value=None,
        filename=None,
        accept=".csv",
        multiple=False,
        height=40,
    )

    psl_info = pn.Column(
        pn.pane.Markdown(
            "<font size=4><span style='color:blue'>[optional]</span> Upload the proposal info:</font>"
            "<font size=4>(<a href='doc/examples/example_admin_pslID.csv' target='_blank'>example</a>)</font>",
        ),
        psl_info_input,
        height=200,
    )

    # range sliders for filtering
    slider_nobj = pn.widgets.EditableRangeSlider(
        name="N (ob_code)",
        start=np.floor(df_files_tgt_psl["n_obj"].min() / 10) * 10,
        end=np.ceil(df_files_tgt_psl["n_obj"].max() / 10) * 10,
        step=1,
        width=350,
    )
    slider_fiberhour = pn.widgets.EditableRangeSlider(
        name="Fiberhour (h)",
        start=np.floor(df_files_tgt_psl["Exptime_tgt (FH)"].min()),
        end=np.ceil(df_files_tgt_psl["Exptime_tgt (FH)"].max()),
        step=1,
        width=350,
    )

    slider_rot_l = pn.widgets.EditableRangeSlider(
        name="ROT_low (h)",
        start=np.floor(df_files_tgt_psl["Time_tot_L (h)"].min()),
        end=np.ceil(df_files_tgt_psl["Time_tot_L (h)"].max()),
        step=1,
        width=350,
    )
    slider_rot_m = pn.widgets.EditableRangeSlider(
        name="ROT_med (h)",
        start=np.floor(df_files_tgt_psl["Time_tot_M (h)"].min()),
        end=np.ceil(df_files_tgt_psl["Time_tot_M (h)"].max()),
        step=1,
        width=350,
    )

    # Target & psl summary table
    def Table_files_tgt_psl(column_checkbox_):
        if psl_info_input.value is not None:
            df_psl_info = load_input(
                BytesIO(psl_info_input.value),
                format="csv",
            )[0]

            _df_files_tgt_psl = pd.merge(
                df_files_tgt_psl, df_psl_info, left_on="Upload ID", right_on="Upload ID"
            )

        else:
            _df_files_tgt_psl = df_files_tgt_psl

        _hidden_columns = list(
            set(list(_df_files_tgt_psl.columns)) - set(column_checkbox_)
        ) + ["index"]

        _table_files_tgt_psl = pn.widgets.Tabulator(
            _df_files_tgt_psl,
            page_size=500,
            theme="bootstrap",
            # theme_classes=["table-striped", "table-sm"],
            theme_classes=["table-striped"],
            frozen_columns=["index"],
            pagination="remote",
            header_filters=True,
            buttons={
                "magnify": "<i class='fa-solid fa-magnifying-glass'></i>",
                "download": "<i class='fa-solid fa-download'></i>",
            },
            layout="fit_data_table",
            hidden_columns=_hidden_columns,
            disabled=True,
            selection=[],
            selectable="checkbox",
            formatters={"TAC_done": {"type": "tickCross"}},
            text_align={
                "TAC_done": "center",
                "observation_type": "center",
                "pointing_status": "center",
            },
        )

        """
        dirs = glob.glob(os.path.join(config["OUTPUT_DIR"], "????/??/*/*"))
        upload_id_tacFin = [
            tt[tt.find("TAC_psl_") + 8 : tt.rfind(".ecsv")]
            for tt in dirs
            if "TAC_psl_" in tt
        ]
        row_tacFin = np.where(
            np.in1d(_table_files_tgt_psl.value["Upload ID"], upload_id_tacFin) == True
        )[0]
        _table_files_tgt_psl.selection = [int(tt) for tt in row_tacFin]
        #"""

        _table_files_tgt_psl.add_filter(slider_nobj, "n_obj")
        _table_files_tgt_psl.add_filter(slider_fiberhour, "t_exp")
        _table_files_tgt_psl.add_filter(slider_rot_l, "Time_tot_L (h)")
        _table_files_tgt_psl.add_filter(slider_rot_m, "Time_tot_M (h)")

        # Open a file by clicking the download buttons
        # https://discourse.holoviz.org/t/how-to-make-a-dynamic-link-in-panel/2137

        def execute_javascript(script):
            script = f'<script type="text/javascript">{script}</script>'
            js_panel.object = script
            js_panel.object = ""

        def open_panel_download(event):
            if event.column == "download":
                href_tgt = df_files_tgt_psl["fullpath_tgt"][event.row]
                href_ppc = df_files_tgt_psl["fullpath_ppc"][event.row]
                # need to fix the path for the download
                href_mod_tgt = href_tgt.replace(config["OUTPUT_DIR"], "data", 1)
                href_mod_ppc = href_ppc.replace(config["OUTPUT_DIR"], "data", 1)
                logger.info(f"{href_tgt=}")
                logger.info(f"{href_mod_tgt=}")
                # c.f. https://www.w3schools.com/jsref/met_win_open.asp
                script_tgt = f"window.open('{href_mod_tgt}', '_blank')"
                script_ppc = f"window.open('{href_mod_ppc}', '_blank')"
                execute_javascript(script_tgt)
                execute_javascript(script_ppc)

        def zip_select():
            if download_group.value == "Target":
                column_ = "fullpath_tgt"
                prefix_ = "target"
            elif download_group.value == "PPC":
                column_ = "fullpath_ppc"
                prefix_ = "ppc"
            elif download_group.value == "PPC (after allocation)":
                column_ = "fullpath_ppc_tac"
                prefix_ = "ppc_tac"

            row_select = _table_files_tgt_psl.selection

            if len(row_select) > 0:
                tmpdir = os.path.join(config["OUTPUT_DIR"], "tmp")
                filepath_zip = os.path.join(tmpdir, f"{prefix_}_selected.zip")
                filepath_zip_href = os.path.join(
                    tmpdir.replace(config["OUTPUT_DIR"], "data/", 1),
                    f"{prefix_}_selected.zip",
                ).replace("//", "/")
                # print(f"{tmpdir=}")
                # print(f"{filepath_zip=}")
                # print(f"{filepath_zip_href=}")

                if not os.path.exists(tmpdir):
                    logger.info(f"{tmpdir} not found. Creating...")
                    os.makedirs(tmpdir)

                if os.path.exists(filepath_zip):
                    logger.info(f"{filepath_zip} already exists. Removing...")
                    os.remove(filepath_zip)

                with ZipFile(filepath_zip, "w") as zipfile:
                    for filepath_ in _table_files_tgt_psl.value[column_][row_select]:
                        if filepath_ is not None:
                            zipfile.write(filepath_, os.path.basename(filepath_))
                zipfile.close()
                logger.info(f"Zipfile saved under {filepath_zip}")
            else:
                filepath_zip = None
                filepath_zip_href = None
            return filepath_zip_href

        def download_select(event):
            filepath_zip = zip_select()
            if filepath_zip is None:
                pn.state.notifications.error(
                    "Can not download due to no selected program.",
                    duration=5000,
                )
            else:
                script_list_select = f"window.open('{filepath_zip}')"
                execute_javascript(script_list_select)
                logger.info(f"{filepath_zip} downloaded")

        def open_panel_magnify(event):
            row_target = event.row
            if event.column == "magnify":
                table_ppc.clear()

                # move to "PPC details" tab
                tab_panels.active = 2

                u_id = _df_files_tgt_psl["Upload ID"][row_target]
                p_ppc = os.path.split(_df_files_tgt_psl["fullpath_psl"][row_target])[0]
                try:
                    psl_id = _df_files_tgt_psl["proposal ID"][row_target]
                except KeyError:
                    psl_id = None

                table_ppc_t = Table.read(os.path.join(p_ppc, f"ppc_{u_id}.ecsv"))
                table_tgt_t = Table.read(os.path.join(p_ppc, f"target_{u_id}.ecsv"))
                table_psl_t = Table.read(os.path.join(p_ppc, f"psl_{u_id}.ecsv"))
                try:
                    table_tac_t = Table.read(
                        os.path.join(p_ppc, f"TAC_psl_{u_id}.ecsv")
                    )
                except FileNotFoundError:
                    table_tac_t = Table()

                panel_targets.show_results(Table.to_pandas(table_tgt_t))

                (
                    nppc_fin,
                    p_result_fig_fin,
                    p_result_ppc_fin,
                    p_result_tab,
                ) = ppp_result_reproduce(
                    table_ppc_t, table_tgt_t, table_psl_t, table_tac_t
                )

                dirs2 = glob.glob(os.path.join(config["OUTPUT_DIR"], "????/??/*/"))
                path_t_all = [tt for tt in dirs2 if u_id in tt]
                if len(path_t_all) == 0:
                    logger.error(f"Path not found for {u_id}")
                    raise ValueError
                elif len(path_t_all) > 1:
                    logger.error(
                        f"Multiple paths found for {u_id}, {path_t_all}, len={len(path_t_all)}"
                    )
                    raise ValueError

                path_t_server = path_t_all[0]

                path_t = path_t_server.replace(config["OUTPUT_DIR"], "data", 1)
                tac_ppc_list_file = f"{path_t}/TAC_ppc_{u_id}.ecsv"

                logger.info(f"{row_target=}")
                if _df_files_tgt_psl["TAC_done"][row_target]:
                    # make the ppc list downloadable
                    fd_link = pn.pane.Markdown(
                        f"<font size=4>(<a href={tac_ppc_list_file} target='_blank'>Download the allocated PPC list</a>)</font>",
                        margin=(0, 0, 0, -40),
                    )
                    logger.info("TAC PPC list is already available.")

                def tab_ppc_save(event):
                    # save tac allocation (TAC_psl/ppc_uploadid.ecsv)
                    # dirs = glob.glob(os.path.join(config["OUTPUT_DIR"], "????/??/*"))

                    Table.from_pandas(p_result_ppc_fin.value).write(
                        f"{path_t_server}/TAC_ppc_{u_id}.ecsv",
                        format="ascii.ecsv",
                        delimiter=",",
                        overwrite=True,
                    )
                    logger.info(
                        f"File TAC_ppc_{u_id}.ecsv is saved under {path_t_server}."
                    )
                    # make the ppc list downloadable
                    fd_link.object = f"<font size=4>(<a href={tac_ppc_list_file} target='_blank'>Download the allocated PPC list</a>)</font>"

                    tb_tac_psl_t = Table.from_pandas(p_result_tab.value)
                    tb_tac_psl_t.write(
                        f"{path_t_server}/TAC_psl_{u_id}.ecsv",
                        format="ascii.ecsv",
                        delimiter=",",
                        overwrite=True,
                    )
                    logger.info(
                        f"File TAC_psl_{u_id}.ecsv is saved under {path_t_server}."
                    )

                    # update tac allocation in program info tab
                    if sum(tb_tac_psl_t["resolution"] == "low") > 0:
                        _df_files_tgt_psl["TAC_done"][row_target] = True
                        _df_files_tgt_psl["TAC_FH_L"][row_target] = tb_tac_psl_t[
                            "Texp (fiberhour)"
                        ][tb_tac_psl_t["resolution"] == "low"]
                        _df_files_tgt_psl["TAC_nppc_L"][row_target] = tb_tac_psl_t[
                            "N_ppc"
                        ][tb_tac_psl_t["resolution"] == "low"]
                        _df_files_tgt_psl["TAC_ROT_L"][row_target] = tb_tac_psl_t[
                            "Request time (h)"
                        ][tb_tac_psl_t["resolution"] == "low"]

                    if sum(tb_tac_psl_t["resolution"] == "medium") > 0:
                        _df_files_tgt_psl["TAC_done"][row_target] = True
                        _df_files_tgt_psl["TAC_FH_M"][row_target] = tb_tac_psl_t[
                            "Texp (fiberhour)"
                        ][tb_tac_psl_t["resolution"] == "medium"]
                        _df_files_tgt_psl["TAC_nppc_M"][row_target] = tb_tac_psl_t[
                            "N_ppc"
                        ][tb_tac_psl_t["resolution"] == "medium"]
                        _df_files_tgt_psl["TAC_ROT_M"][row_target] = tb_tac_psl_t[
                            "Request time (h)"
                        ][tb_tac_psl_t["resolution"] == "medium"]

                    _table_files_tgt_psl.value = _df_files_tgt_psl

                    # update tac allocation summary
                    tac_summary.object = (
                        "<font size=5>Summary of TAC allocation:</font>\n"
                        f"<font size=4> - N (allocated programs): <span style='color:tomato'>**{sum(_df_files_tgt_psl['TAC_done']):.0f}**</span></font>\n"
                        "<font size=4> - Low-res mode: </font>\n"
                        f"<font size=4>&emsp;**FH** allocated = <span style='color:tomato'>**{sum(_df_files_tgt_psl['TAC_FH_L']):.2f}**</span></font>\n"
                        f"<font size=4>&emsp;**Nppc** allocated = <span style='color:tomato'>**{sum(_df_files_tgt_psl['TAC_nppc_L']):.0f}**</span> </font>\n"
                        f"<font size=4>&emsp;**ROT** (h) allocated = <span style='color:tomato'>**{sum(_df_files_tgt_psl['TAC_ROT_L']):.2f}**</span> </font>\n"
                        "<font size=4> - Medium-res mode: </font>\n"
                        f"<font size=4>&emsp;**FH** allocated = <span style='color:tomato'>**{sum(_df_files_tgt_psl['TAC_FH_M']):.2f}**</span></font>\n"
                        f"<font size=4>&emsp;**Nppc** allocated = <span style='color:tomato'>**{sum(_df_files_tgt_psl['TAC_nppc_M']):.0f}**</span> </font>\n"
                        f"<font size=4>&emsp;**ROT** (h) allocated = <span style='color:tomato'>**{sum(_df_files_tgt_psl['TAC_ROT_M']):.2f}**</span> </font>\n"
                    )
                    pn.state.notifications.info(
                        "TAC allocation is made for the program and a new PPC list is saved.",
                        duration=5000,  # 5sec
                    )

                    # move to "Program info" tab
                    # tab_panels.active = 0

                if nppc_fin is not None:
                    output_status = pn.pane.Markdown(
                        "<font size=5>You are checking the program:</font>\n"
                        f"<font size=4> Upload id = <span style='color:tomato'>**{u_id}**</span></font>\n"
                        f"<font size=4> Proposal id = <span style='color:tomato'>**{psl_id}**</span> </font>",
                    )

                    fd_success = pn.widgets.Button(
                        name="Time allocated",
                        button_type="primary",
                        icon="circle-check",
                        icon_size="2em",
                        height=45,
                        max_width=150,
                        margin=(20, 0, 0, 0),
                    )

                    if not _df_files_tgt_psl["TAC_done"][row_target]:
                        fd_link = pn.pane.Markdown(
                            "<font size=4>(Download the allocated PPC list)</font>",
                            margin=(0, 0, 0, -40),
                        )

                    fd_success.on_click(tab_ppc_save)

                    table_ppc.append(
                        pn.Row(output_status, pn.Column(fd_success, fd_link), width=750)
                    )

                else:
                    ####NEED to FIX!!
                    # Do we need this? since 15-min upper limit is set in online PPP, all programs should have some ppp outputs..?
                    output_status = pn.pane.Markdown(
                        "<font size=5>You are checking the program (no PPP outputs):</font>\n"
                        f"<font size=4> Upload id = <span style='color:tomato'>**{u_id}**</span></font>\n"
                        f"<font size=4> Proposal id = <span style='color:tomato'>**{psl_id}**</span> </font>",
                    )

                    table_ppc.append(pn.Row(output_status, width=750))

                table_ppc.append(
                    pn.Row(
                        pn.Column(p_result_ppc_fin, width=700, height=1000),
                        pn.Column(nppc_fin, p_result_tab, p_result_fig_fin),
                    )
                )

        _table_files_tgt_psl.on_click(open_panel_magnify)
        _table_files_tgt_psl.on_click(open_panel_download)
        download_selection.on_click(download_select)

        return _table_files_tgt_psl

    column_checkbox = pn.widgets.MultiChoice(
        name=" ",
        value=[
            "Upload ID",
            "TAC_done",
            "n_obj",
            "Time_tot_L (h)",
            "Time_tot_M (h)",
            "timestamp",
            "TAC_FH_L",
            "TAC_FH_M",
            "observation_type",
            "pointing_status",
        ],
        options=list(df_files_tgt_psl.columns)
        + ["proposal ID", "PI name", "rank", "grade"],
    )

    table_files_tgt_psl = pn.bind(Table_files_tgt_psl, column_checkbox)

    # download buttons
    download_selection = pn.widgets.Button(
        name="Download all the selected programs",
        icon="download",
        button_type="primary",
        stylesheets=[
            """
            .bk-btn {
                color: var(--success-text-color) !important;
                background-color: #C7E2D6 !important;
                border-color: var(--success-border-subtle) !important;
                border-width: 1px;
                font-weight: bold !important;
                font-size: 140% !important;
                // color: #145B33;
                // background-color: #eaf4fc;
                // background-color: var(--success-bg-color);
                // border-color: #008899;
            }
        """
        ],
        width=300,
    )
    download_group = pn.widgets.RadioBoxGroup(
        value="Target",
        options=["Target", "PPC", "PPC (after allocation)"],
        inline=True,
        align="center",
        margin=(0, 0, 0, 25),
    )

    # summary of tac allocation
    tac_summary = pn.pane.Markdown(
        "<font size=5>Summary of TAC allocation:</font>\n"
        f"<font size=4> - N (allocated programs): <span style='color:tomato'>**{sum((df_files_tgt_psl['TAC_nppc_L']>0) | (df_files_tgt_psl['TAC_nppc_M']>0)):.0f}**</span></font>\n"
        "<font size=4> - Low-res mode: </font>\n"
        f"<font size=4>&emsp;FH allocated = <span style='color:tomato'>**{sum(df_files_tgt_psl['TAC_FH_L']):.2f}**</span></font>\n"
        f"<font size=4>&emsp;Nppc allocated = <span style='color:tomato'>**{sum(df_files_tgt_psl['TAC_nppc_L']):.0f}**</span> </font>\n"
        f"<font size=4>&emsp;ROT (h) allocated = <span style='color:tomato'>**{sum(df_files_tgt_psl['TAC_ROT_L']):.2f}**</span> </font>\n"
        "<font size=4> - Medium-res mode: </font>\n"
        f"<font size=4>&emsp;FH allocated = <span style='color:tomato'>**{sum(df_files_tgt_psl['TAC_FH_M']):.2f}**</span></font>\n"
        f"<font size=4>&emsp;Nppc allocated = <span style='color:tomato'>**{sum(df_files_tgt_psl['TAC_nppc_M']):.0f}**</span> </font>\n"
        f"<font size=4>&emsp;ROT (h) allocated = <span style='color:tomato'>**{sum(df_files_tgt_psl['TAC_ROT_M']):.2f}**</span> </font>\n"
    )

    # Details of PPC
    table_ppc = pn.Column()

    # -------------------------------------------------------------------
    js_panel = pn.pane.HTML(width=0, height=0, margin=0, sizing_mode="fixed")

    sidebar_column = pn.Column(
        psl_info,
        tac_summary,
    )

    tab_panels = pn.Tabs(
        (
            "Program info",
            pn.Column(
                pn.pane.Markdown(
                    "<font size=4> Select the columns to show:</font>",
                    height=20,
                ),
                column_checkbox,
                pn.pane.Markdown(
                    "<font size=4> Select the proposals to show:</font>",
                    height=35,
                ),
                pn.Row(
                    slider_nobj,
                    slider_fiberhour,
                    slider_rot_l,
                    slider_rot_m,
                    height=60,
                ),
                pn.Row(download_selection, download_group),
                table_files_tgt_psl,
                js_panel,
            ),
        ),
        ("Target list", panel_targets.pane),
        ("PPC details", table_ppc),
    )

    # put them into the template
    template.sidebar.append(sidebar_column)
    template.main.append(tab_panels)

    app = template

    if use_panel_cli:
        return app.servable()
    else:
        return app
