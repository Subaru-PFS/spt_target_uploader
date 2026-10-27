#!/usr/bin/env python3
"""Panel app for the ToO (Target of Opportunity) mode uploader.

This app allows validation and submission of pfsDesign FITS files.
It follows the same structure and CSS as the normal target uploader
(``pn_app.py``), but only provides the "Validate" and "Submit" operations.

Submitted files are stored under the directory specified by the optional
``OUTPUT_DIR_too`` key in ``.env.shared``; if it is not set, they are stored
under ``<OUTPUT_DIR>/too`` (``OUTPUT_DIR`` is defined in ``.env.shared``,
see ``.env.shared.example``).
"""

import asyncio
import importlib
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pprint import pformat
from zipfile import ZipFile

import pandas as pd
import panel as pn
import param
from astropy.io import fits
from astropy.table import Table
from loguru import logger

try:
    from . import __version__ as pfs_target_uploader_version
    from .utils.config import load_app_config
    from .utils.mail import send_email
    from .utils.session import assign_secret_token
    from .widgets import (
        AnnouncementNoteWidgets,
        DocLinkWidgets,
        SubmitButtonWidgets,
        TimerWidgets,
        UploadNoteWidgets,
        ValidateButtonWidgets,
    )
except ImportError:
    # served directly by path (e.g. "panel serve pn_ToO.py"): no parent package
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pfs_target_uploader import __version__ as pfs_target_uploader_version
    from pfs_target_uploader.utils.config import load_app_config
    from pfs_target_uploader.utils.mail import send_email
    from pfs_target_uploader.utils.session import assign_secret_token
    from pfs_target_uploader.widgets import (
        AnnouncementNoteWidgets,
        DocLinkWidgets,
        SubmitButtonWidgets,
        TimerWidgets,
        UploadNoteWidgets,
        ValidateButtonWidgets,
    )

# pfsDesign file naming convention: pfsDesign-0x<16-digit hex designId>.fits
DESIGN_FILENAME_PATTERN = re.compile(
    r"^pfsDesign-0x(?P<design_id>[0-9a-fA-F]{16})\.fits$"
)

# HDUs and primary header keywords expected in a pfsDesign file
REQUIRED_HDUS = ["DESIGN", "PHOTOMETRY", "GUIDESTARS"]
REQUIRED_HEADER_KEYS = ["RA", "DEC", "POSANG", "ARMS", "DSGN_NAM", "W_PFDSGN"]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBSPROC_PLANNING_SRC = "/home/wanqiu/data/HE/PFS_frame/git/pfs_obsproc_planning_tools/src"


def _toggle_widgets(widgets: list, disabled: bool = True):
    for w in widgets:
        w.disabled = disabled


def _generate_too_readme_text():
    readme_text = """# README

This directory contains pfsDesign files submitted via the PFS ToO uploader.

- `pfsDesign-0x*.fits`: submitted pfsDesign file(s)
- `design_summary_<Upload ID>.ecsv`: summary of the submitted design files
- `pfs_too_design-<UTC>-<Upload ID>.zip`: zip file containing all of the above

Please keep the Upload ID for further inquiries.
"""
    return readme_text


def validate_design_with_obsproc(filename, data, design_id, plot_dir=None):
    """Run the local per-design checks from obsproc planning validation."""
    result = {"status": False, "errors": [], "warnings": [], "plot_path": None}

    try:
        if OBSPROC_PLANNING_SRC not in sys.path:
            sys.path.insert(0, OBSPROC_PLANNING_SRC)
        obsproc_validation = importlib.import_module(
            "pfs_obsproc_planning.utils.validation"
        )

        with tempfile.TemporaryDirectory(prefix="pfs-too-validation-") as temp_dir:
            design_path = os.path.join(temp_dir, filename)
            with open(design_path, "wb") as design_file:
                design_file.write(data)

            pfs_design = obsproc_validation.PfsDesign.read(
                design_id, dirName=temp_dir
            )
            pfs_design.validate()

            obsproc_validation._warn_duplicate_fibers(pfs_design)
            duplicate_fibers = pd.Series(pfs_design.fiberId).duplicated(keep=False)
            if duplicate_fibers.any():
                result["warnings"].append(
                    f"duplicated fiber IDs: {sorted(set(pd.Series(pfs_design.fiberId)[duplicate_fibers]))}"
                )

            df_fib = obsproc_validation._build_df_fib(pfs_design)
            too_bright = df_fib[
                (df_fib["psfMag"] < 13) | (df_fib["totalMag"] < 13)
            ]
            if not too_bright.empty:
                result["warnings"].append(
                    f"{len(too_bright)} source(s) have psfMag or totalMag < 13"
                )

            df_ag = obsproc_validation._build_df_ag(pfs_design)

            if plot_dir is not None:
                os.makedirs(plot_dir, exist_ok=True)
                plot_prefix = os.path.join(plot_dir, f"check_0x{design_id:016x}")
                title = (
                    f"designId=0x{design_id:016x} "
                    f"({pfs_design.raBoresight:.2f},{pfs_design.decBoresight:.2f},"
                    f"PA={pfs_design.posAng:.1f})\n{pfs_design.designName}"
                )
                obsproc_validation.pldes.plot_FoV(
                    df_fib,
                    df_ag,
                    title=title,
                    fname=plot_prefix,
                    save=True,
                    show=False,
                    pa=pfs_design.posAng,
                    conf={"ppp": {"mode": "too"}},
                )
                obsproc_validation.plt.close("design_FoV")
                result["plot_path"] = f"{plot_prefix}.pdf"

        result["status"] = True
    except Exception as e:
        result["errors"].append(f"obsproc planning validation failed: {e}")

    return result


def validate_design_file(filename, data, plot_dir=None):
    """Validate a single pfsDesign FITS file.

    Parameters
    ----------
    filename : str
        Name of the uploaded file.
    data : bytes
        Content of the uploaded file.

    Returns
    -------
    dict
        Per-file summary including a ``status`` flag and a list of ``errors``.
    """
    summary = {
        "filename": filename,
        "design_id": None,
        "design_name": None,
        "ra": None,
        "dec": None,
        "pa": None,
        "arms": None,
        "n_fibers": None,
        "n_guidestars": None,
        "obsproc_status": False,
        "obsproc_warnings": [],
        "plot_path": None,
        "status": False,
        "errors": [],
    }

    match = DESIGN_FILENAME_PATTERN.match(filename)
    if match is None:
        summary["errors"].append(
            f"{filename}: filename does not follow the pfsDesign-0x<16-digit hex>.fits convention"
        )
    else:
        summary["design_id"] = f"0x{match.group('design_id').lower()}"

    try:
        with fits.open(BytesIO(data)) as hdul:
            hdu_names = [hdu.name for hdu in hdul]

            for hdu_name in REQUIRED_HDUS:
                if hdu_name not in hdu_names:
                    summary["errors"].append(
                        f"{filename}: required HDU '{hdu_name}' is missing"
                    )

            header = hdul[0].header
            for key in REQUIRED_HEADER_KEYS:
                if key not in header:
                    summary["errors"].append(
                        f"{filename}: required header keyword '{key}' is missing"
                    )

            summary["design_name"] = header.get("DSGN_NAM", None)
            summary["ra"] = header.get("RA", None)
            summary["dec"] = header.get("DEC", None)
            summary["pa"] = header.get("POSANG", None)
            summary["arms"] = header.get("ARMS", None)

            # check consistency between the filename and the design ID in the header
            if (match is not None) and ("W_PFDSGN" in header):
                design_id_header = int(header["W_PFDSGN"])
                design_id_filename = int(match.group("design_id"), 16)
                if design_id_header != design_id_filename:
                    summary["errors"].append(
                        f"{filename}: design ID in the header (0x{design_id_header:016x}) "
                        f"does not match the filename (0x{design_id_filename:016x})"
                    )

            if "DESIGN" in hdu_names:
                n_fibers = len(hdul["DESIGN"].data)
                summary["n_fibers"] = n_fibers
                if n_fibers == 0:
                    summary["errors"].append(
                        f"{filename}: DESIGN HDU contains no entries"
                    )

            if "GUIDESTARS" in hdu_names:
                summary["n_guidestars"] = len(hdul["GUIDESTARS"].data)

            if (match is not None) and ("W_PFDSGN" in header):
                obsproc_result = validate_design_with_obsproc(
                    filename,
                    data,
                    int(header["W_PFDSGN"]),
                    plot_dir=plot_dir,
                )
                summary["obsproc_status"] = obsproc_result["status"]
                summary["obsproc_warnings"] = obsproc_result["warnings"]
                summary["plot_path"] = obsproc_result["plot_path"]
                summary["errors"].extend(
                    f"{filename}: {error}" for error in obsproc_result["errors"]
                )

    except Exception as e:
        summary["errors"].append(f"{filename}: cannot be read as a FITS file ({e})")

    summary["status"] = len(summary["errors"]) == 0

    return summary


class ToODesignInputWidgets(param.Parameterized):
    """File input widget for pfsDesign FITS files (ToO mode)."""

    def __init__(self):
        self.file_input = pn.widgets.FileInput(
            value=None,
            filename=None,
            accept=".fits",
            multiple=True,
            height=40,
        )

        # store previous information for input comparison
        self.previous_filename = None
        self.previous_value = None

        # hex string to be used as an upload ID
        self.secret_token = None
        self.db_path = None
        self.output_dir = None
        self.use_db = True

        self.pane = pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    "<font size=4><i class='fas fa-list-ul'></i>  **pfsDesign file(s)**</font> "
                    "<font size=4>(FITS)</font>",
                    width=400,
                ),
                pn.widgets.TooltipIcon(
                    value="Select one or more **pfsDesign** files (pfsDesign-0x&lt;hex&gt;.fits).",
                    margin=(0, 0, 0, -165),
                ),
            ),
            self.file_input,
        )

    def reset(self):
        self.file_input.filename = None
        self.file_input.mime_type = None
        self.file_input.value = None

    def _current_files(self):
        # with multiple=True, filename/value are lists; normalize just in case
        filenames = self.file_input.filename
        values = self.file_input.value
        if filenames is None or values is None:
            return None, None
        if isinstance(filenames, str):
            filenames = [filenames]
            values = [values]
        return list(filenames), list(values)

    def validate(self):
        """Validate the uploaded pfsDesign files.

        Returns
        -------
        tuple
            (validation_status dict or None, summary pandas.DataFrame or None)
        """
        filenames, values = self._current_files()

        if filenames is None:
            logger.info("No file selected.")
            pn.state.notifications.error("Please select pfsDesign FITS file(s).")
            return None, None

        # update the upload ID when the input files differ from the previous validation
        if (filenames != self.previous_filename) or (values != self.previous_value):
            self.secret_token = assign_secret_token(
                db_path=self.db_path, output_dir=self.output_dir, use_db=self.use_db
            )
            logger.info("New file(s) detected.")
            logger.info(f"    Upload ID updated: {self.secret_token}")
            logger.info(f"    Filenames: {filenames}")

            self.previous_filename = filenames
            self.previous_value = values
        else:
            logger.info("Identical to the previous validation.")
            logger.info("    Upload ID not updated.")

        logger.info(f"Upload ID: {self.secret_token}")

        summaries = []
        duplicate_errors = []

        seen = set()
        for name in filenames:
            if name in seen:
                duplicate_errors.append(f"{name}: duplicated in the uploaded files")
            seen.add(name)

        for name, data in zip(filenames, values):
            logger.info(f"Validating {name}")
            plot_dir = os.path.join(
                self.output_dir,
                "validation",
                self.secret_token,
            )
            summaries.append(validate_design_file(name, data, plot_dir=plot_dir))

        df_summary = pd.DataFrame(
            summaries,
            columns=[
                "filename",
                "design_id",
                "design_name",
                "ra",
                "dec",
                "pa",
                "arms",
                "n_fibers",
                "n_guidestars",
                "obsproc_status",
                "obsproc_warnings",
                "plot_path",
                "status",
                "errors",
            ],
        )

        errors = duplicate_errors + [
            e for s in summaries for e in s["errors"]
        ]

        validation_status = {
            "status": len(errors) == 0,
            "n_files": len(filenames),
            "n_ok": int(df_summary["status"].sum()),
            "n_error": int((~df_summary["status"]).sum()) + len(duplicate_errors),
            "errors": errors,
        }

        # convert error list to a single string for tabular display
        df_summary["errors"] = df_summary["errors"].apply(lambda x: "; ".join(x))
        df_summary["obsproc_warnings"] = df_summary["obsproc_warnings"].apply(
            lambda x: "; ".join(x)
        )

        return validation_status, df_summary


class ToOStatusWidgets:
    """Sidebar status pane (same look and feel as StatusWidgets)."""

    def __init__(self):
        self.neutral_text = "<font size=4>Waiting...</font>"
        self.success_text = "<font size=5>✅ **Success**</font>"
        self.error_text = "<font size=5>🚫 **Failed**</font>"

        self.status = pn.pane.Alert(
            self.neutral_text, alert_type="secondary", height=80
        )
        self.summary_nobj = pn.pane.Markdown("", visible=False)

        self.pane = pn.Column(self.status, self.summary_nobj)

    def reset(self):
        self.status.object = self.neutral_text
        self.status.alert_type = "secondary"
        self.summary_nobj.object = ""
        self.summary_nobj.visible = False

    def show_results(self, validation_status):
        if validation_status["status"]:
            self.status.object = self.success_text
            self.status.alert_type = "success"
        else:
            self.status.object = self.error_text
            self.status.alert_type = "danger"

        self.summary_nobj.object = (
            f"<font size=3>{validation_status['n_ok']} / {validation_status['n_files']} "
            "design file(s) passed the validation.</font>"
        )
        self.summary_nobj.visible = True


class ToOValidationResultWidgets:
    """Main-area validation results (summary table and error messages)."""

    stylesheet = """
        .tabulator-row-odd { background-color: #ffffff !important; }
        .tabulator-row-even { background-color: #ffffff !important; }
        """

    def __init__(self):
        self.error_pane = pn.pane.Alert("", alert_type="danger", visible=False)
        self.success_pane = pn.pane.Alert("", alert_type="success", visible=False)

        self.summary_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            theme="bootstrap",
            layout="fit_data_table",
            hidden_columns=["index", "plot_path"],
            selectable=1,
            disabled=True,
            visible=False,
            stylesheets=[self.stylesheet],
            configuration={"columnDefaults": {"editor": False}},
        )

        self.pane = pn.Column(
            "# Validation of pfsDesign files",
            self.error_pane,
            self.success_pane,
            self.summary_table,
        )

    def reset(self):
        self.error_pane.object = ""
        self.error_pane.visible = False
        self.success_pane.object = ""
        self.success_pane.visible = False
        self.summary_table.value = pd.DataFrame()
        self.summary_table.selection = []
        self.summary_table.visible = False

    def show_results(self, df_summary, validation_status):
        if validation_status["status"]:
            self.success_pane.object = (
                "<font size=5>✅ **Success**</font>\n\n"
                f"<font size=3>All {validation_status['n_files']} design file(s) passed the validation.</font>"
            )
            self.success_pane.visible = True
        else:
            error_lines = "\n".join([f"- {e}" for e in validation_status["errors"]])
            self.error_pane.object = (
                "<font size=5>🚫 **Failed**</font>\n\n"
                f"<font size=3>The following issue(s) must be fixed before submission:</font>\n\n"
                f"{error_lines}"
            )
            self.error_pane.visible = True

        self.summary_table.value = df_summary
        self.summary_table.visible = True


class ToOValidationPlotWidgets:
    """Embedded obsproc validation plots and download controls."""

    def __init__(self):
        self.downloads = pn.Column(
            pn.pane.Markdown("# pfsDesign validation plots"),
            pn.pane.Markdown(
                "Select a design in the **Validation** table to display its plot."
            ),
        )
        self.pane = self.downloads

    def reset(self):
        self.downloads[:] = [
            pn.pane.Markdown("# pfsDesign validation plots"),
            pn.pane.Markdown(
                "Select a design in the **Validation** table to display its plot."
            ),
        ]

    def show_results(self, df_summary):
        self.reset()

    def show_plot(self, plot_path):
        if not plot_path or not os.path.exists(plot_path):
            self.downloads[:] = [
                pn.pane.Markdown("# pfsDesign validation plots"),
                pn.pane.Alert("No validation plot was generated for this design."),
            ]
            return

        filename = os.path.basename(plot_path)
        self.downloads[:] = [
            pn.pane.Markdown("# pfsDesign validation plots"),
            pn.pane.Markdown(f"## {filename}"),
            pn.pane.PDF(
                plot_path,
                embed=True,
                height=850,
                sizing_mode="stretch_width",
            ),
            pn.widgets.FileDownload(
                file=plot_path,
                filename=filename,
                label=f"Download {filename}",
                icon="download",
                button_type="primary",
                width=360,
            ),
        ]


def upload_too_files(
    filenames,
    values,
    df_summary,
    outdir_prefix=".",
    secret_token=None,
    upload_time=None,
):
    """Write the submitted pfsDesign files to the storage.

    The output directory structure follows the normal uploader:
    ``<outdir_prefix>/<YYYY>/<MM>/<YYYYmmdd-HHMMSS>-<upload_id>/``.

    Returns
    -------
    tuple
        (outdir, outfile_zip)
    """
    dt = upload_time if upload_time is not None else datetime.now(timezone.utc)

    outdir = os.path.join(
        outdir_prefix,
        f"{dt.year:4d}",
        f"{dt.month:02d}",
        f"{dt:%Y%m%d-%H%M%S}-{secret_token}",
    )

    if not os.path.exists(outdir):
        logger.info(f"{outdir} is created")
        os.makedirs(outdir)
    else:
        logger.warning(f"{outdir} already exists, strange")

    outfile_zip_prefix = f"pfs_too_design-{dt:%Y%m%d-%H%M%S}-{secret_token}"

    # summary table with upload metadata
    tb_summary = Table.from_pandas(df_summary)
    tb_summary.meta["original_filenames"] = list(filenames)
    tb_summary.meta["upload_id"] = secret_token
    tb_summary.meta["upload_at"] = dt
    summary_filename = f"design_summary_{secret_token}.ecsv"
    tb_summary.write(
        os.path.join(outdir, summary_filename),
        delimiter=",",
        format="ascii.ecsv",
        overwrite=True,
    )

    readme_filename = "README.txt"
    with open(os.path.join(outdir, readme_filename), "w") as f:
        f.write(_generate_too_readme_text())

    # write the original design files
    for name, data in zip(filenames, values):
        with open(os.path.join(outdir, name), "wb") as f:
            f.write(data)
        logger.info(f"File {name} is saved under {outdir}.")

    # bundle everything into a zip file
    outfile_zip = f"{outfile_zip_prefix}.zip"
    with ZipFile(os.path.join(outdir, outfile_zip), "w") as zipfile:
        for name in [*filenames, summary_filename, readme_filename]:
            zipfile.write(
                os.path.join(outdir, name),
                arcname=os.path.join(outfile_zip_prefix, name),
            )

    logger.info("zip file made")

    return outdir, outfile_zip


def too_uploader_app(use_panel_cli=False):
    pn.state.notifications.position = "bottom-left"

    # Panel executes path-served apps with the shell's working directory.
    os.chdir(PROJECT_ROOT)

    # Load and validate configuration (same as the normal uploader)
    config = load_app_config(
        env_file=os.path.join(PROJECT_ROOT, ".env.shared"),
        create_output_dir=True,
        validate_db=False,
        validate_ann_file=False,
    )

    # Output directory for ToO submissions:
    # use OUTPUT_DIR_too if defined in .env.shared, otherwise <OUTPUT_DIR>/too
    output_dir_too = config.raw_config.get("OUTPUT_DIR_too") or os.path.join(
        config.output_dir, "too"
    )
    if os.path.exists(output_dir_too):
        logger.info(f"{output_dir_too} already exists.")
    else:
        os.makedirs(output_dir_too)
        logger.info(f"{output_dir_too} created.")

    # Configure logger to be multiprocessing-safe
    logger.remove()
    logger.add(sys.stderr, level=config.log_level, enqueue=True)

    # Log request headers
    if pn.state.headers:
        logger.info(f"Request headers:\n{pformat(dict(pn.state.headers), width=100)}")

    # Log request URL
    if pn.state.location and pn.state.location.href:
        logger.info(f"Request URL: {pn.state.location.href}")

    # Log application configuration
    logger.info(f"\n{config.format_for_logging()}")
    logger.info(f"ToO output directory: {output_dir_too}")

    template = pn.template.MaterialTemplate(
        title="PFS design uploader",
        sidebar_width=420,
        header_background="#3A7D7E",
        busy_indicator=None,
        favicon="doc/assets/images/favicon.png",
    )

    # setup panel components
    panel_doc = DocLinkWidgets()

    panel_input = ToODesignInputWidgets()
    panel_validate_button = ValidateButtonWidgets()
    panel_status = ToOStatusWidgets()
    panel_submit_button = SubmitButtonWidgets()

    panel_timer = TimerWidgets(total_seconds=config.max_exetime)

    panel_results = ToOValidationResultWidgets()
    panel_plots = ToOValidationPlotWidgets()

    panel_input.reset()
    panel_input.db_path = None
    panel_input.output_dir = output_dir_too
    panel_input.use_db = False

    button_set = [
        panel_input.file_input,
        panel_validate_button.validate,
    ]

    placeholder_floatpanel = pn.Column(height=0, width=0)
    placeholder_announcement = pn.Column(height=0, width=0)

    if config.ann_file is not None and os.path.exists(config.ann_file):
        panel_annoucement = AnnouncementNoteWidgets(config.ann_file)
        placeholder_announcement[:] = [panel_annoucement.floatpanel]

    # if no file is uploaded, disable the buttons
    def enable_buttons_by_fileinput(v, pv):
        if v is None:
            logger.info("Buttons are disabled because no file is uploaded.")
            _toggle_widgets(
                [panel_validate_button.validate, panel_submit_button.submit],
                disabled=True,
            )
            return
        logger.info("Buttons are enabled because file upload is detected.")
        _toggle_widgets(
            [panel_validate_button.validate],
            disabled=False,
        )
        if (v is not None) and (v != pv):
            _toggle_widgets(
                [panel_submit_button.submit],
                disabled=True,
            )

    fileinput_watcher = pn.bind(
        enable_buttons_by_fileinput,
        panel_input.file_input,
        panel_input.previous_value,
    )

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        pn.Column(
            pn.Row(
                "<font size=4><i class='fas fa-calculator'></i> **Execute an operation**</font>",
                panel_timer.pane,
            ),
            pn.Row(
                panel_validate_button.pane,
                panel_submit_button.pane,
                sizing_mode="stretch_width",
            ),
            margin=(10, 0, 0, 0),
        ),
        pn.Column(
            pn.Row(
                "<font size=4><i class='fa-solid fa-magnifying-glass-chart fa-lg'></i> **Validation status**</font>"
            ),
            panel_status.pane,
            margin=(10, 0, 0, 0),
        ),
        fileinput_watcher,
    )

    sidebar_about = pn.Column(
        pn.Column(
            pn.pane.Markdown(
                "<font size=4><i class='fas fa-tag'></i>  **Version**</font>",
                sizing_mode="stretch_width",
            ),
        ),
        pn.Column(
            pn.pane.Markdown(
                f"{pfs_target_uploader_version}",
                sizing_mode="stretch_width",
            ),
            margin=(-20, 0, 0, 0),
        ),
        margin=(10, 0, 0, 0),
    )

    tab_sidebar = pn.Tabs(
        ("Home", sidebar_column),
        ("About", sidebar_about),
    )

    # bundle panel(s) in the main area
    tab_panels = pn.Tabs(
        ("Validation", panel_results.pane),
        ("Design plots", panel_plots.pane),
    )

    def show_selected_design(event):
        if not event.new:
            return

        selected = event.new[0]
        df_summary = panel_results.summary_table.value
        try:
            row = df_summary.loc[selected]
        except KeyError:
            row = df_summary.iloc[selected]

        panel_plots.show_plot(row["plot_path"])
        tab_panels.active = 1

    panel_results.summary_table.param.watch(
        show_selected_design,
        "selection",
    )

    sidepanel_column = pn.Column(
        panel_doc.pane,
        tab_sidebar,
    )

    main_column = pn.Column(
        placeholder_floatpanel,
        placeholder_announcement,
        tab_panels,
        margin=(30, 0, 0, 0),
    )

    # put them into the template
    template.sidebar.append(sidepanel_column)
    template.main.append(main_column)

    tab_panels.visible = False

    # define on_click callback for the "validate" button
    async def cb_validate(event):
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)

        placeholder_floatpanel.objects = []

        tab_panels.visible = False

        panel_status.reset()
        panel_results.reset()
        panel_plots.reset()

        pn.state.notifications.clear()

        panel_timer.timer(on=True, time_limit=False)

        validation_status, df_summary = await asyncio.to_thread(panel_input.validate)

        _toggle_widgets(button_set, disabled=False)

        if validation_status is None:
            panel_timer.timer(on=False, time_limit=False)
            return

        panel_status.show_results(validation_status)
        panel_results.show_results(df_summary, validation_status)
        panel_plots.show_results(df_summary)

        tab_panels.active = 0
        tab_panels.visible = True

        panel_timer.timer(on=False, time_limit=False)

        if validation_status["status"]:
            panel_submit_button.enable_button(True)

    # define on_click callback for the "submit" button
    async def cb_submit(event):
        _toggle_widgets(button_set, disabled=True)
        _toggle_widgets([panel_submit_button.submit], disabled=True)

        placeholder_floatpanel.objects = []

        logger.info("Submit button clicked.")
        logger.info("Validation before actually writing to the storage")

        panel_timer.timer(on=True, time_limit=False)

        # do the validation again (input files can be different)
        validation_status, df_summary = await asyncio.to_thread(panel_input.validate)

        if (validation_status is None) or (not validation_status["status"]):
            logger.error("Validation failed for some reason")

            tab_panels.visible = False

            panel_status.reset()
            panel_results.reset()
            panel_plots.reset()

            pn.state.notifications.clear()

            _toggle_widgets(button_set, disabled=False)

            if validation_status is None:
                panel_timer.timer(on=False, time_limit=False)
                return
            else:
                panel_status.show_results(validation_status)
                panel_results.show_results(df_summary, validation_status)
                panel_plots.show_results(df_summary)
                tab_panels.visible = True
                panel_timer.timer(on=False, time_limit=False)
                return

        filenames, values = panel_input._current_files()
        upload_time = datetime.now(timezone.utc)
        secret_token = panel_input.secret_token

        outdir, outfile_zip = await asyncio.to_thread(
            upload_too_files,
            filenames,
            values,
            df_summary,
            outdir_prefix=output_dir_too,
            secret_token=secret_token,
            upload_time=upload_time,
        )

        try:
            if (
                (
                    "EMAIL_FROM" not in config.raw_config.keys()
                    or config.raw_config["EMAIL_FROM"] == ""
                )
                or (
                    "EMAIL_TO" not in config.raw_config.keys()
                    or config.raw_config["EMAIL_TO"] == ""
                )
                or (
                    "SMTP_SERVER" not in config.raw_config.keys()
                    or config.raw_config["SMTP_SERVER"] == ""
                )
            ):
                logger.warning(
                    "Email configuration is not found. No email will be sent."
                )
            else:
                await asyncio.to_thread(
                    send_email,
                    config.raw_config,
                    outdir=outdir,
                    outfile=outfile_zip,
                    upload_id=secret_token,
                    upload_time=upload_time,
                    url=pn.state.location.href,
                )
        except Exception as e:
            logger.error(f"Failed to send an email: {str(e)}")

        # href for the download link in the upload note
        # (same convention as the normal uploader: OUTPUT_DIR is served as "data/")
        if output_dir_too.startswith(config.output_dir):
            href_dir = outdir.replace(config.output_dir, "data/", 1).replace("//", "/")
        else:
            href_dir = outdir.replace(output_dir_too, "data_too/", 1).replace(
                "//", "/"
            )

        panel_notes = UploadNoteWidgets(
            secret_token,
            upload_time,
            True,
            href_dir,
            outfile_zip,
        )
        placeholder_floatpanel[:] = [panel_notes.floatpanel]

        _toggle_widgets(button_set, disabled=False)
        _toggle_widgets([panel_submit_button.submit], disabled=True)

        panel_input.secret_token = await asyncio.to_thread(
            assign_secret_token,
            db_path=None,
            output_dir=output_dir_too,
            use_db=False,
        )

        panel_timer.timer(on=False, time_limit=False)

    # set callback to the buttons
    panel_validate_button.validate.on_click(cb_validate)
    panel_submit_button.submit.on_click(cb_submit)

    app = template

    if use_panel_cli:
        return app.servable()
    else:
        return app


def _configure_panel_extension():
    """Mirror the pn.extension setup done by the CLI (for standalone serving)."""
    pn.extension(
        "floatpanel",
        "mathjax",
        "tabulator",
        notifications=True,
        loading_spinner="dots",
        sizing_mode="stretch_width",
        js_files={
            "font-awesome": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js",
        },
        css_files=[
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
        ],
        layout_compatibility="error",
        disconnect_notification="Connection lost, try reloading the page.",
    )


# allow "panel serve src/pfs_target_uploader/pn_ToO.py" without changing the CLI
if __name__.startswith("bokeh"):
    _configure_panel_extension()
    too_uploader_app(use_panel_cli=True)
