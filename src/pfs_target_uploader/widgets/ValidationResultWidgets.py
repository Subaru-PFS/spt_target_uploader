#!/usr/bin/env python3

from io import BytesIO

import numpy as np
import pandas as pd
import panel as pn
from astropy import units as u
from bokeh.models import NumberFormatter
from pandas.io.formats.style import Styler as _PandasStyler

# Shrink the label and icon (the icon scales with the inherited font-size)
# so the Download CSV buttons sit closer to the surrounding body text size.
# !important is required to win the cascade over Panel/Bokeh's own button
# CSS, which otherwise keeps the default font-size.
_download_button_size_stylesheet = """
    .bk-btn {
        font-size: 0.85rem !important;
    }
    """


def _csv_download_callback(table_or_getter):
    """Return a FileDownload callback that streams data as CSV.

    *table_or_getter* may be either a Tabulator widget or a zero-argument
    callable that returns a ``pd.DataFrame`` (or ``None``).  The callable form
    is useful when the export DataFrame differs from what the Tabulator displays
    (e.g. extra indicator columns that should not appear in the UI).

    The callback is evaluated at click time, so it always reflects the data set
    by the most recent ``show_results`` call.

    The hidden numeric ``obj_id`` column is dropped, and ``obj_id_str`` is
    renamed to ``obj_id`` so the CSV matches the display label.
    """

    def _cb():
        if callable(table_or_getter):
            df = table_or_getter()
            if df is None:
                df = pd.DataFrame()
        else:
            val = table_or_getter.value
            if val is None:
                df = pd.DataFrame()
            elif isinstance(val, _PandasStyler):
                df = val.data.copy()
            else:
                df = val.copy()
        # Drop the hidden numeric obj_id column; rename obj_id_str → obj_id.
        df = df.drop(
            columns=[c for c in ("obj_id",) if c in df.columns], errors="ignore"
        )
        if "obj_id_str" in df.columns:
            df = df.rename(columns={"obj_id_str": "obj_id"})
        bio = BytesIO()
        bio.write(df.to_csv(index=False).encode("utf-8"))
        bio.seek(0)
        return bio

    return _cb


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
        self.warning_text_flux = pn.pane.Markdown("", max_width=self.box_width)
        self.warning_text_fluxrange = pn.pane.Markdown("", max_width=self.box_width)
        self.warning_text_visibility = pn.pane.Markdown("", max_width=self.box_width)
        self.warning_text_intdups = pn.pane.Markdown("", max_width=self.box_width)

        self.info_text_keys = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_str = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_vals = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_flux = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_visibility = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_dups = pn.pane.Markdown("", max_width=self.box_width)
        self.info_text_intdups = pn.pane.Markdown("", max_width=self.box_width)

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

        self.warning_table_fluxrange = pn.widgets.Tabulator(
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

        self.warning_table_intdups = pn.widgets.Tabulator(
            pd.DataFrame(), **self.tabulator_kwargs
        )

        # --- Download buttons (one per flagged table) ---
        # Built by a factory rather than a shared dict so that each widget gets
        # its own ``stylesheets`` list; Panel stores the list by reference and a
        # shared one would let a mutation on any button leak into all the others.
        def _dl_kwargs():
            return dict(
                label="Download CSV",
                button_type="default",
                button_style="solid",
                icon="download",
                icon_size="1.25em",
                visible=False,
                max_width=180,
                stylesheets=[_download_button_size_stylesheet],
            )

        self.download_button_str = pn.widgets.FileDownload(
            callback=_csv_download_callback(self.error_table_str),
            filename="pfs_validation_invalid_string.csv",
            **_dl_kwargs(),
        )
        self.download_button_vals = pn.widgets.FileDownload(
            callback=_csv_download_callback(self.error_table_vals),
            filename="pfs_validation_value_errors.csv",
            **_dl_kwargs(),
        )
        self.download_button_flux = pn.widgets.FileDownload(
            callback=_csv_download_callback(self.error_table_flux),
            filename="pfs_validation_missing_flux.csv",
            **_dl_kwargs(),
        )
        # Separate export DataFrame for the flux-range table: identical to the
        # displayed data but with an extra ``out_of_range_bands`` column that
        # lists which bands are out of range for each row (e.g. "g,r").
        self._df_fluxrange_csv: pd.DataFrame | None = None
        self.download_button_fluxrange = pn.widgets.FileDownload(
            callback=_csv_download_callback(lambda: self._df_fluxrange_csv),
            filename="pfs_validation_flux_out_of_range.csv",
            **_dl_kwargs(),
        )
        self.download_button_visibility = pn.widgets.FileDownload(
            callback=_csv_download_callback(self.warning_table_visibility),
            filename="pfs_validation_no_visibility.csv",
            **_dl_kwargs(),
        )
        self.download_button_dups = pn.widgets.FileDownload(
            callback=_csv_download_callback(self.error_table_dups),
            filename="pfs_validation_duplicate_obcode.csv",
            **_dl_kwargs(),
        )
        self.download_button_intdups = pn.widgets.FileDownload(
            callback=_csv_download_callback(self.warning_table_intdups),
            filename="pfs_validation_internal_duplication.csv",
            **_dl_kwargs(),
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
            self.warning_text_flux,
            self.warning_text_fluxrange,
            self.warning_text_visibility,
            self.warning_text_intdups,
            self.info_text_keys,
            self.info_text_str,
            self.info_text_vals,
            self.info_text_flux,
            self.info_text_visibility,
            self.info_text_dups,
            self.info_text_intdups,
        ]:
            t.object = ""

        for t in [
            self.error_table_str,
            self.warning_table_str,
            self.error_table_vals,
            self.warning_table_vals,
            self.error_table_flux,
            self.warning_table_fluxrange,
            self.error_table_visibility,
            self.warning_table_visibility,
            self.error_table_dups,
            self.warning_table_intdups,
        ]:
            if t.value is not None:
                t.value[0:0]
            t.visible = False

        for btn in [
            self.download_button_str,
            self.download_button_vals,
            self.download_button_flux,
            self.download_button_fluxrange,
            self.download_button_visibility,
            self.download_button_dups,
            self.download_button_intdups,
        ]:
            btn.visible = False

        self._df_fluxrange_csv = None

        # Drop the flux-range highlighting/formatting left over from the previous
        # validation: the Styler holds a closure over that run's
        # ``out_of_range_bands``, and the formatters over its band columns.
        # Both are rebuilt from scratch whenever the flux-range section is shown
        # again, so nothing here needs to survive a reset.
        self.warning_table_fluxrange.style = None
        self.warning_table_fluxrange.formatters = {}

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

    def _identify_out_of_range_bands(self, df, min_flux_nJy, max_flux_nJy):
        """
        Identify which flux bands are out of range for each row (vectorized).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with flux columns
        min_flux_nJy : float or None
            Minimum flux in nJy (lower bound)
        max_flux_nJy : float or None
            Maximum flux in nJy (upper bound)

        Returns
        -------
        dict
            Dictionary mapping row index to list of out-of-range bands
        """
        out_of_range_bands = {idx: [] for idx in df.index}

        for band in ["g", "r", "i", "z", "y", "j"]:
            flux_col = f"flux_{band}"
            if flux_col not in df.columns:
                continue

            # Vectorized check
            flux_vals = df[flux_col]
            is_finite = np.isfinite(flux_vals)
            is_out = np.zeros(len(flux_vals), dtype=bool)

            if min_flux_nJy is not None:
                is_out |= (flux_vals < min_flux_nJy) & is_finite
            if max_flux_nJy is not None:
                is_out |= (flux_vals > max_flux_nJy) & is_finite

            # Store band name for out-of-range rows
            out_indices = df.index[is_out]
            for idx in out_indices:
                out_of_range_bands[idx].append(band)

        return out_of_range_bands

    def show_results(self, df, validation_status):
        # reset title panes
        self.reset()

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
            self.error_pane.append(self.download_button_str)
            self.download_button_str.visible = True
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
            self.info_text_vals.object = (
                "<font size=4><u>Data ranges</u></font>\n\n"
                "<font size=3>All values of `ra`, `dec`, `priority`, `exptime`, `resolution`, and `reference_arm` "
                "satisfy the allowed ranges (see [documentation](doc/validation.html)).</font>"
            )
            self.info_pane.append(self.info_text_vals)
        elif not validation_status["values"]["status"]:
            self.append_title("error")
            self.error_text_vals.object = (
                "<font size=4><u>Value errors</u></font>\n\n"
                "<font size=3>Invalid values are detected for the following columns in the following entries "
                "(see [documentation](doc/validation.html)).</font>"
            )
            for k, v in zip(
                ["ra", "dec", "priority", "exptime", "resolution", "reference_arm"],
                [
                    "0 < `ra` < 360",
                    "-90 < `dec` < 90",
                    "[0, 9]",
                    "positive `float` value",
                    "`L` or `M`",
                    "`b`, `r`, `n`, or `m`, and `r` and `m` cannot be used with `resolution` of `M` and `L`, respectively",
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
            self.error_pane.append(self.download_button_vals)
            self.download_button_vals.visible = True
            self.error_pane.append(self.error_table_vals)
            self.error_table_vals.visible = True

        # If invalid values are detected, return immediately
        if not validation_status["values"]["status"]:
            return

        # flux columns
        # TODO: show a list of detected/undetected flux columns
        if validation_status["flux_columns"]["status"]:
            self.append_title("info")
            self.info_text_flux.object = "<font size=4><u>Flux information</u></font>\n\n<font size=3>All `ob_code`s have at least one flux information. The detected filters are the following: </font>"
            for f in validation_status["flux_columns"]["filters"]:
                self.info_text_flux.object += f"<font size=3>`{f}`</font>, "
            self.info_text_flux.object = self.info_text_flux.object[:-2]

            self.info_pane.append(self.info_text_flux)
            self.error_table_flux.visible = False
        else:
            self.append_title("error")
            # add an error message and data table for duplicates
            self.error_text_flux.object = "<font size=4><u>Missing flux information</u></font>\n\n<font size=3>No flux information found in the following `ob_code`s. Detected filters are the following: </font>"
            for f in validation_status["flux_columns"]["filters"]:
                self.error_text_flux.object += f"<font size=3>`{f}`</font>, "
            if len(validation_status["flux_columns"]["filters"]) > 0:
                self.error_text_flux.object = self.error_text_flux.object[:-2]

            self.error_table_flux.frozen_columns = []
            if self.error_table_flux.value is not None:
                self.error_table_flux.value[0:0]
            self.error_table_flux.value = df.loc[
                ~validation_status["flux_columns"]["success"], :
            ]
            self.error_table_flux.frozen_columns = ["index"]
            self.error_pane.append(self.error_text_flux)
            self.error_pane.append(self.download_button_flux)
            self.download_button_flux.visible = True
            self.error_pane.append(self.error_table_flux)
            self.error_table_flux.visible = True

        # Flux values (only check if flux columns were successfully detected)
        if (
            validation_status["flux_columns"]["status"]
            and validation_status["flux_values"]["status"]
        ):
            # Ensure info_text_flux is visible in the info pane before appending text
            if self.info_text_flux not in self.info_pane:
                self.append_title("info")
                # Provide a basic header if none was set by the flux-columns section
                if not getattr(self.info_text_flux, "object", None):
                    self.info_text_flux.object = (
                        "<font size=4><u>Flux information</u></font>"
                    )
                self.info_pane.append(self.info_text_flux)
            self.info_text_flux.object += "\n\n<font size=3>Flux values can be regarded as properly provided with the unit of nano Jansky (nJy).</font>"
        elif validation_status["flux_columns"]["status"]:
            self.append_title("warning")
            self.warning_text_flux.object = (
                "<font size=4><u>Suspicious flux values</u></font>\n\n"
            )
            self.warning_text_flux.object += (
                "<font size=3>Significant fraction "
                f"({validation_status['flux_values']['frac_suspicious_flux_all']:.2%}) "
                "of flux values are in the suspicious range "
                f"({validation_status['flux_values']['min_flux']} and {validation_status['flux_values']['max_flux']}). "
                "Please verify that the flux values are in the unit of nano Jansky (nJy), not, e.g., magnitude.</font>"
            )

            self.warning_pane.append(self.warning_text_flux)

        # Flux range (AB magnitude based - only check if flux columns were successfully detected)
        if (
            validation_status["flux_columns"]["status"]
            and "flux_range" in validation_status
            and validation_status["flux_range"]["status"] is not None
        ):
            if validation_status["flux_range"]["status"]:
                # Success case: all flux values within range
                min_mag = validation_status["flux_range"]["min_mag"]
                max_mag = validation_status["flux_range"]["max_mag"]

                # Create range description
                if min_mag is not None and max_mag is not None:
                    range_desc = f"AB magnitude range [{min_mag}, {max_mag}]"
                elif min_mag is not None:
                    range_desc = f"not brighter than AB magnitude {min_mag}"
                elif max_mag is not None:
                    range_desc = f"not fainter than AB magnitude {max_mag}"
                else:
                    range_desc = "within expected range"

                # Ensure info_text_flux is in info_pane before appending
                if self.info_text_flux not in self.info_pane:
                    self.append_title("info")
                    self.info_pane.append(self.info_text_flux)

                self.info_text_flux.object += (
                    f"\n\n<font size=3>All flux values are {range_desc}.</font>"
                )
            else:
                self.append_title("warning")
                min_mag = validation_status["flux_range"]["min_mag"]
                max_mag = validation_status["flux_range"]["max_mag"]
                num_out = validation_status["flux_range"]["num_out_of_range_flux"]
                num_total = validation_status["flux_range"]["num_total_flux"]

                # Create range description
                if min_mag is not None and max_mag is not None:
                    range_desc = f"AB magnitude range [{min_mag}, {max_mag}]"
                elif min_mag is not None:
                    range_desc = f"brighter than AB magnitude {min_mag}"
                elif max_mag is not None:
                    range_desc = f"fainter than AB magnitude {max_mag}"
                else:
                    range_desc = "specified range"

                self.warning_text_fluxrange.object = "<font size=4><u>Flux values out of AB magnitude range</u></font>\n\n"
                self.warning_text_fluxrange.object += (
                    f"<font size=3>{num_out}/{num_total} flux values are {range_desc}. "
                    "Please verify that these targets are intended to be outside this range. "
                    "See the <a href='https://www.naoj.org/Instruments/PFS/observations.html' target='_blank'>Observations</a> section of the PFS instrument documentation for more details.</font>"
                )

                # Show table of out-of-range targets with flux and magnitude columns
                success_all = validation_status["flux_range"]["success"]
                df_out_of_range = df.loc[~success_all, :].copy()

                # Get flux limits for identifying out-of-range bands
                min_flux_nJy = validation_status["flux_range"]["min_flux_nJy"]
                max_flux_nJy = validation_status["flux_range"]["max_flux_nJy"]

                # Identify which bands are out of range for each row
                out_of_range_bands = self._identify_out_of_range_bands(
                    df_out_of_range, min_flux_nJy, max_flux_nJy
                )

                # Add magnitude columns for each flux band
                for band in ["g", "r", "i", "z", "y", "j"]:
                    flux_col = f"flux_{band}"
                    mag_col = f"mag_{band}"
                    if flux_col in df_out_of_range.columns:
                        # Convert flux (nJy) to AB magnitude
                        flux_vals = df_out_of_range[flux_col].values
                        mag_vals = np.full_like(flux_vals, np.nan, dtype=float)
                        finite_mask = np.isfinite(flux_vals) & (flux_vals > 0)
                        if np.any(finite_mask):
                            mag_vals[finite_mask] = (
                                flux_vals[finite_mask] * u.nJy
                            ).to_value(u.ABmag)
                        df_out_of_range[mag_col] = mag_vals

                # Select columns to display
                base_cols = ["ob_code", "obj_id_str", "ra", "dec"]
                flux_mag_cols = []
                for band in ["g", "r", "i", "z", "y", "j"]:
                    flux_col = f"flux_{band}"
                    filter_col = f"filter_{band}"
                    mag_col = f"mag_{band}"
                    if flux_col in df_out_of_range.columns:
                        if filter_col in df_out_of_range.columns:
                            flux_mag_cols.extend([filter_col, flux_col, mag_col])
                        else:
                            flux_mag_cols.extend([flux_col, mag_col])

                display_cols = base_cols + flux_mag_cols
                available_cols = [
                    col for col in display_cols if col in df_out_of_range.columns
                ]

                # Keep numeric dtype intact for clean CSV export.
                # Display precision (2 d.p.) is handled by Tabulator formatters below.
                df_display = df_out_of_range[available_cols].copy()

                # Apply a Pandas Styler to highlight out-of-range cells with an
                # amber tint, matching the warning severity of this section.
                # Using a Styler keeps float values in .value.data, which the
                # download callback reads to produce a clean CSV.
                def _highlight_out_of_range(df, bands_map=out_of_range_bands):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for row_idx, bands in bands_map.items():
                        if bands and row_idx in styles.index:
                            for band in bands:
                                for col in (
                                    f"filter_{band}",
                                    f"flux_{band}",
                                    f"mag_{band}",
                                ):
                                    if col in styles.columns:
                                        styles.at[row_idx, col] = (
                                            "background-color: #fff3cd;"
                                            " color: #856404;"
                                            " font-weight: bold;"
                                        )
                    return styles

                # Configure Tabulator formatters for numeric display precision
                # (2 decimal places).  formatters only affect rendering, not .value.
                formatters = {}
                for band in ["g", "r", "i", "z", "y", "j"]:
                    for col in (f"flux_{band}", f"mag_{band}"):
                        if col in available_cols:
                            formatters[col] = NumberFormatter(format="0.00")
                self.warning_table_fluxrange.formatters = formatters

                self.warning_table_fluxrange.frozen_columns = []
                if self.warning_table_fluxrange.value is not None:
                    self.warning_table_fluxrange.value[0:0]

                # Build the CSV export DataFrame: same as df_display but with an
                # extra ``out_of_range_bands`` column that lists which bands are
                # out of range for each row (e.g. "g,r"), so users can identify
                # flagged values after downloading without losing numeric dtypes.
                out_of_range_str = pd.Series("", index=df_display.index, dtype=str)
                for row_idx, bands in out_of_range_bands.items():
                    if bands and row_idx in out_of_range_str.index:
                        out_of_range_str.at[row_idx] = ",".join(sorted(bands))
                df_for_csv = df_display.copy()
                df_for_csv["out_of_range_bands"] = out_of_range_str
                self._df_fluxrange_csv = df_for_csv

                # Set .style BEFORE .value.  Panel's _validate watcher copies the
                # _todo list (containing _highlight_out_of_range) to the Styler
                # built from df_display when value changes.  Setting .value directly
                # to a Styler breaks pagination='remote' (_get_data calls df.iloc[]).
                self.warning_table_fluxrange.style = df_display.style.apply(
                    _highlight_out_of_range, axis=None
                )
                self.warning_table_fluxrange.value = df_display
                self.warning_table_fluxrange.frozen_columns = ["index"]
                self.warning_pane.append(self.warning_text_fluxrange)
                self.warning_pane.append(self.download_button_fluxrange)
                self.download_button_fluxrange.visible = True
                self.warning_pane.append(self.warning_table_fluxrange)
                self.warning_table_fluxrange.visible = True

        # Visibility
        # TODO: add begin_date and end_date in the message
        if validation_status["visibility"]["status"]:
            if np.all(validation_status["visibility"]["success"]):
                self.append_title("info")
                self.info_text_visibility.object = (
                    "<font size=4><u>Visibility</u></font>\n\n"
                    "<font size=3>All `ob_code`s are visible in the input observing period.</font>"
                )
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
                self.warning_pane.append(self.download_button_visibility)
                self.download_button_visibility.visible = True
                self.warning_pane.append(self.warning_table_visibility)
                self.warning_table_visibility.visible = True
            self.error_table_visibility.visible = False
        else:
            self.append_title("error")
            # add an error message and data table for duplicates
            self.error_text_visibility.object = (
                "<font size=4><u>Visibility</u></font>\n\n<font size='3'>"
                "None of `ob_code`s in the list is visible in the input observing period.</font>"
            )
            self.error_pane.append(self.error_text_visibility)

        # Duplication
        if validation_status["unique"]["status"]:
            self.append_title("info")
            self.info_text_dups.object = (
                "<font size=4><u>Uniqueness of `ob_code` and `(obj_id, resolution)`</u></font>\n\n"
                "<font size=3>All `ob_code` and `(obj_id, resolution)` are unique.</font>"
            )
            self.info_pane.append(self.info_text_dups)
            self.error_table_dups.visible = False
        else:
            self.append_title("error")
            # add an error message and data table for duplicates
            self.error_text_dups.object = (
                "<font size=4><u>Duplication of `ob_code` and `(obj_id, resolution)` </u></font>\n\n"
                "<font size=3>Each `ob_code` and `(obj_id, resolution)` must be unique within a proposal, "
                "but duplicate `ob_code` and/or `(obj_id, resolution)` are detected in the following targets.</font>"
            )
            self.error_table_dups.frozen_columns = []
            if self.error_table_dups.value is not None:
                self.error_table_dups.value[0:0]
            self.error_table_dups.value = df.loc[
                validation_status["unique"]["flags"], :
            ]
            self.error_pane.append(self.error_text_dups)
            self.error_pane.append(self.download_button_dups)
            self.download_button_dups.visible = True
            self.error_pane.append(self.error_table_dups)
            self.error_table_dups.frozen_columns = ["index"]
            self.error_table_dups.visible = True

        # internal duplication
        if validation_status["internal_duplication"]["status"] is None:
            pass
        elif validation_status["internal_duplication"]["status"]:
            self.append_title("info")
            self.info_text_intdups.object = (
                "<font size=4><u>Internal duplication by coordinate</u></font>"
                "\n\n<font size=3>No internal duplication is detected.</font>"
            )
            self.info_pane.append(self.info_text_intdups)
            self.warning_table_intdups.visible = False
        else:
            self.append_title("warning")
            # add a warning message and data table for internal duplicates
            self.warning_text_intdups.object = (
                "<font size=4><u>Internal duplication</u></font>\n\n"
                "<font size=3>Targets with identical coordinates or with nearby coordinates with the same resolution mode are detected in the following targets. "
                "Please verify if these targets are not duplicates.</font>"
            )
            self.warning_table_intdups.frozen_columns = []
            if self.warning_table_intdups.value is not None:
                self.warning_table_intdups.value[0:0]

            # Add nn_sep column to duplicated targets
            df_intdups = df.loc[
                validation_status["internal_duplication"]["flags"], :
            ].copy()

            # Add nearest neighbor separation in arcsec
            # Use a Series indexed like df to ensure correct alignment even if df
            # has a non-sequential or non-zero-based index.
            nn_sep_array = validation_status["internal_duplication"]["nn_sep"]
            nn_sep_series = pd.Series(nn_sep_array, index=df.index)
            df_intdups["separation"] = nn_sep_series.loc[df_intdups.index].values

            # Sort by ra and dec for easier visual inspection of spatial clustering
            df_intdups = df_intdups.sort_values(by=["ra", "dec"])

            # Select and reorder columns for display
            display_columns = [
                "ob_code",
                "obj_id_str",
                "ra",
                "dec",
                "resolution",
                "reference_arm",
                "exptime",
                "separation",
            ]
            # Only include columns that exist in the dataframe
            available_columns = [
                col for col in display_columns if col in df_intdups.columns
            ]
            df_intdups_display = df_intdups[available_columns]

            self.warning_table_intdups.value = df_intdups_display
            self.warning_pane.append(self.warning_text_intdups)
            self.warning_pane.append(self.download_button_intdups)
            self.download_button_intdups.visible = True
            self.warning_pane.append(self.warning_table_intdups)
            self.warning_table_intdups.frozen_columns = []
            self.warning_table_intdups.visible = True

        # overall success
        if (
            validation_status["required_keys"]["status"]
            and validation_status["str"]["status"]
            and validation_status["values"]["status"]
            and validation_status["flux_columns"]["status"]
            # flux_values is warning-only, not included in success criteria
            and validation_status["visibility"]["status"]
            and validation_status["unique"]["status"]
        ):
            self.error_text_success.visible = False
