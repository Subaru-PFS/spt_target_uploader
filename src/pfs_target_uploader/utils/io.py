#!/usr/bin/env python3

# import collections
import glob
import math
import os
import secrets
import sys
import warnings
from datetime import datetime, timezone
from io import BytesIO, StringIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from bokeh.resources import INLINE
from dotenv import dotenv_values
from logzero import logger

from . import target_datatype

warnings.filterwarnings("ignore")


def load_input(byte_string, format="csv", dtype=target_datatype, logger=logger):
    def check_integer(value):
        try:
            int_value = int(value)
            if math.isclose(int_value, float(value)):
                return int_value
            else:
                raise ValueError(f"Non integer value detected: {value}")
        except (ValueError, TypeError):
            raise ValueError(f"Non integer value detected {value}")

    if format in ["csv", "ecsv"]:
        try:
            df_input = pd.read_csv(
                byte_string,
                encoding="utf8",
                comment="#",
                dtype=dtype,
                converters={
                    "ob_code": str,
                    "obj_id": check_integer,
                    "priority": check_integer,
                    "resolution": str,
                    "tract": check_integer,
                    "patch": check_integer,
                    "equinox": str,
                    "comment": str,
                },
            )
            load_status = True
            load_error = None
        except ValueError as e:
            df_input = None
            load_status = False
            load_error = e
    else:
        logger.error("Only CSV or ECSV format is supported at this moment.")
        return None, dict(status=False, error="No CSV or ECSV file selected")

    dict_load = dict(status=load_status, error=load_error)

    return df_input, dict_load


def upload_file(
    df_target,
    df_psl,
    df_ppc,
    df_target_summary,
    ppp_fig,
    outdir_prefix=".",
    origname="example.csv",
    origdata=None,
    secret_token=None,
    upload_time=None,
    ppp_status=True,
    export=False,
):
    # use the current UTC time and random hash string to construct an output filename
    if upload_time is None:
        upload_time = datetime.now(timezone.utc)
        logger.warning(
            f"upload_time {upload_time.isoformat(timespec='seconds')} is newly generated as None is provided."
        )
    dt = upload_time

    if export is False:
        if secret_token is None:
            secret_token = secrets.token_hex(8)
            logger.warning(
                f"secret_token {secret_token} is newly generated as None is provided."
            )

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
    else:
        secret_token = "export"
        outdir = None

    # convert pandas.DataFrame to astropy.Table
    tb_target = Table.from_pandas(df_target)
    tb_target_summary = Table.from_pandas(df_target_summary)
    if ppp_status:
        tb_psl = Table.from_pandas(df_psl)
        tb_ppc = Table.from_pandas(df_ppc)
    else:
        tb_psl = Table(
            {
                "resolution": [None],
                "N_ppc": [None],
                "Texp (h)": [np.nan],
                "Texp (fiberhour)": [np.nan],
                "Request time (h)": [np.nan],
                "Used fiber fraction (%)": [np.nan],
                "Fraction of PPC < 30% (%)": [np.nan],
                "P_all": [np.nan],
                "P_0": [np.nan],
            }
        )
        tb_ppc = Table(
            {
                "ppc_code": [None],
                "ppc_ra": [np.nan],
                "ppc_dec": [np.nan],
                "ppc_pa": [np.nan],
                "ppc_priority": [-1],
                "Fiber usage fraction (%)": [np.nan],
                "ppc_resolution": [None],
            }
        )

    outfiles_dict = {"filename": [], "object": [], "type": []}

    for file_prefix, tb in zip(
        ["target", "target_summary", "psl", "ppc"],
        [tb_target, tb_target_summary, tb_psl, tb_ppc],
    ):
        logger.info("Adding metadata")
        # add metadata
        tb.meta["original_filename"] = origname
        if export:
            tb.meta["upload_id"] = secret_token
            tb.meta["upload_at"] = upload_time
            tb.meta["ppp_status"] = ppp_status

        outfiles_dict["filename"].append(f"{file_prefix}_{secret_token}.ecsv")
        outfiles_dict["object"].append(tb)
        outfiles_dict["type"].append("table")

    if ppp_status:
        outfiles_dict["filename"].append(f"ppp_figure_{secret_token}.html")
        outfiles_dict["object"].append(ppp_fig)
        outfiles_dict["type"].append("figure")

    outfiles_dict["filename"].append(origname)
    outfiles_dict["object"].append(origdata)
    outfiles_dict["type"].append("original")

    if export:
        outfile_zip_prefix = f"pfs_target-{dt:%Y%m%d-%H%M%S}"
    else:
        outfile_zip_prefix = f"pfs_target-{dt:%Y%m%d-%H%M%S}-{secret_token}"

    outdir, outfile_zip, sio = upload_write(
        outfiles_dict, outfile_zip_prefix, outdir, export=export
    )

    return outdir, outfile_zip, sio


def upload_write(outfiles_dict, outfile_zip_prefix, outdir, export=False):
    if export is False:
        for i in range(len(outfiles_dict["filename"])):
            if outfiles_dict["type"][i] == "table":
                outfiles_dict["object"][i].write(
                    os.path.join(outdir, outfiles_dict["filename"][i]),
                    delimiter=",",
                    format="ascii.ecsv",
                    overwrite=True,
                )
            elif outfiles_dict["type"][i] == "figure":
                outfiles_dict["object"][i].save(
                    os.path.join(outdir, outfiles_dict["filename"][i]),
                    resources=INLINE,
                    title="Pointing Result",
                )
            elif outfiles_dict["type"][i] == "original":
                with open(
                    os.path.join(outdir, outfiles_dict["filename"][i]), "wb"
                ) as f:
                    f.write(outfiles_dict["object"][i])
            logger.info(f"File {outfiles_dict['filename'][i]} is saved under {outdir}.")

        # put all in a zip file
        # outfile_zip_prefix = f"pfs_target-{dt:%Y%m%d-%H%M%S}-{secret_token}"
        with ZipFile(os.path.join(outdir, f"{outfile_zip_prefix}.zip"), "w") as zipfile:
            for outfile in outfiles_dict["filename"]:
                absname = os.path.join(outdir, outfile)
                arcname = os.path.join(outfile_zip_prefix, os.path.basename(outfile))
                zipfile.write(absname, arcname=arcname)
        logger.info(f"Zip all output files in {outfile_zip_prefix}.zip in {outdir}")
        return outdir, f"{outfile_zip_prefix}.zip", None

    elif export:
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zipfile:
            for i in range(len(outfiles_dict["filename"])):
                buffer = StringIO()
                arcname = os.path.join(
                    outfile_zip_prefix,
                    os.path.basename(outfiles_dict["filename"][i]),
                )
                logger.info(
                    f"zipping with {outfiles_dict['filename'][i]} and {arcname}"
                )
                if outfiles_dict["type"][i] == "table":
                    outfiles_dict["object"][i].write(
                        buffer, format="ascii.ecsv", delimiter=","
                    )
                    zipfile.writestr(arcname, buffer.getvalue())
                if outfiles_dict["type"][i] == "figure":
                    outfiles_dict["object"][i].save(
                        buffer,
                        resources=INLINE,
                        title="Pointing Result",
                    )
                    zipfile.writestr(arcname, buffer.getvalue())
                if outfiles_dict["type"][i] == "original":
                    zipfile.writestr(arcname, outfiles_dict["object"][i])
        logger.info("zip file made")
        zip_buffer.seek(0)
        return None, f"{outfile_zip_prefix}.zip", zip_buffer


def load_file_properties(datadir, ext="ecsv", n_uid=16):
    dirs = glob.glob(f"{datadir}/????/??/*")
    n_files = len(dirs)

    orignames = np.full(n_files, None, dtype=object)
    upload_ids = np.full(n_files, None, dtype=object)
    timestamps = np.full(n_files, None, dtype="datetime64[s]")
    filesizes = np.zeros(n_files, dtype=float)
    n_obj = np.zeros(n_files, dtype=int)
    t_exp = np.zeros(n_files, dtype=float)
    links = np.full(n_files, None, dtype=object)

    fullpath_target = np.full(n_files, None, dtype=object)
    fullpath_psl = np.full(n_files, None, dtype=object)

    exp_sci_l = np.zeros(n_files, dtype=float)
    exp_sci_m = np.zeros(n_files, dtype=float)
    exp_sci_fh_l = np.zeros(n_files, dtype=float)
    exp_sci_fh_m = np.zeros(n_files, dtype=float)
    tot_time_l = np.zeros(n_files, dtype=float)
    tot_time_m = np.zeros(n_files, dtype=float)

    for i, d in enumerate(dirs):
        uid = d[-n_uid:]
        if ext == "ecsv":
            f_target = os.path.join(d, f"target_{uid}.{ext}")
            f_psl = os.path.join(d, f"psl_{uid}.{ext}")

            try:
                tb_target = Table.read(f_target)
                tb_psl = Table.read(f_psl)
                # logger.info(f"{f_target} and {f_psl} are found")
            except FileNotFoundError as e:
                logger.warning(
                    f"{e}: {f_target} and/or {f_psl} are not found. Skip them"
                )
                continue

            filesizes[i] = (os.path.getsize(f_target) * u.byte).to(u.kbyte).value
            # links[i] = f"<a href='{f_target}'><i class='fa-solid fa-download'></i></a>"

            fullpath_target[i] = f_target
            fullpath_psl[i] = f_psl

            try:
                orignames[i] = tb_target.meta["original_filename"]
            except KeyError:
                orignames[i] = None

            try:
                upload_ids[i] = tb_target.meta["upload_id"]
            except KeyError:
                upload_ids[i] = None

            try:
                if isinstance(tb_target.meta["upload_at"], str):
                    timestamps[i] = datetime.fromisoformat(tb_target.meta["upload_at"])
                elif isinstance(tb_target.meta["upload_at"], datetime):
                    timestamps[i] = tb_target.meta["upload_at"]
            except KeyError:
                timestamps[i] = None

            n_obj[i] = tb_target["ob_code"].size
            t_exp[i] = np.sum(tb_target["exptime"]) / 3600.0

            tb_l = tb_psl[tb_psl["resolution"] == "low"]
            tb_m = tb_psl[tb_psl["resolution"] == "medium"]

            if len(tb_l) > 0:
                exp_sci_l[i] = tb_l["Texp (h)"]
                exp_sci_fh_l[i] = tb_l["Texp (fiberhour)"]
                try:
                    tot_time_l[i] = tb_l["Request time (h)"]
                except KeyError:
                    tot_time_l[i] = tb_l["Request time 1 (h)"]

            if len(tb_m) > 0:
                exp_sci_m[i] = tb_m["Texp (h)"]
                exp_sci_fh_m[i] = tb_m["Texp (fiberhour)"]
                try:
                    tot_time_m[i] = tb_m["Request time (h)"]
                except KeyError:
                    tot_time_m[i] = tb_m["Request time 1 (h)"]

    df_target = pd.DataFrame(
        {
            "upload_id": upload_ids,
            "n_obj": n_obj,
            "t_exp": t_exp,
            "origname": orignames,
            "filesize": filesizes,
            "timestamp": timestamps,
            # "link": links,
            "fullpath": fullpath_target,
        }
    )

    df_psl = pd.DataFrame(
        {
            "Upload ID": upload_ids,
            # "Filename": orignames,
            "Exptime_sci_L (h)": exp_sci_l,
            "Exptime_sci_M (h)": exp_sci_m,
            "Exptime_sci_L (FH)": exp_sci_fh_l,
            "Exptime_sci_M (FH)": exp_sci_fh_m,
            "Time_tot_L (h)": tot_time_l,
            "Time_tot_M (h)": tot_time_m,
            "fullpath": fullpath_psl,
            # "timestamp": timestamps,
        }
    )

    return (
        df_target.sort_values("timestamp", ascending=False, ignore_index=True),
        df_psl.sort_values("Upload ID", ascending=False, ignore_index=True),
    )
