#!/usr/bin/env python3

# import collections
import glob
import math
import os
import secrets
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
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


def upload_file(df, origname=None, outdir=".", secret_token=None, upload_time=None):
    # convert pandas.DataFrame to astropy.Table
    tb = Table.from_pandas(df)

    if secret_token is None:
        secret_token = secrets.token_hex(8)
        logger.warning(
            f"secret_token {secret_token} is newly generated as None is provided."
        )

    # use the current UTC time and random hash string to construct an output filename
    if upload_time is None:
        upload_time = datetime.now(timezone.utc)
        logger.warning(
            f"upload_time {upload_time.isoformat(timespec='seconds')} is newly generated as None is provided."
        )

    # add metadata
    tb.meta["original_filename"] = origname
    tb.meta["upload_id"] = secret_token
    tb.meta["upload_at"] = upload_time.isoformat(timespec="seconds")

    # filename = f"{uploaded_time.strftime('%Y%m%d-%H%M%S')}_{secret_token}.ecsv"
    filename = (
        # f"targets_{uploaded_time.isoformat(timespec='seconds')}_{secret_token}.ecsv"
        f"targets_{secret_token}.ecsv"
    )

    logger.info(f"File `{filename}` was saved under `{outdir}`")

    # save the table in the output directory as an ECSV file
    tb.write(
        os.path.join(outdir, filename),
        delimiter=",",
        format="ascii.ecsv",
        overwrite=True,
    )

    return filename, upload_time, secret_token


def load_file_properties(dir=".", ext="ecsv"):
    config = dotenv_values(".env.shared")
    if dir == os.path.join(config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_data"]):
        files_in_dir = glob.glob(f"{dir}/*.{ext}")
        n_files = len(files_in_dir)
        filenames = np.zeros(n_files, dtype=object)
        fullpath = np.zeros(n_files, dtype=object)
        orignames = np.zeros(n_files, dtype=object)
        upload_ids = np.zeros(n_files, dtype=object)
        timestamps = np.zeros(n_files, dtype="datetime64[s]")
        filesizes = np.zeros(n_files, dtype=float)
        n_obj = np.zeros(n_files, dtype=int)
        t_exp = np.zeros(n_files, dtype=float)
        links = np.zeros(n_files, dtype=object)

        for i, f in enumerate(files_in_dir):
            if ext == "ecsv":
                tb = Table.read(f)
                fullpath[i] = f
                filenames[i] = os.path.basename(f)
                filesizes[i] = (os.path.getsize(f) * u.byte).to(u.kbyte).value
                links[i] = f"<a href='{f}'><i class='fa-solid fa-download'></i></a>"

                try:
                    orignames[i] = tb.meta["original_filename"]
                except KeyError:
                    orignames[i] = None

                try:
                    upload_ids[i] = tb.meta["upload_id"]
                except KeyError:
                    upload_ids[i] = None

                try:
                    timestamps[i] = datetime.fromisoformat(tb.meta["upload_at"])
                except KeyError:
                    timestamps[i] = None

                n_obj[i] = tb["ob_code"].size
                t_exp[i] = np.sum(tb["exptime"]) / 3600.0

        df = pd.DataFrame(
            {
                "upload_id": upload_ids,
                "filename": filenames,
                "n_obj": n_obj,
                "t_exp": t_exp,
                "origname": orignames,
                "filesize": filesizes,
                "timestamp": timestamps,
                "link": links,
                "fullpath": fullpath,
            }
        )
        return df.sort_values("timestamp", ascending=False, ignore_index=True)

    elif dir == os.path.join(config["OUTPUT_DIR_PREFIX"], config["OUTPUT_DIR_ppp"]):
        files_in_dir = glob.glob(f"{dir}/*.{ext}")
        n_files = len(files_in_dir)
        orignames = np.zeros(n_files, dtype=object)
        upload_ids = np.zeros(n_files, dtype=object)
        exp_sci_l = np.zeros(n_files, dtype=float)
        exp_sci_m = np.zeros(n_files, dtype=float)
        exp_sci_fh_l = np.zeros(n_files, dtype=float)
        exp_sci_fh_m = np.zeros(n_files, dtype=float)
        tot_time_l = np.zeros(n_files, dtype=float)
        tot_time_m = np.zeros(n_files, dtype=float)
        # timestamps = np.zeros(n_files, dtype="datetime64[s]")

        for i, f in enumerate(files_in_dir):
            if ext == "ecsv":
                tb = Table.read(f)
                orignames[i] = tb.meta["original_filename"]
                upload_ids[i] = tb.meta["upload_id"]
                # try:
                #     timestamps[i] = timestamps[i] = datetime.fromisoformat(
                #         tb.meta["upload_at"]
                #     )
                # except KeyError:
                #     timestamps[i] = None

                tb_l = tb[tb["resolution"] == "low"]
                tb_m = tb[tb["resolution"] == "medium"]

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

        df = pd.DataFrame(
            {
                "Upload ID": upload_ids,
                "Filename": orignames,
                "Exptime_sci_L (h)": exp_sci_l,
                "Exptime_sci_M (h)": exp_sci_m,
                "Exptime_sci_L (FH)": exp_sci_fh_l,
                "Exptime_sci_M (FH)": exp_sci_fh_m,
                "Time_tot_L (h)": tot_time_l,
                "Time_tot_M (h)": tot_time_m,
                # "timestamp": timestamps,
                # "Science category":
                # "Community":
            }
        )

        return df.sort_values("Upload ID", ascending=False, ignore_index=True)
        # return df_out.sort_values("timestamp", ascending=False, ignore_index=True)
