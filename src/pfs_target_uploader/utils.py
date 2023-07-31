#!/usr/bin/env python3

import glob
import os
import re
import secrets
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from logzero import logger

# import panel as pn

required_keys = [
    "obj_id",
    "ob_code",
    "ra",
    "dec",
    "equinox",
    "priority",
    "exptime",
    "resolution",
]

optional_keys = [
    "pmra",
    "pmdec",
    "parallax",
    "tract",
    "patch",
    # TODO: filters must be in the filter_name table in targetDB
    "filter_g",
    "filter_r",
    "filter_i",
    "filter_z",
    "filter_y",
    "filter_j",
    # TODO: fluxes can be fiber, psf, total, etc.
    "flux_g",
    "flux_r",
    "flux_i",
    "flux_z",
    "flux_y",
    "flux_j",
    "flux_error_g",
    "flux_error_r",
    "flux_error_i",
    "flux_error_z",
    "flux_error_y",
    "flux_error_j",
]

target_datatype = {
    # required keys
    "ob_code": str,
    "obj_id": np.int64,
    "ra": float,
    "dec": float,
    "equinox": str,
    "exptime": float,
    "priority": float,
    "resolution": str,
    "dummy": float,
    # optional keys
    "pmra": float,
    "pmdec": float,
    "parallax": float,
    "tract": int,
    "patch": int,
    "filter_g": str,
    "filter_r": str,
    "filter_i": str,
    "filter_z": str,
    "filter_y": str,
    "filter_j": str,
    "flux_g": float,
    "flux_r": float,
    "flux_i": float,
    "flux_z": float,
    "flux_y": float,
    "flux_j": float,
    "flux_error_g": float,
    "flux_error_r": float,
    "flux_error_i": float,
    "flux_error_z": float,
    "flux_error_y": float,
    "flux_error_j": float,
}

filter_names = [
    "g_hsc",
    "r_old_hsc",
    "r2_hsc",
    "i_old_hsc",
    "i2_hsc",
    "z_hsc",
    "y_hsc",
    "g_ps1",
    "r_ps1",
    "i_ps1",
    "z_ps1",
    "y_ps1",
    "bp_gaia",
    "rp_gaia",
    "g_gaia",
    "u_sdss",
    "g_sdss",
    "r_sdss",
    "i_sdss",
    "z_sds",
]


def load_input(byte_string, format="csv", dtype=target_datatype, logger=logger):
    if format == "csv":
        try:
            df_input = pd.read_csv(byte_string, encoding="utf8", dtype=dtype)
            load_status = True
            load_error = None
        except ValueError as e:
            df_input = None
            load_status = False
            load_error = e
    else:
        logger.error("Only CSV format is supported at this moment.")
        return None, None, None

    dict_load = dict(status=load_status, error=load_error)

    return df_input, dict_load


def check_keys(
    df, required_keys=required_keys, optional_keys=optional_keys, logger=logger
):
    required_status = []
    optional_status = []

    required_desc_success = []
    required_desc_error = []
    optional_desc_success = []
    optional_desc_warning = []

    for k in required_keys:
        if k in df.columns:
            desc = f"Required key `{k}` is found"
            required_status.append(True)
            required_desc_success.append(desc)
            logger.info(desc)
        else:
            desc = f"Required key `{k}` is missing"
            required_status.append(False)
            required_desc_error.append(desc)
            logger.error(desc)

    for k in optional_keys:
        if k in df.columns:
            desc = f"Optional key `{k}` is found"
            optional_status.append(True)
            optional_desc_success.append(desc)
            logger.info(desc)
        else:
            desc = f"Optional key `{k}` is missing"
            optional_status.append(False)
            optional_desc_warning.append(desc)
            logger.warn(desc)

    dict_required_keys = dict(
        status=np.all(required_status),  # True for success
        desc_success=required_desc_success,
        desc_error=required_desc_error,
    )
    dict_optional_keys = dict(
        status=np.all(optional_status),  # True for success
        desc_success=optional_desc_success,
        desc_warning=optional_desc_warning,
    )

    return dict_required_keys, dict_optional_keys


def check_str(
    df,
    required_keys=required_keys,
    optional_keys=optional_keys,
    dtype=target_datatype,
    logger=logger,
):
    # TODO: I guess validation of datatypes for float and integer numbers can be skipped
    # because pd.read_csv() raises an error.
    # Possible checks are:
    # - sanity check for string columns to prevent unexpected behavior in the downstream
    #   such as SQL injection. Maybe limit the string to [A-Za-z0-9_+-.]?

    dict_str = {}

    # Allow only [A-Za-z0-9] and _+-. for string values. I hope this is sufficient.
    pattern = r"^[A-Za-z0-9_+\-\.]+$"

    def check_pattern(element):
        return bool(re.match(pattern, element))

    vectorized_check = np.vectorize(check_pattern)

    is_success = True
    is_optional_success = True
    success_required_keys = np.ones(df.index.size, dtype=bool)
    success_optional_keys = np.ones(df.index.size, dtype=bool)

    for k in required_keys:
        if (k in df.columns) and (dtype[k] is str):
            is_match = vectorized_check(df[k].to_numpy())
            # True for good value; False for violation
            dict_str[f"status_{k}"] = np.all(is_match)
            dict_str[f"success_{k}"] = is_match
            success_required_keys = np.logical_and(success_required_keys, is_match)
            is_success = is_success and np.all(is_match)

    for k in optional_keys:
        if (k in df.columns) and (dtype[k] is str):
            is_match = vectorized_check(df[k].to_numpy())
            # True for good value; False for violation
            dict_str[f"status_{k}"] = np.all(is_match)
            dict_str[f"success_{k}"] = is_match
            success_optional_keys = np.logical_and(success_optional_keys, is_match)
            is_optional_success = is_optional_success and np.all(is_match)

    dict_str["status"] = is_success
    dict_str["status_optional"] = is_optional_success
    dict_str["success_required"] = success_required_keys
    dict_str["success_optional"] = success_optional_keys

    return dict_str


def check_values(df, logger=logger):
    # TODO: check data range including:
    # - ra must be in 0 to 360
    # - dec must be in -90 to 90
    # - equinox must be up to seven character string starting with "J" or "B"
    # - priority must be positive
    # - exptime must be positive
    # - resolution must be 'L' or 'M'
    #
    # - filters must be in targetdb
    # - fluxes must be positive
    #

    # Required keys
    is_ra = np.logical_and(df["ra"] >= 0.0, df["ra"] <= 360.0)
    is_dec = np.logical_and(df["dec"] >= -90.0, df["dec"] <= 90.0)

    is_priority = df["priority"] >= 0.0
    is_exptime = df["exptime"] > 0.0
    is_resolution = np.logical_or(df["resolution"] == "L", df["resolution"] == "M")

    def check_equinox(e):
        # check if an equinox string starts with "J" or "B"
        is_epoch = (e[0] == "J") or (e[0] == "B")
        # check if the rest of the input can be converted to a float value
        # Here I don't check if it's in a reasonable range or not.
        # TODO: We may make the equinox optional (J2000.0), need some discussion with obsproc.
        try:
            _ = float(e[1:])
            is_year = True
        except ValueError:
            is_year = False
        return is_epoch and is_year

    vectorized_check_equinox = np.vectorize(check_equinox)
    is_equinox = vectorized_check_equinox(df["equinox"].to_numpy())

    dict_values = {}
    is_success = True
    success_all = np.ones(df.index.size, dtype=bool)  # True if success
    for k, v in zip(
        ["ra", "dec", "equinox", "priority", "exptime", "resolution"],
        [is_ra, is_dec, is_equinox, is_priority, is_exptime, is_resolution],
    ):
        dict_values[f"status_{k}"] = np.all(v)
        dict_values[f"success_{k}"] = v
        is_success = is_success and np.all(v)
        success_all = np.logical_and(success_all, v)
    dict_values["status"] = is_success
    dict_values["success"] = success_all

    # shall we check values for optional fields?

    return dict_values


def check_unique(df, logger=logger):
    # if the dataframe is None or empty, skip validation
    if df is None or df.empty:
        unique_status = False
        flag_duplicate = None
        description = "Empty data detected (maybe failure in loading the inputs)"
        return dict(status=unique_status, flags=flag_duplicate, description=description)

    # make a status flag for duplication check
    flag_duplicate = np.zeros(df.index.size, dtype=bool)
    # find unique elements in 'ob_code'
    unique_elements, unique_counts = np.unique(
        df["ob_code"].to_numpy(), return_counts=True
    )

    # If the number of unique elements is identical to that of the size of the dataframe,
    # 'success' status is returned.
    if unique_elements.size == df.index.size:
        unique_status = True
        description = "All 'ob_code' entries are unique."
        logger.info("All 'ob_code' are unique.")
    else:
        # If duplicates are detected, flag elements is switched to True
        idx_dup = unique_counts > 1
        for dup in unique_elements[idx_dup]:
            flag_duplicate[df["ob_code"] == dup] = True
        unique_status = False
        description = "Duplicate 'ob_code' found. 'ob_code' must be unique."
        logger.error("Duplicates in 'ob_code' detected!")
        logger.error(f"""Duplicates by flag:\n{df.loc[flag_duplicate,:]}""")

    return dict(status=unique_status, flags=flag_duplicate, description=description)


def validate_input(df, logger=logger):
    validation_status = {}

    validation_status["status"] = False

    # check mandatory columns
    logger.info("[STAGE 1] Checking column names")
    dict_required_keys, dict_optional_keys = check_keys(df)
    logger.info(
        f"[STAGE 1] required_keys status: {dict_required_keys['status']} (Success if True)"
    )
    logger.info(
        f"[STAGE 1] optional_keys status: {dict_required_keys['status']} (Success if True)"
    )
    validation_status["required_keys"] = dict_required_keys
    validation_status["optional_keys"] = dict_optional_keys

    if not dict_required_keys["status"]:
        validation_status["str"] = {"status": None}
        validation_status["values"] = {"status": None}
        validation_status["unique"] = {"status": None}
        return validation_status

    # check string values
    logger.info("[STAGE 2] Checking string values")
    dict_str = check_str(df)
    logger.info(f"[STAGE 2] status: {dict_str['status']} (Success if True)")
    validation_status["str"] = dict_str
    if not dict_str["status"]:
        validation_status["values"] = {"status": None}
        validation_status["unique"] = {"status": None}
        return validation_status

    # check value against allowed ranges
    logger.info("[STAGE 3] Checking wheter values are in allowed ranges")
    dict_values = check_values(df)
    logger.info(f"[STAGE 3] status: {dict_values['status']} (Success if True)")
    validation_status["values"] = dict_values
    if not dict_values["status"]:
        validation_status["unique"] = {"status": None}
        return validation_status

    # check unique constraint for `ob_code`
    logger.info("[STAGE 4] Checking whether all ob_code are unique")
    dict_unique = check_unique(df)
    logger.info(f"[STAGE 4] status: {dict_unique['status']} (Success if True)")
    validation_status["unique"] = dict_unique

    if (
        validation_status["required_keys"]["status"]
        and validation_status["str"]["status"]
        and validation_status["values"]["status"]
        and validation_status["unique"]["status"]
    ):
        validation_status["status"] = True

    return validation_status


def upload_file(df, origname=None, outdir="."):
    # convert pandas.DataFrame to astropy.Table
    tb = Table.from_pandas(df)

    # use the current UTC time and random hash string to construct an output filename
    uploaded_time = datetime.now(timezone.utc)
    secret_token = secrets.token_hex(8)

    # add metadata
    tb.meta["original_filename"] = origname
    tb.meta["upload_id"] = secret_token
    tb.meta["upload_at"] = uploaded_time.isoformat(timespec="seconds")

    # filename = f"{uploaded_time.strftime('%Y%m%d-%H%M%S')}_{secret_token}.ecsv"
    filename = (
        f"targets_{uploaded_time.isoformat(timespec='seconds')}_{secret_token}.ecsv"
    )

    logger.info(f"File `{filename}` was saved under `{outdir}`")

    # save the table in the output directory as an ECSV file
    tb.write(os.path.join(outdir, filename), delimiter=",", format="ascii.ecsv")

    return filename, uploaded_time, secret_token


def load_file_properties(dir=".", ext="ecsv"):
    files_in_dir = glob.glob(f"{dir}/*.{ext}")
    n_files = len(files_in_dir)
    filenames = np.zeros(n_files, dtype=object)
    fullpath = np.zeros(n_files, dtype=object)
    orignames = np.zeros(n_files, dtype=object)
    upload_ids = np.zeros(n_files, dtype=object)
    timestamps = np.zeros(n_files, dtype="datetime64[s]")
    filesizes = np.zeros(n_files, dtype=float)
    n_obj = np.empty(n_files, dtype=int)
    t_exp = np.empty(n_files, dtype=float)
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
            except:
                orignames[i] = None

            try:
                upload_ids[i] = tb.meta["upload_id"]
            except:
                upload_ids[i] = None

            try:
                timestamps[i] = tb.meta["upload_at"]
            except:
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
