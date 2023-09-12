#!/usr/bin/env python3

# import collections
import glob
import os
import random
import re
import secrets
import time
import warnings
from datetime import datetime, timezone
from functools import partial
from itertools import chain

# from collections import defaultdict
import colorcet as cc
import holoviews as hv
import hvplot.pandas  # need to run pandas.DataFrame.hvplot
import matplotlib.pyplot as plt
import multiprocess
import numpy as np
import pandas as pd
import panel as pn
import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from dateutil import parser, tz
from dotenv import dotenv_values

# from IPython.display import clear_output
from logzero import logger
from matplotlib.path import Path
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KernelDensity

# below for netflow
# isort: split
import ets_fiber_assigner.netflow as nf
from ics.cobraOps.Bench import Bench

# from ics.cobraOps import plotUtils
# from ics.cobraOps.cobraConstants import NULL_TARGET_ID, NULL_TARGET_POSITION
# from ics.cobraOps.CobrasCalibrationProduct import CobrasCalibrationProduct
# from ics.cobraOps.CollisionSimulator import CollisionSimulator
# from ics.cobraOps.TargetGroup import TargetGroup

# below for qplan
# isort: split
from qplan.entity import StaticTarget
from qplan.util.site import site_subaru as observer

warnings.filterwarnings("ignore")

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
    "z_sdss",
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


def upload_file(df, origname=None, outdir=".", secret_token=secrets.token_hex(8)):
    # convert pandas.DataFrame to astropy.Table
    tb = Table.from_pandas(df)

    # use the current UTC time and random hash string to construct an output filename
    uploaded_time = datetime.now(timezone.utc)

    # add metadata
    tb.meta["original_filename"] = origname
    tb.meta["upload_id"] = secret_token
    tb.meta["upload_at"] = uploaded_time.isoformat(timespec="seconds")

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

    return filename, uploaded_time, secret_token


def load_file_properties(dir=".", ext="ecsv"):
    config = dotenv_values(".env.shared")
    if dir == config["OUTPUT_DIR_data"]:
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

    elif dir == config["OUTPUT_DIR_ppp"]:
        files_in_dir = glob.glob(f"{dir}/*.{ext}")
        n_files = len(files_in_dir)
        orignames = np.zeros(n_files, dtype=object)
        upload_ids = np.zeros(n_files, dtype=object)
        exp_sci_l = np.empty(n_files, dtype=float)
        exp_sci_m = np.empty(n_files, dtype=float)
        exp_sci_fh_l = np.empty(n_files, dtype=float)
        exp_sci_fh_m = np.empty(n_files, dtype=float)
        tot_time_l = np.empty(n_files, dtype=float)
        tot_time_m = np.empty(n_files, dtype=float)

        for i, f in enumerate(files_in_dir):
            if ext == "ecsv":
                tb = Table.read(f)
                orignames[i] = tb.meta["original_filename"]
                upload_ids[i] = tb.meta["upload_id"]
                exp_sci_l[i] = tb[tb["resolution"] == "low"]["Texp (h)"]
                exp_sci_m[i] = tb[tb["resolution"] == "medium"]["Texp (h)"]
                exp_sci_fh_l[i] = tb[tb["resolution"] == "low"]["Texp (fiberhour)"]
                exp_sci_fh_m[i] = tb[tb["resolution"] == "medium"]["Texp (fiberhour)"]
                tot_time_l[i] = tb[tb["resolution"] == "low"]["Request time 1 (h)"]
                tot_time_m[i] = tb[tb["resolution"] == "medium"]["Request time 1 (h)"]

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
                # "Science category":
                # "Community":
            }
        )
        return df.sort_values("Upload ID", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------
# below for PPP online tool
# ---------------------------------------------------------------------


def PPPrunStart(uS, weight_para):
    def count_N(sample):
        """calculate local count of targets

        Parameters
        ==========
        sample : table

        Returns
        =======
        sample added with local density (bin_width is 1 deg in ra&dec)
        """
        # lower limit of dec is -40
        count_bin = [[0 for i in np.arange(0, 361, 1)] for j in np.arange(-40, 91, 1)]
        for ii in range(len(sample["ra"])):
            m = int(sample["ra"][ii])
            n = int(sample["dec"][ii] + 40)  # dec>-40
            count_bin[n][m] += 1
        den_local = [
            count_bin[int(sample["dec"][ii] + 40)][int(sample["ra"][ii])]
            for ii in range(len(sample["ra"]))
        ]

        if "local_count" not in sample.colnames:
            sample.add_column(den_local, name="local_count")
        else:
            sample["local_count"] = den_local

        return sample

    def weight(sample, conta, contb, contc):
        """calculate weights of targets (larger weights mean more important)

        Parameters
        ==========
        sample : table
        conta,contb,contc: float
            parameters of weighting scheme: conta--> science grade,>0; contb--> remaining time; contc--> local density

        Returns
        =======
        sample: table added with weight col
        """
        weight_t = (
            pow(conta, 2.0 + 0.1 * (9 - sample["priority"]))
            * pow(sample["exptime_PPP"] / 900.0, contb)
            * pow(sample["local_count"], contc)
        )

        if "weight" not in sample.colnames:
            sample.add_column(weight_t, name="weight")
        else:
            sample["weight"] = weight_t

        return sample

    def target_DBSCAN(sample, sep=1.38):
        """separate pointings/targets into different groups

        Parameters
        ==========
        sample:table
        sep: float
            angular separation set to group, degree
        Print:boolean

        Returns
        =======
        list of pointing centers in different group
        """
        # haversine uses (dec,ra) in radian;
        db = DBSCAN(eps=np.radians(sep), min_samples=1, metric="haversine").fit(
            np.radians([sample["dec"], sample["ra"]]).T
        )

        labels = db.labels_
        unique_labels = set(labels)
        n_clusters = len(unique_labels)

        tgt_group = []

        for ii in range(n_clusters):
            tgt_t = sample[labels == ii]
            tgt_group.append(tgt_t)

        return tgt_group

    def target_collision(sample, sep=2 / 3600.0):
        """check targets collide with each other

        Parameters
        ==========
        sample:table
        sep: float
            angular separation set define collided targets, degree, default=2 arcsec
        Print:boolean

        Returns
        =======
        list of pointing centers in different group
        """
        # haversine uses (dec,ra) in radian;
        db = AgglomerativeClustering(
            distance_threshold=np.radians(sep),
            n_clusters=None,
            affinity="haversine",
            linkage="single",
        ).fit(np.radians([sample["dec"], sample["ra"]]).T)

        labels = db.labels_
        unique_labels = set(labels)
        labels_c = [lab for lab in unique_labels if list(labels).count(lab) > 1]

        if len(labels_c) == 0:
            index = np.arange(0, len(sample), 1)
            return index
        else:
            index = list(np.where(~np.in1d(labels, labels_c))[0]) + [
                np.where(np.in1d(labels, kk))[0][0] for kk in labels_c
            ]
            return sorted(index)

    def PFS_FoV(ppc_ra, ppc_dec, PA, sample, mode=None):
        """pick up targets in the pointing

        Parameters
        ==========
        ppc_ra,ppc_dec,PA : float
            ra,dec,PA of the pointing center
        sample : table
        mode: default=None
            if "KDE_collision", consider collision avoid in KDE part (separation=2 arcsec)


        Returns
        =======
        list of index of targets, which fall into the pointing, in the input sample
        """
        if len(sample) > 1 and mode == "KDE_collision":
            index = target_collision(sample)
            point = np.vstack((sample[index]["ra"], sample[index]["dec"])).T
        else:
            point = np.vstack((sample["ra"], sample["dec"])).T
        center = SkyCoord(ppc_ra * u.deg, ppc_dec * u.deg)

        # PA=0 along y-axis, PA=90 along x-axis, PA=180 along -y-axis...
        hexagon = center.directional_offset_by(
            [30 + PA, 90 + PA, 150 + PA, 210 + PA, 270 + PA, 330 + PA, 30 + PA] * u.deg,
            1.38 / 2.0 * u.deg,
        )
        ra_h = hexagon.ra.deg
        dec_h = hexagon.dec.deg

        # for pointings around RA~0 or 360, parts of it will move to the opposite side (e.g., [[1,0],[-1,0]] -->[[1,0],[359,0]])
        # correct for it
        ra_h_in = np.where(np.fabs(ra_h - ppc_ra) > 180)
        if len(ra_h_in[0]) > 0:
            if ra_h[ra_h_in[0][0]] > 180:
                ra_h[ra_h_in[0]] -= 360
            elif ra_h[ra_h_in[0][0]] < 180:
                ra_h[ra_h_in[0]] += 360

        polygon = Path([(ra_h[t], dec_h[t]) for t in range(len(ra_h))])
        index_ = np.where(polygon.contains_points(point))[0]

        return index_

    def KDE_xy(sample, X, Y):
        """calculate a single KDE

        Parameters
        ==========
        sample: table
        X,Y: grid to calculate KDE

        Returns
        =======
        Z: KDE estimate
        """
        values = np.vstack((np.deg2rad(sample["dec"]), np.deg2rad(sample["ra"])))
        kde = KernelDensity(
            bandwidth=np.deg2rad(1.38 / 2.0),
            kernel="linear",
            algorithm="ball_tree",
            metric="haversine",
        )
        kde.fit(values.T, sample_weight=sample["weight"])

        X1 = np.deg2rad(X)
        Y1 = np.deg2rad(Y)
        positions = np.vstack([Y1.ravel(), X1.ravel()])
        Z = np.reshape(np.exp(kde.score_samples(positions.T)), Y.shape)

        return Z

    def KDE(sample, multiProcesing):
        """define binning and calculate KDE

        Parameters
        ==========
        sample: table
        multiProcesing: boolean
            allow multiprocessing or not
            (n_thread set to be the maximal threads allowed in the machine)

        Returns
        =======
        ra_bin, dec_bin, significance of KDE over the field, ra of peak in KDE, dec of peak in KDE
        """
        if len(sample) == 1:
            # if only one target, set it as the peak
            return (
                sample["ra"].data[0],
                sample["dec"].data[0],
                np.nan,
                sample["ra"].data[0],
                sample["dec"].data[0],
            )
        else:
            # determine the binning for the KDE cal.
            # set a bin width of 0.5 deg in ra&dec if the sample spans over a wide area (>50 degree)
            # give some blank spaces in binning, otherwide KDE will be wrongly calculated
            ra_low = min(min(sample["ra"]) * 0.9, min(sample["ra"]) - 1)
            ra_up = max(max(sample["ra"]) * 1.1, max(sample["ra"]) + 1)
            dec_up = max(max(sample["dec"]) * 1.1, max(sample["dec"]) + 1)
            dec_low = min(min(sample["dec"]) * 0.9, min(sample["dec"]) - 1)

            if (max(sample["ra"]) - min(sample["ra"])) / 100 < 0.5 and (
                max(sample["dec"]) - min(sample["dec"])
            ) / 100 < 0.5:
                X_, Y_ = np.mgrid[ra_low:ra_up:101j, dec_low:dec_up:101j]
            elif (max(sample["dec"]) - min(sample["dec"])) / 100 < 0.5:
                X_, Y_ = np.mgrid[0:360:721j, dec_low:dec_up:101j]
            elif (max(sample["ra"]) - min(sample["ra"])) / 100 < 0.5:
                X_, Y_ = np.mgrid[ra_low:ra_up:101j, -40:90:261j]
            else:
                X_, Y_ = np.mgrid[0:360:721j, -40:90:261j]
            positions1 = np.vstack([Y_.ravel(), X_.ravel()])

            if multiProcesing:
                threads_count = round(multiprocess.cpu_count() / 2)
                thread_n = min(
                    threads_count, round(len(sample) * 0.5)
                )  # threads_count=10 in this machine

                with multiprocess.Pool(thread_n) as p:
                    dMap_ = p.map(
                        partial(KDE_xy, X=X_, Y=Y_), np.array_split(sample, thread_n)
                    )

                Z = sum(dMap_)

            else:
                Z = KDE_xy(sample, X_, Y_)

            # calculate significance level of KDE
            obj_dis_sig_ = (Z - np.mean(Z)) / np.std(Z)
            peak_pos = np.where(obj_dis_sig_ == obj_dis_sig_.max())

            peak_y = positions1[0, peak_pos[1][round(len(peak_pos[1]) * 0.5)]]
            peak_x = sorted(set(positions1[1, :]))[
                peak_pos[0][round(len(peak_pos[0]) * 0.5)]
            ]

            return X_, Y_, obj_dis_sig_, peak_x, peak_y

    def PPP_centers(sample_f, mutiPro, conta, contb, contc):
        """determine pointing centers

        Parameters
        ==========
        sample_f : table
        mutiPro : boolean
            allow multiprocessing to calculate KDE or not
        conta,contb,contc: float
            parameters of weighting scheme: conta--> science grade,>0;
                                            contb--> remaining time;
                                            contc--> local density

        Returns
        =======
        sample with list of pointing centers in meta
        """

        time_start = time.time()
        Nfiber = int(2394 - 200)  # 200 for calibrators
        sample_f = count_N(sample_f)
        sample_f = weight(sample_f, conta, contb, contc)

        peak = []

        for sample in target_DBSCAN(sample_f, 1.38):
            sample_s = sample[sample["exptime_PPP"] > 0]  # targets not finished

            while any(sample_s["exptime_PPP"] > 0):
                # -------------------------------
                ####peak_xy from KDE peak with weights
                X_, Y_, obj_dis_sig_, peak_x, peak_y = KDE(sample_s, mutiPro)

                # -------------------------------
                index_ = PFS_FoV(
                    peak_x, peak_y, 0, sample_s
                )  # all PA set to be 0 for simplicity

                if len(index_) > 0:
                    peak.append(
                        [len(peak), peak_x, peak_y, 0]
                    )  # ppc_id,ppc_ra,ppc_dec,ppc_PA=0

                else:
                    # add a small random shift so that it will not repeat over a blank position
                    while len(index_) == 0:
                        peak_x_t = peak_x + np.random.choice([0.15, -0.15, 0], 1)[0]
                        peak_y_t = peak_y + np.random.choice([0.15, -0.15, 0], 1)[0]
                        index_ = PFS_FoV(peak_x_t, peak_y_t, 0, sample_s)
                    peak.append(
                        [len(peak), peak_x_t, peak_y_t, 0]
                    )  # ppc_id,ppc_ra,ppc_dec,ppc_PA=0

                # -------------------------------
                if len(index_) > Nfiber:
                    index_ = random.sample(list(index_), Nfiber)
                sample_s["exptime_PPP"][
                    list(index_)
                ] -= 900  # targets in the PPC observed with 900 sec

                sample_s = sample_s[sample_s["exptime_PPP"] > 0]  # targets not finished
                sample_s = count_N(sample_s)
                sample_s = weight(sample_s, conta, contb, contc)

        sample_f.meta["PPC"] = np.array(peak)

        return sample_f

    def plot_KDE(sample_f):
        sample_g = target_DBSCAN(sample_f, 1.38, True)
        peak = sample_f.meta["PPC"]
        for sample in sample_g:
            plt.figure(figsize=(7, 5))
            plt.scatter(
                sample["ra"],
                sample["dec"],
                c=sample["priority"],
                marker="o",
                cmap="Paired_r",
                vmin=0,
                vmax=9,
                s=7,
                zorder=10,
            )
            plt.colorbar(label="User priority")
            PFS_FoV_plot(peak[:, 1], peak[:, 2], peak[:, 3], "orange", 0.5, "--")
            plt.xlim(min(sample["ra"]) - 1, max(sample["ra"]) + 1)
            plt.ylim(min(sample["dec"]) - 1, max(sample["dec"]) + 1)
            plt.xlabel("RA", fontsize=10)
            plt.ylabel("DEC", fontsize=10)
            plt.title
            plt.show()

    def point_DBSCAN(sample, Plot):
        """separate pointings into different group

        Parameters
        ==========
        sample:table
        Plot, Print:boolean

        Returns
        =======
        list of pointing centers in different group
        """
        ppc_xy = sample.meta["PPC"]

        # haversine uses (dec,ra) in radian;
        db = DBSCAN(eps=np.radians(1.38), min_samples=1, metric="haversine").fit(
            np.fliplr(np.radians(ppc_xy[:, [1, 2]]))
        )

        labels = db.labels_
        unique_labels = set(labels)
        n_clusters = len(unique_labels)

        if Plot:
            colors = sns.color_palette(cc.glasbey_warm, n_clusters)

        ppc_group = []

        for ii in range(n_clusters):
            ppc_t = ppc_xy[labels == ii]
            ppc_group.append(ppc_t)

            if Plot:
                xy = ppc_t[:, [1, 2]]
                for uu in xy:
                    PFS_FoV_plot(uu[0], uu[1], 0, colors[ii], 0.2, "-")
                    plt.plot(uu[0], uu[1], "o", mfc=colors[ii], mew=0, ms=5)
                plt.show()

        return ppc_group

    def sam2netflow(sample):
        """put targets to the format which can be read by netflow

        Parameters
        ==========
        sample : table

        Returns
        =======
        list of targets readable by netflow
        """
        targetL = []

        int_ = 0
        for tt in sample:
            id_, ra, dec, tm = (tt["ob_code"], tt["ra"], tt["dec"], tt["exptime_PPP"])
            targetL.append(nf.ScienceTarget(id_, ra, dec, tm, int_, "sci"))
            int_ += 1

        # for ii in range(50): #mock Fstars
        #    targetL.append(nf.CalibTarget('Fs_'+str(ii),0,0, "cal"))

        # for jj in range(150):#mock skys
        #    targetL.append(nf.CalibTarget('Sky_'+str(jj),0,0,"sky"))

        return targetL

    def NetflowPreparation(sample):
        """assign cost to each target

        Parameters
        ==========
        sample : sample

        Returns
        =======
        class of targets with costs
        """

        classdict = {}

        int_ = 0
        for ii in sample:
            classdict["sci_P" + str(int_)] = {
                "nonObservationCost": ii["weight"],
                "partialObservationCost": ii["weight"] * 1.5,
                "calib": False,
            }
            int_ += 1

        # classdict["sky"] = {"numRequired": 150,
        #                    "nonObservationCost": max(sample['weight'])*1., "calib": True}

        # classdict["cal"] = {"numRequired": 50,
        #                    "nonObservationCost": max(sample['weight'])*1., "calib": True}

        return classdict

    def cobraMoveCost(dist):
        """optional: penalize assignments where the cobra has to move far out"""
        return 0.1 * dist

    def netflowRun_single(Tel, sample):
        """run netflow (without iteration)

        Parameters
        ==========
        sample : sample
        Tel: PPC info (id,ra,dec,PA)

        Returns
        =======
        solution of Gurobi, PPC list
        """
        Telra = Tel[:, 1]
        Teldec = Tel[:, 2]
        Telpa = Tel[:, 3]

        bench = Bench(layout="full")
        tgt = sam2netflow(sample)
        classdict = NetflowPreparation(sample)
        otime = "2024-05-20T08:00:00Z"

        telescopes = []

        nvisit = len(Telra)
        for ii in range(nvisit):
            telescopes.append(nf.Telescope(Telra[ii], Teldec[ii], Telpa[ii], otime))
        tpos = [tele.get_fp_positions(tgt) for tele in telescopes]

        # optional: slightly increase the cost for later observations,
        # to observe as early as possible
        vis_cost = [0 for i in range(nvisit)]

        gurobiOptions = dict(
            seed=0,
            presolve=1,
            method=4,
            degenmoves=0,
            heuristics=0.8,
            mipfocus=0,
            mipgap=5.0e-2,
            LogToConsole=0,
        )

        # partially observed? no
        alreadyObserved = {}

        forbiddenPairs = [[] for i in range(nvisit)]

        # compute observation strategy
        prob = nf.buildProblem(
            bench,
            tgt,
            tpos,
            classdict,
            900,
            vis_cost,
            cobraMoveCost=cobraMoveCost,
            collision_distance=2.0,
            elbow_collisions=True,
            gurobi=True,
            gurobiOptions=gurobiOptions,
            alreadyObserved=alreadyObserved,
            forbiddenPairs=forbiddenPairs,
        )

        prob.solve()

        res = [{} for _ in range(min(nvisit, len(Telra)))]
        for k1, v1 in prob._vardict.items():
            if k1.startswith("Tv_Cv_"):
                visited = prob.value(v1) > 0
                if visited:
                    _, _, tidx, cidx, ivis = k1.split("_")
                    res[int(ivis)][int(tidx)] = int(cidx)

        return res, telescopes, tgt

    def netflowRun_nofibAssign(Tel, sample):
        """run netflow (with iteration)
            if no fiber assignment in some PPCs, shift these PPCs with 0.2 deg

        Parameters
        ==========
        sample : sample
        Tel: PPC info (id,ra,dec,PA)

        Returns
        =======
        solution of Gurobi, PPC list
        """
        res, telescope, tgt = netflowRun_single(Tel, sample)

        if sum(np.array([len(tt) for tt in res]) == 0) == 0:
            # All PPCs have fiber assignment
            return res, telescope, tgt

        else:
            # if there are PPCs with no fiber assignment
            index = np.where(np.array([len(tt) for tt in res]) == 0)[0]

            Tel_t = Tel[:]
            iter_1 = 0

            while len(index) > 0 and iter_1 < 3:
                # shift PPCs with 0.2 deg, but only run three iterations to save computational time
                # typically one iteration is enough
                for ind in index:
                    Tel_t[ind, 1] = Tel[ind, 1] + np.random.choice([-0.2, 0.2], 1)[0]
                    Tel_t[ind, 2] = Tel[ind, 2] + np.random.choice([-0.2, 0.2], 1)[0]

                res, telescope, tgt = netflowRun_single(Tel_t, sample)
                index = np.where(np.array([len(tt) for tt in res]) == 0)[0]

                iter_1 += 1

            return res, telescope, tgt

    def netflowRun(sample):
        """run netflow (with iteration and DBSCAN)

        Parameters
        ==========
        sample : sample

        Returns
        =======
        Fiber assignment in each PPC
        """
        time_start = time.time()

        ppc_g = point_DBSCAN(sample, False)  # separate ppc into different groups

        point_list = []
        point_c = 0

        for uu in range(len(ppc_g)):  # run netflow for each ppc group
            # only consider sample in the group
            sample_index = list(
                chain.from_iterable(
                    [
                        list(
                            PFS_FoV(
                                ppc_g[uu][iii, 1],
                                ppc_g[uu][iii, 2],
                                ppc_g[uu][iii, 3],
                                sample,
                            )
                        )
                        for iii in range(len(ppc_g[uu]))
                    ]
                )
            )
            if len(sample_index) == 0:
                continue
            sample_inuse = sample[list(set(sample_index))]

            res, telescope, tgt = netflowRun_nofibAssign(ppc_g[uu], sample_inuse)

            for i, (vis, tel) in enumerate(zip(res, telescope)):
                fib_eff_t = len(vis) / 2394.0 * 100

                # assigned targets in each ppc
                obj_allo_id = []
                for tidx, cidx in vis.items():
                    obj_allo_id.append(tgt[tidx].ID)

                # calculate the total weights in each ppc (smaller value means more important)
                if len(vis) == 0:
                    tot_weight = np.nan
                else:
                    tot_weight = 1 / sum(
                        sample[np.in1d(sample["ob_code"], obj_allo_id)]["weight"]
                    )

                point_list.append(
                    [
                        "Point_" + str(point_c + 1),
                        "Group_" + str(uu + 1),
                        tel._ra,
                        tel._dec,
                        tel._posang,
                        tot_weight,
                        fib_eff_t,
                        obj_allo_id,
                        sample["resolution"][0],
                    ]
                )
                point_c += 1

        point_t = Table(
            np.array(point_list, dtype=object),
            names=[
                "ppc_code",
                "group_id",
                "ppc_ra",
                "ppc_dec",
                "ppc_pa",
                "ppc_priority",
                "tel_fiber_usage_frac",
                "allocated_targets",
                "ppc_resolution",
            ],
            dtype=[
                np.str_,
                np.str_,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                object,
                np.str_,
            ],
        )

        return point_t

    def complete_ppc(sample, point_l):
        """examine the completeness fraction of the user sample

        Parameters
        ==========
        sample : table

        point_l: table of ppc information

        Returns
        =======
        sample with allocated time

        completion rate: in each user-defined priority + overall
        """
        sample.add_column(0, name="allocate_time")
        point_l_pri = point_l[
            point_l.argsort(keys="ppc_priority")
        ]  # sort ppc by its total priority == sum(weights of the assigned targets in ppc)

        # sub-groups of the input sample, catagarized by the user defined priority
        sub_l = sorted(list(set(sample["priority"])))
        n_sub = len(sub_l)
        count_sub = [len(sample)] + [sum(sample["priority"] == ll) for ll in sub_l]
        completeR = []  # count
        completeR_ = []  # percentage

        for ppc in point_l_pri:
            lst = np.where(np.in1d(sample["ob_code"], ppc["allocated_targets"]))[0]
            sample["allocate_time"].data[lst] += 900

            comp_s = np.where(sample["exptime_PPP"] == sample["allocate_time"])[0]
            comT_t = [len(comp_s)] + [
                sum(sample["priority"].data[comp_s] == ll) for ll in sub_l
            ]
            completeR.append(comT_t)
            completeR_.append(
                [comT_t[oo] / count_sub[oo] * 100 for oo in range(len(count_sub))]
            )

        return sample, np.array(completeR), np.array(completeR_), sub_l

    def netflow_iter(uS, obj_allo, conta, contb, contc):
        """iterate the total procedure to re-assign fibers to targets which have not been assigned
            in the previous/first iteration

        Parameters
        ==========
        uS: table
            sample with exptime>allocate_time

        obj_allo: table
            ppc information

        conta,contb,contc: float
            weight parameters

        PrintTF: boolean , default==True

        Returns
        =======
        table of ppc information after all targets are assigned
            # note that some targets in the dense region may need very long time to be assigned with fibers
            # if targets can not be successfully assigned with fibers in >5 iterations, then directly stop
            # if total number of ppc >200 (~5 nights), then directly stop
        """
        time_start = time.time()
        if sum(uS["allocate_time"] == uS["exptime_PPP"]) == len(uS):
            # remove ppc with no fiber assignment
            obj_allo.remove_rows(np.where(obj_allo["tel_fiber_usage_frac"] == 0)[0])
            return obj_allo

        else:
            #  select non-assigned targets --> PPC determination --> netflow --> if no fibre assigned: shift PPC
            iter_m2 = 0

            while any(uS["allocate_time"] < uS["exptime_PPP"]) and iter_m2 < 10:
                uS_t1 = uS[uS["allocate_time"] < uS["exptime_PPP"]]
                uS_t1["exptime_PPP"] = (
                    uS_t1["exptime_PPP"] - uS_t1["allocate_time"]
                )  # remained exposure time
                uS_t1.remove_column("allocate_time")

                uS_t2 = PPP_centers(uS_t1, True, conta, contb, contc)

                obj_allo_t = netflowRun(uS_t2)

                if len(obj_allo) > 200 or iter_m2 >= 10:
                    # stop if n_ppc>200
                    return obj_allo

                else:
                    obj_allo = vstack([obj_allo, obj_allo_t])
                    obj_allo.remove_rows(
                        np.where(obj_allo["tel_fiber_usage_frac"] == 0)[0]
                    )
                    uS = complete_ppc(uS_t2, obj_allo)[0]
                    iter_m2 += 1

            return obj_allo

    conta, contb, contc = weight_para

    exptime_ppp = np.ceil(uS["exptime"] / 900) * 900
    uS.add_column(exptime_ppp, name="exptime_PPP")

    uS_L = uS[uS["resolution"] == "L"]
    uS_M = uS[uS["resolution"] == "M"]

    if len(uS_L) > 0 and len(uS_M) == 0:
        uS_L_s2 = PPP_centers(uS_L, True, conta, contb, contc)
        obj_allo_L = netflowRun(uS_L_s2)
        uS_L2 = complete_ppc(uS_L_s2, obj_allo_L)[0]
        obj_allo_L_fin = netflow_iter(uS_L2, obj_allo_L, conta, contb, contc)

        uS_L_s2.remove_column("allocate_time")
        uS_L2, cR_L, cR_L_, sub_l = complete_ppc(uS_L_s2, obj_allo_L_fin)

        return uS_L2, cR_L, cR_L_, sub_l, obj_allo_L_fin, [], [], [], [], []

    if len(uS_M) > 0 and len(uS_L) == 0:
        uS_M_s2 = PPP_centers(uS_M, True, conta, contb, contc)
        obj_allo_M = netflowRun(uS_M_s2)
        uS_M2 = complete_ppc(uS_M_s2, obj_allo_M)[0]
        obj_allo_M_fin = netflow_iter(uS_M2, obj_allo_M, conta, contb, contc)

        uS_M_s2.remove_column("allocate_time")
        uS_M2, cR_M, cR_M_, sub_m = complete_ppc(uS_M_s2, obj_allo_M_fin)

        return [], [], [], [], [], uS_M2, cR_M, cR_M_, sub_m, obj_allo_M_fin

    if len(uS_L) > 0 and len(uS_M) > 0:
        uS_L_s2 = PPP_centers(uS_L, True, conta, contb, contc)
        obj_allo_L = netflowRun(uS_L_s2)
        uS_L2 = complete_ppc(uS_L_s2, obj_allo_L)[0]
        obj_allo_L_fin = netflow_iter(uS_L2, obj_allo_L, conta, contb, contc)

        uS_L_s2.remove_column("allocate_time")
        uS_L2, cR_L, cR_L_, sub_l = complete_ppc(uS_L_s2, obj_allo_L_fin)

        uS_M_s2 = PPP_centers(uS_M, True, conta, contb, contc)
        obj_allo_M = netflowRun(uS_M_s2)
        uS_M2 = complete_ppc(uS_M_s2, obj_allo_M)[0]
        obj_allo_M_fin = netflow_iter(uS_M2, obj_allo_M, conta, contb, contc)

        uS_M_s2.remove_column("allocate_time")
        uS_M2, cR_M, cR_M_, sub_m = complete_ppc(uS_M_s2, obj_allo_M_fin)

        return (
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
        )


def ppp_result(cR_l, sub_l, obj_allo_l, uS_L2, cR_m, sub_m, obj_allo_m, uS_M2):
    def overheads(n_sci_frame):
        # in seconds
        t_exp_sci: float = 900.0
        t_exp_bias: float = 0.0
        t_exp_dark: float = 900.0
        t_exp_flat: float = 10.0
        t_exp_arc: float = 10.0
        t_lamp_flat: float = 60.0
        t_lamp_arc: float = 60.0
        t_focus: float = 300.0
        t_overhead_misc: float = 60.0
        t_overhead_fiber: float = 180.0

        n_frame_bias: float = 10.0
        n_frame_dark: float = 10.0
        n_frame_flat: float = 10.0
        n_frame_arc: float = 10.0
        n_focus: float = 3.0

        # t_tot_bias = (t_exp_bias + t_overhead_misc) * n_frame_bias
        # t_tot_dark = (t_exp_dark + t_overhead_misc) * n_frame_dark

        # total time for calibration
        t_calib = (
            t_lamp_flat
            + (t_exp_flat + t_overhead_misc) * n_frame_flat
            + t_lamp_arc
            + (t_exp_arc + t_overhead_misc) * n_frame_arc
        )
        t_focus_tot = t_focus * n_focus

        # maximal number of pointings for each night
        n_sci_perNight = (10 * 3600 - t_calib - t_focus_tot) / (
            t_exp_sci + t_overhead_misc + t_overhead_fiber
        )

        # the required number of night
        n_sci_night = np.ceil(n_sci_frame / n_sci_perNight)

        # total required time
        Toverheads_tot_best = (t_calib + t_focus_tot) * n_sci_night + (
            t_exp_sci + t_overhead_misc + t_overhead_fiber
        ) * n_sci_frame

        Toverheads_tot_worst = (
            t_overhead_fiber + t_calib + t_exp_sci + t_overhead_misc + t_overhead_fiber
        ) * n_sci_frame + t_focus_tot * n_sci_night

        return Toverheads_tot_best / 3600.0, Toverheads_tot_worst / 3600.0

    def ppp_plotFig(RESmode, cR, sub, obj_allo, uS):
        nppc = pn.widgets.IntSlider(
            name="You can modify the number of pointings for the "
            + RESmode
            + " resolution",
            value=len(cR),
            step=1,
            start=1,
            end=len(cR),
            bar_color="gray",
            width=500,
        )

        name = ["P_all"] + ["P_" + str(int(ii)) for ii in sub] + ["PPC_id"]
        colors = sns.color_palette(cc.glasbey_bw, len(sub) - 1)
        obj_allo1 = obj_allo[obj_allo.argsort(keys="ppc_priority")]
        obj_allo1["PPC_id"] = np.arange(0, len(obj_allo), 1)
        obj_allo1.rename_column("tel_fiber_usage_frac", "Fiber usage fraction (%)")
        obj_allo2 = Table.to_pandas(obj_allo1)
        uS_ = Table.to_pandas(uS)

        def plot_ppc(nppc_fin):
            def PFS_FoV_plot(ppc_ra, ppc_dec, PA):
                ppc_coord = []

                # PA=0 along y-axis, PA=90 along x-axis, PA=180 along -y-axis...
                for ii in range(len(ppc_ra)):
                    ppc_ra_t, ppc_dec_t, pa_t = ppc_ra[ii], ppc_dec[ii], PA[ii]
                    center = SkyCoord(ppc_ra_t * u.deg, ppc_dec_t * u.deg)

                    hexagon = center.directional_offset_by(
                        [
                            30 + pa_t,
                            90 + pa_t,
                            150 + pa_t,
                            210 + pa_t,
                            270 + pa_t,
                            330 + pa_t,
                            30 + pa_t,
                        ]
                        * u.deg,
                        1.38 / 2.0 * u.deg,
                    )
                    ra_h = hexagon.ra.deg
                    dec_h = hexagon.dec.deg

                    # for pointings around RA~0 or 360, parts of it will move to the opposite side (e.g., [[1,0],[-1,0]] -->[[1,0],[359,0]])
                    # correct for it
                    ra_h_in = np.where(np.fabs(ra_h - center.ra.deg) > 180)
                    if len(ra_h_in[0]) > 0:
                        if ra_h[ra_h_in[0][0]] > 180:
                            ra_h[ra_h_in[0]] -= 360
                        elif ra_h[ra_h_in[0][0]] < 180:
                            ra_h[ra_h_in[0]] += 360

                    ppc_coord.append([[ra_h[o], dec_h[o]] for o in range(len(ra_h))])

                ppc_tot_plot = [
                    hv.Area(ii).opts(color="gray", alpha=0.2, line_width=0)
                    for ii in ppc_coord
                ]

                pd_ppc = pd.DataFrame({"RA": ppc_ra, "DEC": ppc_dec, "PA": PA})
                p1 = pd_ppc.hvplot.scatter(
                    x="RA",
                    y="DEC",
                    by="PA",
                    title="Distribution of targets & PPC",
                    color="gray",
                    marker="s",
                    s=40,
                    legend=False,
                )

                return p1 * hv.Overlay(ppc_tot_plot)

            p_ppc = PFS_FoV_plot(
                obj_allo1["ppc_ra"][:nppc_fin],
                obj_allo1["ppc_dec"][:nppc_fin],
                obj_allo1["ppc_pa"][:nppc_fin],
            )

            p_tgt = uS_.hvplot.scatter(
                x="ra",
                y="dec",
                by="priority",
                color=["r"] + colors,
                marker="o",
                s=20,
                legend=True,
            )

            return (p_tgt * p_ppc).opts(show_grid=True)

        def plot_CR(nppc_fin):
            cR_ = np.array([list(cR[ii]) + [ii + 1] for ii in range(len(cR))])
            cR__ = pd.DataFrame(dict(zip(name, cR_.T)))

            p1 = cR__.hvplot.line(
                "PPC_id",
                name[:-1],
                value_label="Completion rate (%)",
                title=f"{RESmode:s}-resolution mode",
                color=["k", "r"] + colors,
                line_width=[3, 2] + [1] * (len(sub) - 1),
                line_dash=["solid"] * 2 + ["dashed"] * (len(sub) - 1),
            )
            p2 = hv.Rectangles([(30, 88, 95, 100)]).opts(
                color="orange", line_width=0, alpha=0.2
            )
            p3 = hv.Rectangles([(20, 43, 130, 93)]).opts(
                color="dodgerblue", line_width=0, alpha=0.2
            )
            p4 = hv.VLine(nppc_fin).opts(color="gray", line_dash="dashed", line_width=5)

            return (p1 * p2 * p3 * p4).opts(
                xlim=(0, len(obj_allo) + 1), ylim=(0, 105), show_grid=True
            )

        def plot_FE(nppc_fin):
            mean_FE = np.mean(obj_allo2["Fiber usage fraction (%)"][:nppc_fin])
            p1 = obj_allo2.hvplot.bar(
                "PPC_id",
                "Fiber usage fraction (%)",
                title=f"{RESmode:s}-resolution mode",
                rot=90,
                width=1,
                color="tomato",
                alpha=0.5,
                line_width=0,
            )
            p2 = (hv.HLine(mean_FE).opts(color="red", line_width=3)) * (
                hv.Text(
                    int(len(cR) * 0.85), mean_FE * 1.5, "{:.2f}%".format(mean_FE)
                ).opts(color="red")
            )
            p3 = hv.VLine(nppc_fin).opts(color="gray", line_dash="dashed", line_width=5)

            return (p1 * p2 * p3).opts(
                fontsize={"xticks": "0pt"},
                xlim=(0, len(obj_allo) + 1),
                ylim=(0, max(obj_allo2["Fiber usage fraction (%)"][:nppc_fin]) + 1),
            )

        def ppp_res_tab1(nppc_fin):
            hour_tot = nppc_fin * 15.0 / 60.0  # hour
            Fhour_tot = (
                sum([len(tt) for tt in obj_allo1[:nppc_fin]["allocated_targets"]])
                * 15.0
                / 60.0
            )  # fiber_count*hour
            Ttot_best, Ttot_worst = overheads(nppc_fin)
            fib_eff_mean = np.mean(obj_allo1["Fiber usage fraction (%)"][:nppc_fin])
            fib_eff_small = (
                sum(obj_allo1["Fiber usage fraction (%)"][:nppc_fin] < 30)
                / nppc_fin
                * 100.0
            )

            cR1 = pd.DataFrame(
                dict(zip(name[:-1], cR[nppc_fin - 1])),
                index=[0],
            )

            ppc_summary = pd.DataFrame(
                {
                    "resolution": [RESmode],
                    "N_ppc": [nppc_fin],
                    "Texp (h)": [hour_tot],
                    "Texp (fiberhour)": [Fhour_tot],
                    "Request time 1 (h)": [Ttot_best],
                    "Request time 2 (h)": [Ttot_worst],
                    "Used fiber fraction (%)": [fib_eff_mean],
                    "Fraction of PPC < 30% (%)": [fib_eff_small],
                },
            )

            ppc_summary_fin = pd.concat([ppc_summary, cR1], axis=1)

            return ppc_summary_fin

        def ppp_res_tab2(nppc_fin):
            obj_alloc = obj_allo1[:nppc_fin]
            obj_alloc.remove_column("allocated_targets")
            obj_alloc.remove_column("group_id")
            obj_alloc.remove_column("PPC_id")
            return Table.to_pandas(obj_alloc)

        p_result_fig = pn.Row(
            pn.Column(
                pn.bind(plot_CR, nppc),
                width=700,
                height=300,
            ),
            pn.Column(
                pn.bind(plot_FE, nppc),
                width=500,
                height=285,
            ),
            pn.Column(
                pn.bind(plot_ppc, nppc),
                width=600,
                height=300,
            ),
        )

        p_result_tab = pn.widgets.Tabulator(
            pn.bind(ppp_res_tab1, nppc),
            page_size=4,
            theme="bootstrap",
            theme_classes=["table-sm"],
            pagination="remote",
            visible=True,
            layout="fit_data_table",
            hidden_columns=["index"],
            selectable=False,
            header_align="right",
            configuration={"columnDefaults": {"headerSort": False}},
        )

        p_result_ppc = pn.widgets.Tabulator(
            pn.bind(ppp_res_tab2, nppc),
            visible=False,
        )

        return nppc, p_result_fig, p_result_tab, p_result_ppc

    if len(cR_l) > 0 and len(cR_m) == 0:
        nppc_l, p_result_fig_l, p_result_tab_l, p_result_ppc_l = ppp_plotFig(
            "low", cR_l, sub_l, obj_allo_l, uS_L2
        )

        return "low", nppc_l, p_result_fig_l, p_result_ppc_l, p_result_tab_l

    elif len(cR_m) > 0 and len(cR_l) == 0:
        nppc_m, p_result_fig_m, p_result_tab_m, p_result_ppc_m = ppp_plotFig(
            "medium", cR_m, sub_m, obj_allo_m, uS_M2
        )

        return "medium", nppc_m, p_result_fig_m, p_result_ppc_m, p_result_tab_m

    elif len(cR_l) > 0 and len(cR_m) > 0:
        nppc_l, p_result_fig_l, p_result_tab_l, p_result_ppc_l = ppp_plotFig(
            "low", cR_l, sub_l, obj_allo_l, uS_L2
        )
        nppc_m, p_result_fig_m, p_result_tab_m, p_result_ppc_m = ppp_plotFig(
            "medium", cR_m, sub_m, obj_allo_m, uS_M2
        )

        nppc_fin = pn.Row(nppc_l, nppc_m)
        p_result_fig_fin = pn.Column(p_result_fig_l, p_result_fig_m)

        def p_result_tab_tot(p_result_tab_l, p_result_tab_m):
            ppc_sum = pd.concat([p_result_tab_l, p_result_tab_m], axis=0)
            ppc_sum.loc[2] = ppc_sum.sum(numeric_only=True)
            ppc_sum.loc[2, "resolution"] = "Total"
            ppc_sum.iloc[2, 6:] = np.nan
            return ppc_sum

        def p_result_ppc_tot(p_result_ppc_l, p_result_ppc_m):
            ppc_lst = pd.concat([p_result_ppc_l, p_result_ppc_m], axis=0)
            return ppc_lst

        p_result_tab = pn.widgets.Tabulator(
            pn.bind(p_result_tab_tot, p_result_tab_l, p_result_tab_m),
            page_size=4,
            theme="bootstrap",
            theme_classes=["table-sm"],
            pagination="remote",
            visible=True,
            layout="fit_data_table",
            hidden_columns=["index"],
            selectable=False,
            header_align="right",
            configuration={"columnDefaults": {"headerSort": False}},
        )

        p_result_ppc_fin = pn.widgets.Tabulator(
            pn.bind(p_result_ppc_tot, p_result_ppc_l, p_result_ppc_m),
            visible=False,
        )

        return (
            "low & medium",
            nppc_fin,
            p_result_fig_fin,
            p_result_ppc_fin,
            p_result_tab,
        )  #'''


def visibility_checker(uS, semester):
    if len(uS) == 0:
        return np.array([])

    tz_HST = tz.gettz("US/Hawaii")

    if semester == "A":
        daterange = pd.date_range("20240201", "20240731")
    elif semester == "B":
        daterange = pd.date_range("20240801", "20250131")

    ob_code, RA, DEC, exptime = uS["ob_code"], uS["ra"], uS["dec"], uS["exptime"]

    min_el = 30.0
    max_el = 85.0

    tgt_obs_ok = []

    for i_t in range(len(RA)):
        target = StaticTarget(
            name=ob_code[i_t], ra=RA[i_t], dec=DEC[i_t], equinox=2000.0
        )
        total_time = exptime[i_t]  # SEC

        t_obs_ok = 0

        for dd in range(len(daterange) - 1):
            night_begin = parser.parse(
                daterange[dd].strftime("%Y-%m-%d") + " 18:30:00"
            ).replace(tzinfo=tz_HST)
            night_end = parser.parse(
                daterange[dd + 1].strftime("%Y-%m-%d") + " 05:30:00"
            ).replace(tzinfo=tz_HST)
            observer.set_date(night_begin)

            obs_ok, t_start, t_stop = observer.observable(
                target,
                night_begin,
                night_end,
                min_el,
                max_el,
                total_time,
                airmass=None,
                moon_sep=None,
            )

            if t_start is None or t_stop is None:
                t_obs_ok += 0
                continue

            if t_stop > t_start:
                t_obs_ok += (t_stop - t_start).seconds  # SEC
            else:
                t_obs_ok += 0

        if t_obs_ok >= exptime[i_t]:
            tgt_obs_ok.append(True)
        else:
            tgt_obs_ok.append(False)

    return np.array(tgt_obs_ok, dtype=bool)
