#!/usr/bin/env python3

import os
import random
import sys
import time
import warnings
from contextlib import redirect_stdout
from functools import partial
from itertools import chain

import colorcet as cc
import hdbscan
import holoviews as hv
import hvplot.pandas  # noqa need to run pandas.DataFrame.hvplot
import multiprocess as mp
import numpy as np
import pandas as pd
import panel as pn
import spatialpandas as sp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, vstack
from bokeh.models.widgets.tables import NumberFormatter
from loguru import logger
from matplotlib.path import Path
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.neighbors import KernelDensity
from spatialpandas.geometry import PolygonArray

# below for netflow
# isort: split
import ets_fiber_assigner.netflow as nf
from ics.cobraOps.Bench import Bench

# check bokeh version
# ref: https://discourse.holoviz.org/t/strange-behavior-in-legend-when-curve-line-dash-not-solid/5547/2
# isort: split
import bokeh
from pkg_resources import parse_version

if parse_version(bokeh.__version__) < parse_version("3.3"):
    hv.renderer("bokeh").webgl = False

warnings.filterwarnings("ignore")

pn.extension(notifications=True)


def calc_total_obstime(
    n_sci_frame: int,
    single_exptime: float,
    t_night_hours: float = 10.0,
    t_overhead_misc: float = 120.0,
    t_overhead_fiber: float = 180.0,
    t_overhead_shared_hours: float = 1.2,
) -> float:
    """
    Calculate the total observing time including overheads for the given number of science frames and single exposure time.

    Parameters
    ----------
    n_sci_frame : int
        Number of scientific frames (pointings) to be observed.
    single_exptime : float
        Exposure time for a single scientific frame in seconds.
    t_night_hours : float, optional
        Total observing time per night in hours. Default is 10.0 hours.
    t_overhead_misc : float, optional
        Miscellaneous overheads in seconds. Default is 120.0 seconds.
    t_overhead_fiber : float, optional
        Fiber reconfiguration overheads in seconds. Default is 180.0 seconds.
    t_overhead_shared_hours : float, optional
        Shared calibration overheads in hours. Default is 1.2 hours.

    Returns
    -------
    float
        Total observing time in hours.

    Notes
    -----
    Overhead time is 1.2 h per night. A night is defined as 10 h. So, 1.2 h is charged
    to the program for every 8.8 h of observing time.
    In addition to that, 5 min (3 min for fiber configuration and 2 min for misc. processes)
    are charged per pointing.
    """

    # in seconds
    t_exp_sci: float = single_exptime  # [s]

    # [s] total time for all pointings
    t_total_pointings = (t_exp_sci + t_overhead_misc + t_overhead_fiber) * n_sci_frame

    # [s] overheads for the program
    t_overhead_program = (
        t_total_pointings
        / (t_night_hours - t_overhead_shared_hours)
        * t_overhead_shared_hours
    )

    # [s] total request observing time (ROT)
    Toverheads_tot_best = t_total_pointings + t_overhead_program

    # return the total observing time in hours
    return Toverheads_tot_best / 3600.0


def calc_nppc_from_obstime(
    total_obstime: float,
    single_exptime: float,
    t_night_hours: float = 10.0,
    t_overhead_misc: float = 120.0,
    t_overhead_fiber: float = 180.0,
    t_overhead_shared_hours: float = 1.2,
) -> int:
    """
    Calculate the number of pointings (N_PPCs) that can be observed within the given total observing time.

    Parameters
    ----------
    total_obstime : float
        Total observing time in hours.
    single_exptime : float
        Exposure time for a single scientific frame in seconds.
    t_night_hours : float, optional
        Total observing time per night in hours. Default is 10.0 hours.
    t_overhead_misc : float, optional
        Miscellaneous overheads in seconds. Default is 120.0 seconds.
    t_overhead_fiber : float, optional
        Fiber reconfiguration overheads in seconds. Default is 180.0 seconds.
    t_overhead_shared_hours : float, optional
        Shared calibration overheads in hours. Default is 1.2 hours.

    Returns
    -------
    int
        Number of pointings (N_PPCs) that can be observed within the given total observing time.
    """

    # science exposure time in seconds
    t_exp_sci: float = single_exptime

    # total time for single pointing in seconds
    t_total_per_pointings = t_exp_sci + t_overhead_misc + t_overhead_fiber

    # effective night length in hours
    t_night_eff = t_night_hours / (t_night_hours - t_overhead_shared_hours)

    # time used for exposure + per-exposure overheads in seconds
    t_exp_eff = total_obstime / t_night_eff * 3600.0

    # number of pointings
    n_pointings = int(round(t_exp_eff / t_total_per_pointings, 0))

    return n_pointings


def PPPrunStart(
    uS,
    uPPC,
    weight_para,
    single_exptime: int = 900,
    d_pfi=1.38,
    quiet=True,
    clustering_algorithm="HDBSCAN",
    queue=None,
    logger=None,
):
    if logger is None:
        logger.remove()
        logger.add(sys.stderr, level="INFO", enqueue=True)

    r_pfi = d_pfi / 2.0

    ppp_quiet = quiet

    if weight_para is None:
        weight_para = [2.02, 0.01, 0.01]

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
            # * pow(sample["exptime_PPP"] / 900.0, contb)
            * pow(sample["exptime_PPP"] / single_exptime, contb)
            * pow(sample["local_count"], contc)
        )

        if "weight" not in sample.colnames:
            sample.add_column(weight_t, name="weight")
        else:
            sample["weight"] = weight_t

        return sample

    def target_clustering(sample, sep=d_pfi, algorithm="DBSCAN"):
        """separate pointings/targets into different groups

        Parameters
        ==========
        sample: astropy table
            astropy table with columns of ra, dec, weight
        sep: float
            angular separation set to group, degree
        algorithm: str
            clustering algorithm, either "DBSCAN" or "HDBSCAN"

        Returns
        =======
        list of pointing centers in different group
        """
        logger.debug(f"{sample['ra']=}")
        logger.debug(f"{sample['dec']=}")

        # haversine uses (dec,ra) in radian
        # HDBSCAN needs more than 1 data point to work, so use DBSCAN for single target clustering.
        if algorithm.upper() == "DBSCAN" or len(sample["ra"]) < 2:
            logger.info("algorithm for target clustering: DBSCAN")
            db = DBSCAN(eps=np.radians(sep), min_samples=1, metric="haversine").fit(
                np.radians([sample["dec"], sample["ra"]]).T
            )
            labels = db.labels_
        elif algorithm.upper() == "HDBSCAN":
            logger.info("algorithm for target clustering: HDBSCAN")
            db = HDBSCAN(min_cluster_size=2, metric="haversine").fit(
                np.radians([sample["dec"], sample["ra"]]).T
            )
            labels = db.dbscan_clustering(np.radians(sep), min_cluster_size=1)
        elif algorithm.upper() == "FAST_HDBSCAN":
            logger.info("algorithm for target clustering: FAST_HDBSCAN")
            db = HDBSCAN(min_cluster_size=2, metric="haversine").fit(
                np.radians([sample["dec"], sample["ra"]]).T
            )
            labels = db.dbscan_clustering(np.radians(sep), min_cluster_size=1)
        else:
            logger.error("algorithm should be one of DBSCAN, HDBSCAN, and FAST_HDBSCAN")
            raise ValueError(
                "algorithm should be one of DBSCAN, HDBSCAN, and FAST_HDBSCAN"
            )

        logger.info("Clustering finished")

        unique_labels = set(labels)
        n_clusters = len(unique_labels)

        tgt_group = []
        tgt_pri_ord = []

        for ii in range(n_clusters):
            tgt_t_pri_tot = sum(sample[labels == ii]["weight"])
            tgt_pri_ord.append([ii, tgt_t_pri_tot])

        tgt_pri_ord.sort(key=lambda x: x[1], reverse=True)

        for jj in np.array(tgt_pri_ord)[:, 0]:
            tgt_t = sample[labels == jj]
            tgt_group.append(tgt_t)

        # delete db object to release memory
        del db

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
            r_pfi * u.deg,
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
            bandwidth=np.deg2rad(r_pfi),
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
                threads_count = round(mp.cpu_count() / 2)
                thread_n = min(
                    threads_count, round(len(sample) * 0.5)
                )  # threads_count=10 in this machine

                kde_p = mp.Pool(thread_n)
                dMap_ = kde_p.map_async(
                    partial(KDE_xy, X=X_, Y=Y_),
                    np.array_split(sample, thread_n),
                )

                kde_p.close()
                kde_p.join()

                dMap_ = dMap_.get()

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

    def _make_obj_allo_table(
        peaks,
        resolution,
    ):
        """
        Build an astropy Table for PPC allocations from a list of peaks.

        Each peak entry should be:
        [id, ra, dec, pa, priority, assign_mask, target_ids]

        Returns:
        Table with columns: ppc_code_, group_id, ppc_ra, ppc_dec, ppc_pa,
        ppc_priority, tel_fiber_usage_frac, allocated_targets, ppc_resolution.
        """
        from astropy.table import Column, Table

        n = len(peaks)
        codes = [f"Point_{p[0]+1}" for p in peaks]
        groups = ["Group_1"] * n

        ras = [p[1] for p in peaks]
        decs = [p[2] for p in peaks]
        pas = [p[3] for p in peaks]
        priorities = [p[4] for p in peaks]
        masks = [p[5] for p in peaks]
        tgt_lists = [p[6] for p in peaks]
        fiber_fracs = [sum(mask) / 2394.0 * 100.0 for mask in masks]

        tbl = Table()
        tbl["ppc_code_"] = Column(codes, dtype=np.str_)
        tbl["group_id"] = Column(groups, dtype=np.str_)
        tbl["ppc_ra"] = Column(ras, dtype=np.float64)
        tbl["ppc_dec"] = Column(decs, dtype=np.float64)
        tbl["ppc_pa"] = Column(pas, dtype=np.float64)
        tbl["ppc_priority"] = Column(priorities, dtype=np.float64)
        tbl["tel_fiber_usage_frac"] = Column(fiber_fracs, dtype=np.float64)
        tbl["allocated_targets"] = Column(tgt_lists, dtype=object)
        tbl["ppc_resolution"] = Column([resolution] * n, dtype=np.str_)
        return tbl

    def PPP_centers(
        sample_f,
        ppc_f,
        mutiPro,
        weight_para,
        uS_L2=Table(),
        cR_L=[],
        cR_L_=[],
        sub_l=[],
        obj_allo_L_fin=Table(),
    ):
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
        conta, contb, contc = weight_para
        status = 999
        Nfiber = 2394 - 200  # 200 reserved for calibrators

        # Precompute densities and weights
        sample_f = count_N(sample_f)
        sample_f = weight(sample_f, conta, contb, contc)

        # If user-specified PPC provided, use that
        if len(ppc_f) > 0:
            sample_f.meta["PPC"] = np.vstack(
                [
                    (i, ppc_f["ppc_ra"][i], ppc_f["ppc_dec"][i], ppc_f["ppc_pa"][i])
                    for i in range(len(ppc_f))
                ]
            )
            status = 2
            logger.info("PPC from user input [PPP_centers s1]")
            return sample_f, status

        peaks = []  # list of [id, ra, dec, pa]
        # iterate clusters
        for cluster in target_clustering(
            sample_f, d_pfi, algorithm=clustering_algorithm
        ):
            # only unfinished targets
            remaining = cluster[cluster["exptime_PPP"] > 0]
            # continue until all exposure done
            while any(remaining["exptime_PPP"] > 0):
                # KDE peak
                _, _, _, ra_peak, dec_peak = KDE(remaining, True)
                pa_peak = 0.0

                # find targets in FoV
                idx = PFS_FoV(ra_peak, dec_peak, pa_peak, remaining)

                # run netflow once
                tgt_ids = netflowRun4PPC(
                    remaining[list(idx)], ra_peak, dec_peak, pa_peak
                )

                # retry if none assigned
                for attempt in range(2):
                    if len(tgt_ids) > 0:
                        break
                    ra_peak += np.random.uniform(-0.15, 0.15)
                    dec_peak += np.random.uniform(-0.15, 0.15)
                    tgt_ids = netflowRun4PPC(
                        remaining[list(idx)],
                        ra_peak,
                        dec_peak,
                        pa_peak,
                        otime="2025-04-10T08:00:00Z",
                    )

                mask_assign = np.in1d(remaining["ob_code"], tgt_ids)
                priority_val = 1.0 / remaining[mask_assign]["weight"].sum()

                # record peak
                peaks.append(
                    [
                        len(peaks),
                        ra_peak,
                        dec_peak,
                        pa_peak,
                        priority_val,
                        mask_assign,
                        list(tgt_ids),
                    ]
                )

                # flush every 1 peaks if queue provided
                if queue:
                    res_ = sample_f["resolution"][0]
                    sample_f.meta["PPC"] = np.array([p[:4] for p in peaks])
                    obj_table = _make_obj_allo_table(peaks, res_)

                    if res_ == "L":
                        uS_L2_ = sample_f.copy()
                        obj_allo_L_fin = obj_table.copy()
                        (uS_L2, cR_L_fh, cR_L_fh_, cR_L_n, cR_L_n_, sub_l) = (
                            complete_ppc(uS_L2_, obj_allo_L_fin)
                        )

                        queue.put(
                            [
                                uS_L2,
                                [cR_L_fh, cR_L_n],
                                [cR_L_fh_, cR_L_n_],
                                sub_l,
                                obj_allo_L_fin,
                                Table(),
                                [],
                                [],
                                [],
                                Table(),
                                status,
                            ]
                        )

                    elif res_ == "M":
                        uS_M2_ = sample_f.copy()
                        obj_allo_M_fin = obj_table.copy()
                        (uS_M2, cR_M_fh, cR_M_fh_, cR_M_n, cR_M_n_, sub_m) = (
                            complete_ppc(uS_M2_, obj_allo_M_fin)
                        )

                        queue.put(
                            [
                                uS_L2,
                                cR_L,
                                cR_L_,
                                sub_l,
                                obj_allo_L_fin,
                                uS_M2,
                                [cR_M_fh, cR_M_n],
                                [cR_M_fh_, cR_M_n_],
                                sub_m,
                                obj_allo_M_fin,
                                status,
                            ]
                        )

                # decrement exposure
                remaining["exptime_PPP"][mask_assign] -= single_exptime
                remaining = remaining[remaining["exptime_PPP"] > 0]
                remaining = count_N(weight(remaining, conta, contb, contc))

        # final meta and return
        sample_f.meta["PPC"] = np.array([p[:4] for p in peaks])
        return sample_f, status

    def point_DBSCAN(sample):
        """separate pointings into different group

        Parameters
        ==========
        sample:table

        Returns
        =======
        list of pointing centers in different group
        """
        ppc_xy = sample.meta["PPC"]

        # haversine uses (dec,ra) in radian;
        db = DBSCAN(eps=np.radians(d_pfi), min_samples=1, metric="haversine").fit(
            np.fliplr(np.radians(ppc_xy[:, [1, 2]].astype(np.float_)))
        )

        labels = db.labels_
        unique_labels = set(labels)
        n_clusters = len(unique_labels)

        ppc_group = []

        for ii in range(n_clusters):
            ppc_t = ppc_xy[labels == ii]
            ppc_group.append(ppc_t)

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
            # targetL.append(nf.ScienceTarget(id_, ra, dec, tm, int_, "sci"))
            targetL.append(nf.ScienceTarget(id_, ra, dec, tm, tt["priority"], "sci"))
            int_ += 1

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

        """
        int_ = 0
        for ii in sample:
            classdict["sci_P" + str(int_)] = {
                "nonObservationCost": ii["weight"],
                "partialObservationCost": ii["weight"] * 1.5,
                "calib": False,
            }
            int_ += 1
        #"""

        classdict["sci_P0"] = {
            "nonObservationCost": 100,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P1"] = {
            "nonObservationCost": 90,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P2"] = {
            "nonObservationCost": 80,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P3"] = {
            "nonObservationCost": 70,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P4"] = {
            "nonObservationCost": 60,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P5"] = {
            "nonObservationCost": 50,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P6"] = {
            "nonObservationCost": 40,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P7"] = {
            "nonObservationCost": 30,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P8"] = {
            "nonObservationCost": 20,
            "partialObservationCost": 200,
            "calib": False,
        }
        classdict["sci_P9"] = {
            "nonObservationCost": 10,
            "partialObservationCost": 200,
            "calib": False,
        }

        return classdict

    def cobraMoveCost(dist):
        """optional: penalize assignments where the cobra has to move far out"""
        return 0.1 * dist

    def netflowRun_single(Tel, sample, otime="2024-05-20T08:00:00Z"):
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

        if ppp_quiet:
            out_target = open(os.devnull, "w")
        else:
            out_target = sys.stdout
        # disable netflow output
        with redirect_stdout(out_target):
            # compute observation strategy
            prob = nf.buildProblem(
                bench,
                tgt,
                tpos,
                classdict,
                # 900,
                single_exptime,
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

    def netflowRun_nofibAssign(
        Tel,
        sample,
        otime="2025-04-20T08:00:00Z",
    ):
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

            Tel = np.array(Tel)
            Tel_t = np.copy(Tel)
            otime_ = otime
            iter_1 = 0

            while len(index) > 0 and iter_1 < 8:
                # shift PPCs with 0.2 deg, but only run 6 iterations to save computational time
                # typically one iteration is enough
                shift_ra = np.random.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], 1)[0]
                shift_dec = np.random.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], 1)[0]

                Tel_t[index, 1] = Tel[index, 1] + shift_ra
                Tel_t[index, 2] = Tel[index, 2] + shift_dec

                res, telescope, tgt = netflowRun_single(Tel_t, sample, otime_)
                index = np.where(np.array([len(tt) for tt in res]) == 0)[0]

                iter_1 += 1

                if iter_1 >= 4:
                    otime_ = "2024-04-20T08:00:00Z"

            return res, telescope, tgt

    def netflowRun4PPC(
        _tb_tgt_inuse,
        ppc_x,
        ppc_y,
        ppc_pa,
        otime="2025-05-20T08:00:00Z",
    ):
        # run netflow (for PPP_centers)
        ppc_lst = np.array([[0, ppc_x, ppc_y, ppc_pa, 0]])

        res, telescope, tgt_lst_netflow = netflowRun_nofibAssign(
            ppc_lst,
            _tb_tgt_inuse,
            otime=otime,
        )

        for i, (vis, tel) in enumerate(zip(res, telescope)):
            # assigned targets in each ppc
            tgt_assign_id_lst = []
            for tidx, cidx in vis.items():
                tgt_assign_id_lst.append(tgt_lst_netflow[tidx].ID)

        return tgt_assign_id_lst

    def netflowRun(sample):
        """run netflow (with iteration and DBSCAN)

        Parameters
        ==========
        sample : sample

        Returns
        =======
        Fiber assignment in each PPC
        """

        if len(sample.meta["PPC"]) == 0:
            point_t = []
            logger.info("No PPC is determined due to running out of time [netflowRun]")
            return point_t

        ppc_g = point_DBSCAN(sample)  # separate ppc into different groups

        point_list = []
        point_c = 0

        for uu in range(len(ppc_g)):  # run netflow for each ppc group
            # only consider sample in the group
            sample_index = list(
                chain.from_iterable(
                    [
                        list(
                            PFS_FoV(
                                float(ppc_g[uu][iii, 1]),
                                float(ppc_g[uu][iii, 2]),
                                float(ppc_g[uu][iii, 3]),
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
                "ppc_code_",
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
        sample["exptime_assign"] = 0
        sub_l = sorted(list(set(sample["priority"])))
        n_sub = len(sub_l)

        if len(point_l) == 0:
            return (
                sample,
                np.array([[0] * (n_sub + 1)]),
                np.array([[0] * (n_sub + 1)]),
                sub_l,
            )

        point_l_pri = point_l[
            point_l.argsort(keys="ppc_priority")
        ]  # sort ppc by its total priority == sum(weights of the assigned targets in ppc)

        # sub-groups of the input sample, catagarized by the user defined priority
        count_sub_fh = [sum(sample["exptime_PPP"]) / 3600.0] + [
            sum(sample[sample["priority"] == ll]["exptime_PPP"]) / 3600.0
            for ll in sub_l
        ]  # fiber hours
        count_sub_n = [len(sample)] + [
            sum(sample["priority"] == ll) for ll in sub_l
        ]  # number count of complete targets

        completeR_fh = []  # fiber hours
        completeR_fh_ = []  # percentage

        completeR_n = []  # number count of complete targets
        completeR_n_ = []  # percentage

        for ppc in point_l_pri:
            lst = np.where(np.in1d(sample["ob_code"], ppc["allocated_targets"]))[0]
            sample["exptime_assign"].data[lst] += single_exptime

            # achieved fiber hours (in total, in P[0-9])
            comT_t_fh = [sum(sample["exptime_assign"]) / 3600.0] + [
                sum(sample[sample["priority"] == ll]["exptime_assign"]) / 3600.0
                for ll in sub_l
            ]

            comp_s = np.where(sample["exptime_PPP"] <= sample["exptime_assign"])[0]
            comT_t_n = [len(comp_s)] + [
                sum(sample["priority"].data[comp_s] == ll) for ll in sub_l
            ]

            completeR_fh.append(comT_t_fh)
            completeR_fh_.append(
                [
                    comT_t_fh[oo] / count_sub_fh[oo] * 100
                    for oo in range(len(count_sub_fh))
                ]
            )

            completeR_n.append(comT_t_n)
            completeR_n_.append(
                [comT_t_n[oo] / count_sub_n[oo] * 100 for oo in range(len(count_sub_n))]
            )

        return (
            sample,
            np.array(completeR_fh),
            np.array(completeR_fh_),
            np.array(completeR_n),
            np.array(completeR_n_),
            sub_l,
        )

    def netflow_iter(uS, obj_allo, weight_para, status):
        """iterate the total procedure to re-assign fibers to targets which have not been assigned
            in the previous/first iteration

        Parameters
        ==========
        uS: table
            sample with exptime>exptime_assign

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
        """

        if sum(uS["exptime_assign"] == uS["exptime_PPP"]) == len(uS):
            # remove ppc with no fiber assignment
            obj_allo.remove_rows(np.where(obj_allo["tel_fiber_usage_frac"] == 0)[0])
            return obj_allo, status

        else:
            #  select non-assigned targets --> PPC determination --> netflow --> if no fibre assigned: shift PPC
            iter_m2 = 0

            while any(uS["exptime_assign"] < uS["exptime_PPP"]) and iter_m2 < 10:
                uS_t1 = uS[uS["exptime_assign"] < uS["exptime_PPP"]]
                uS_t1["exptime_PPP"] = (
                    uS_t1["exptime_PPP"] - uS_t1["exptime_assign"]
                )  # remained exposure time

                uS_t2, status = PPP_centers(uS_t1, [], True, weight_para)

                obj_allo_t = netflowRun(uS_t2)

                obj_allo = vstack([obj_allo, obj_allo_t])
                obj_allo.remove_rows(np.where(obj_allo["tel_fiber_usage_frac"] == 0)[0])
                uS = complete_ppc(uS_t2, obj_allo)[0]
                iter_m2 += 1

            return obj_allo, status

    # computation starts here
    logger.info("PPP run started")
    t_ppp_start = time.time()

    out_uS_L2 = []
    out_cR_L = []
    out_cR_L_ = []
    out_sub_l = []
    out_obj_allo_L_fin = []
    out_uS_M2 = []
    out_cR_M = []
    out_cR_M_ = []
    out_sub_m = []
    out_obj_allo_M_fin = []

    exptime_ppp = np.ceil(uS["exptime"] / single_exptime) * single_exptime
    uS.add_column(exptime_ppp, name="exptime_PPP")

    uS_L = uS[uS["resolution"] == "L"]
    uS_M = uS[uS["resolution"] == "M"]
    if (uPPC is not None) and (len(uPPC) > 0):
        uPPC_L = uPPC[uPPC["ppc_resolution"] == "L"]
        uPPC_M = uPPC[uPPC["ppc_resolution"] == "M"]
    else:
        uPPC_L = []
        uPPC_M = []

    if len(uS_L) > 0 and len(uS_M) == 0:
        uS_L_s2, status_ = PPP_centers(uS_L, uPPC_L, True, weight_para)
        obj_allo_L = netflowRun(uS_L_s2)

        if len(uPPC_L) == 0:
            uS_L2 = complete_ppc(uS_L_s2, obj_allo_L)[0]
            obj_allo_L_fin, status_ = netflow_iter(
                uS_L2, obj_allo_L, weight_para, status_
            )
            uS_L2, cR_L_fh, cR_L_fh_, cR_L_n, cR_L_n_, sub_l = complete_ppc(
                uS_L_s2, obj_allo_L_fin
            )
            out_obj_allo_L_fin = obj_allo_L_fin

        elif len(uPPC_L) > 0:
            uS_L2, cR_L_fh, cR_L_fh_, cR_L_n, cR_L_n_, sub_l = complete_ppc(
                uS_L_s2, obj_allo_L
            )
            out_obj_allo_L_fin = obj_allo_L

        out_uS_L2 = uS_L2
        out_cR_L = [cR_L_fh, cR_L_n]
        out_cR_L_ = [cR_L_fh_, cR_L_n_]
        out_sub_l = sub_l

    if len(uS_M) > 0 and len(uS_L) == 0:
        uS_M_s2, status_ = PPP_centers(uS_M, uPPC_M, True, weight_para)
        obj_allo_M = netflowRun(uS_M_s2)

        if len(uPPC_M) == 0:
            uS_M2 = complete_ppc(uS_M_s2, obj_allo_M)[0]
            obj_allo_M_fin, status_ = netflow_iter(
                uS_M2, obj_allo_M, weight_para, status_
            )
            uS_M2, cR_M_fh, cR_M_fh_, cR_M_n, cR_M_n_, sub_m = complete_ppc(
                uS_M_s2, obj_allo_M_fin
            )
            out_obj_allo_M_fin = obj_allo_M_fin
        elif len(uPPC_M) > 0:
            uS_M2, cR_M_fh, cR_M_fh_, cR_M_n, cR_M_n_, sub_m = complete_ppc(
                uS_M_s2, obj_allo_M
            )
            out_obj_allo_M_fin = obj_allo_M

        out_uS_M2 = uS_M2
        out_cR_M = [cR_M_fh, cR_M_n]
        out_cR_M_ = [cR_M_fh_, cR_M_n_]
        out_sub_m = sub_m

    if len(uS_L) > 0 and len(uS_M) > 0:
        uS_L_s2, status_ = PPP_centers(uS_L, uPPC_L, True, weight_para)
        obj_allo_L = netflowRun(uS_L_s2)
        if len(uPPC_L) == 0:
            uS_L2 = complete_ppc(uS_L_s2, obj_allo_L)[0]
            obj_allo_L_fin, status_ = netflow_iter(
                uS_L2, obj_allo_L, weight_para, status_
            )
            uS_L2, cR_L_fh, cR_L_fh_, cR_L_n, cR_L_n_, sub_l = complete_ppc(
                uS_L_s2, obj_allo_L_fin
            )
            out_obj_allo_L_fin = obj_allo_L_fin
        elif len(uPPC_L) > 0:
            uS_L2, cR_L_fh, cR_L_fh_, cR_L_n, cR_L_n_, sub_l = complete_ppc(
                uS_L_s2, obj_allo_L
            )
            out_obj_allo_L_fin = obj_allo_L

        out_uS_L2 = uS_L2
        out_cR_L = [cR_L_fh, cR_L_n]
        out_cR_L_ = [cR_L_fh_, cR_L_n_]
        out_sub_l = sub_l

        uS_M_s2, status_ = PPP_centers(
            uS_M,
            uPPC_M,
            True,
            weight_para,
            out_uS_L2,
            out_cR_L,
            out_cR_L_,
            out_sub_l,
            out_obj_allo_L_fin,
        )
        obj_allo_M = netflowRun(uS_M_s2)
        if len(uPPC_M) == 0:
            uS_M2 = complete_ppc(uS_M_s2, obj_allo_M)[0]
            obj_allo_M_fin, status_ = netflow_iter(
                uS_M2, obj_allo_M, weight_para, status_
            )
            uS_M2, cR_M_fh, cR_M_fh_, cR_M_n, cR_M_n_, sub_m = complete_ppc(
                uS_M_s2, obj_allo_M_fin
            )
            out_obj_allo_M_fin = obj_allo_M_fin
        elif len(uPPC_M) > 0:
            uS_M2, cR_M_fh, cR_M_fh_, cR_M_n, cR_M_n_, sub_m = complete_ppc(
                uS_M_s2, obj_allo_M
            )
            out_obj_allo_M_fin = obj_allo_M

        out_uS_M2 = uS_M2
        out_cR_M = [cR_M_fh, cR_M_n]
        out_cR_M_ = [cR_M_fh_, cR_M_n_]
        out_sub_m = sub_m

    t_ppp_stop = time.time()
    logger.info(f"PPP run finished in {t_ppp_stop-t_ppp_start:.1f} seconds")
    logger.info(f"PPP running status: {status_:.0f}")

    queue.put(
        [
            out_uS_L2,
            out_cR_L,
            out_cR_L_,
            out_sub_l,
            out_obj_allo_L_fin,
            out_uS_M2,
            out_cR_M,
            out_cR_M_,
            out_sub_m,
            out_obj_allo_M_fin,
            status_,
        ]
    )

    return (
        out_uS_L2,
        out_cR_L,
        out_cR_L_,
        out_sub_l,
        out_obj_allo_L_fin,
        out_uS_M2,
        out_cR_M,
        out_cR_M_,
        out_sub_m,
        out_obj_allo_M_fin,
        status_,
    )


def ppp_result(
    cR_l,
    sub_l,
    obj_allo_l,
    uS_L2,
    cR_m,
    sub_m,
    obj_allo_m,
    uS_M2,
    single_exptime=900,
    d_pfi=1.38,
    box_width=1200.0,
    plot_height=400,
):
    r_pfi = d_pfi / 2.0

    tabulator_stylesheet = """
    .tabulator-row-odd { background-color: #ffffff !important; }
    .tabulator-row-even { background-color: #ffffff !important; }
    .tabulator-row-odd:hover { color: #000000 !important; background-color: #ffffff !important; }
    .tabulator-row-even:hover { color: #000000 !important; background-color: #ffffff !important; }
    """

    # add styling/formatting to the table
    tabulator_formatters = {
        "N_ppc": NumberFormatter(format="0", text_align="right"),
        "Texp (h)": NumberFormatter(format="0.00", text_align="right"),
        "Texp (fiberhour)": NumberFormatter(format="0.00", text_align="right"),
        "Request time (h)": NumberFormatter(format="0.00", text_align="right"),
        "Used fiber fraction (%)": NumberFormatter(format="0.000", text_align="right"),
        "Fraction of PPC < 30% (%)": NumberFormatter(format="0.0", text_align="right"),
    }
    for p in ["all"] + np.arange(10).tolist():
        tabulator_formatters[f"P_{p}"] = NumberFormatter(
            format="0.0", text_align="right"
        )

    def ppp_plotFig(RESmode, cR, sub, obj_allo, uS):
        nppc = pn.widgets.EditableIntSlider(
            name=(f"{RESmode.capitalize()}-resolution mode"),
            value=len(cR[0]),
            step=1,
            start=1,
            end=len(cR[0]),
            fixed_start=1,
            fixed_end=len(cR[0]),
            bar_color="gray",
            max_width=450,
        )

        legend_cols = 2 if len(sub) >= 6 else 1

        name = ["P_all"] + ["P_" + str(int(ii)) for ii in sub] + ["PPC_id"]
        # colors for priority 0-9
        # red + first colors from glasbey_dark colormap as strings
        colors_all = ["red"] + cc.b_glasbey_bw_minc_20_maxl_70[:9]
        colors = [colors_all[i] for i in sub]

        obj_allo1 = obj_allo[obj_allo.argsort(keys="ppc_priority")]
        obj_allo1["PPC_id"] = np.arange(0, len(obj_allo), 1) + 1
        """
        if len(uPPC) > 0:
            if "ppc_priority" in uPPC.colnames:
                uPPC.remove_column("ppc_priority")
            obj_allo1 = join(
                obj_allo1, uPPC, keys=["ppc_ra", "ppc_dec", "ppc_pa", "ppc_resolution"]
            )
        else:
            obj_allo1["ppc_code"] = [
                "Point_" + RESmode + "_" + str(count)
                for count in (np.arange(0, len(obj_allo), 1) + 1)
            ]
        #"""
        obj_allo1["ppc_code"] = [
            "Point_" + RESmode + "_" + str(count)
            for count in (np.arange(0, len(obj_allo), 1) + 1)
        ]

        # obj_allo1 = obj_allo1.group_by("ppc_code")
        obj_allo1.rename_column("tel_fiber_usage_frac", "Fiber usage fraction (%)")
        simple_cols = [col for col in obj_allo1.colnames if obj_allo1[col].ndim == 1 or obj_allo1[col].shape == ()]
        obj_allo2 = obj_allo1[simple_cols].to_pandas()
        # obj_allo2 = Table.to_pandas(obj_allo1)
        uS_ = Table.to_pandas(uS)

        # add a column to indicate the color for the scatter plot
        uS_["ppc_color"] = [colors_all[i] for i in uS_["priority"]]

        cR_fh_ = np.array([list(cR[0][ii]) + [ii + 1] for ii in range(len(cR[0]))])
        cR_fh__ = pd.DataFrame(dict(zip(name, cR_fh_.T)))

        cR_n_ = np.array([list(cR[1][ii]) + [ii + 1] for ii in range(len(cR[1]))])
        cR_n__ = pd.DataFrame(dict(zip(name, cR_n_.T)))

        # create polygons for PFS FoVs for each pointing
        ppc_coord = []
        # PA=0 along y-axis, PA=90 along x-axis, PA=180 along -y-axis...
        for ii in range(len(obj_allo1["ppc_ra"])):
            ppc_ra_t, ppc_dec_t, pa_t = (
                obj_allo1["ppc_ra"][ii],
                obj_allo1["ppc_dec"][ii],
                obj_allo1["ppc_pa"][ii],
            )
            center = SkyCoord(ppc_ra_t * u.deg, ppc_dec_t * u.deg)
            hexagon = center.directional_offset_by(
                np.array([deg + pa_t for deg in [30, 90, 150, 210, 270, 330, 30]])
                * u.deg,
                r_pfi * u.deg,
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
            ppc_coord_one = []
            for a, d in zip(ra_h, dec_h):
                ppc_coord_one += [a, d]
            ppc_coord.append([np.array(ppc_coord_one)])
        ppc_coord_polygon_array = PolygonArray(ppc_coord)
        df_polygon = sp.GeoDataFrame(({"polygons": ppc_coord_polygon_array}))

        #
        # The following p_ are static plots and neeed to be created only once
        #
        # for sky distributions
        ra_min = np.min([obj_allo1["ppc_ra"].min(), uS_["ra"].min()]) - d_pfi
        ra_max = np.max([obj_allo1["ppc_ra"].max(), uS_["ra"].max()]) + d_pfi
        dec_min = np.min([obj_allo1["ppc_dec"].min(), uS_["dec"].min()]) - d_pfi
        dec_max = np.max([obj_allo1["ppc_dec"].max(), uS_["dec"].max()]) + d_pfi

        priorities = sorted(uS_["priority"].unique())
        uS_["priority"] = pd.Categorical(
            uS_["priority"], categories=priorities, ordered=True
        )
        uS_ = uS_.sort_values("priority")

        p_tgt = uS_.hvplot.scatter(
            x="ra",
            y="dec",
            by="priority",
            color="ppc_color",
            marker="o",
            s=60,
            line_color="white",
            line_width=0.5,
            legend=True,
        )

        obj_allo2_for_ppcplot = obj_allo2.rename(
            columns={"ppc_ra": "RA", "ppc_dec": "Dec", "ppc_pa": "PA"}
        )

        # static part for compolation rate plot
        p_comp_rate_fh = cR_fh__.hvplot.line(
            x="PPC_id",
            y=name[:-1],
            value_label="Completion rate (%)",
            title="Achieved fiberhours / total fiberhours",
            color=["k"] + colors,
            line_width=[4, 3] + [2] * (len(sub) - 1),
            line_dash=["solid"] * 2 + ["dashed"] * (len(sub) - 1),
            legend="right",
        )

        p_comp_rate_n = cR_n__.hvplot.line(
            x="PPC_id",
            y=name[:-1],
            value_label="Completion rate (%)",
            title="N(fully complete targets) / N(targets)",
            color=["k"] + colors,
            line_width=[4, 3] + [2] * (len(sub) - 1),
            line_dash=["solid"] * 2 + ["dashed"] * (len(sub) - 1),
            legend="right",
        )

        """
        # static part for fiber usage fraction plot
        p_fibereff_bar = hv.Bars(
            obj_allo2,
            kdims=["PPC_id"],
            vdims=["Fiber usage fraction (%)"],
        ).opts(
            title="Fiber usage fraction by pointing",
            width=1,
            color="tomato",
            alpha=0.5,
            line_width=0,
            # xlabel="",
            tools=["hover"],
        )
        #"""

        # @pn.io.profile("update_ppp_figures")
        def update_ppp_figures(nppc_fin):
            # update the plot of sky distribution of pointings and targets
            p_ppc_polygon = hv.Polygons(df_polygon.iloc[:nppc_fin, :]).opts(
                fill_color="darkgray",
                line_color="dimgray",
                line_width=0.5,
                alpha=0.2,
            )
            p_ppc_center = hv.Scatter(
                obj_allo2_for_ppcplot.iloc[:nppc_fin, :],
                kdims=["RA"],
                vdims=["Dec", "PA"],
            ).opts(
                tools=["hover"],
                fill_color="lightgray",
                line_color="gray",
                size=10,
                marker="s",
                show_legend=False,
            )
            p_ppc_tot = (p_ppc_polygon * p_ppc_center * p_tgt).opts(
                title="Distributions of targets & pointing centers",
                xlabel="RA (deg)",
                ylabel="Dec (deg)",
                xlim=(ra_max, ra_min),
                ylim=(dec_min, dec_max),
                toolbar="left",
                active_tools=["box_zoom"],
                show_grid=True,
                shared_axes=False,
                legend_offset=(10, 0),
                legend_cols=legend_cols,
                height=int(plot_height * 0.75),
                # height=plot_height,
            )

            # update completion rates as a function of PPC ID
            p_comp_nppc = hv.VLine(nppc_fin).opts(
                color="gray", line_dash="dashed", line_width=5
            )
            p_comp_tot_fh = (p_comp_rate_fh * p_comp_nppc).opts(
                xlim=(0.5, len(obj_allo) + 0.5),
                ylim=(0, 105),
                show_grid=True,
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                legend_position="right",
                legend_offset=(10, -30),
                legend_cols=legend_cols,
                height=int(plot_height * 0.55),
                # height=plot_height,
            )
            p_comp_tot_n = (p_comp_rate_n * p_comp_nppc).opts(
                xlim=(0.5, len(obj_allo) + 0.5),
                ylim=(0, 105),
                show_grid=True,
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                legend_position="right",
                legend_offset=(10, -30),
                legend_cols=legend_cols,
                height=int(plot_height * 0.55),
                # height=plot_height,
            )

            """
            # update fiber efficiency as a function of PPC ID
            mean_FE = np.mean(obj_allo2["Fiber usage fraction (%)"][:nppc_fin])
            p_fibereff_mean = (
                hv.HLine(mean_FE).opts(
                    color="red",
                    line_width=3,
                )
            ) * (
                hv.Text(
                    int(len(cR) * 0.9),
                    mean_FE * 1.15,
                    # mean_FE * 1.5,
                    "{:.2f}%".format(mean_FE),
                ).opts(color="red")
            )
            p_fibereff_nppc = hv.VLine(nppc_fin - 0.5).opts(
                color="gray", line_dash="dashed", line_width=5
            )

            ymax_fibereff = max(obj_allo2["Fiber usage fraction (%)"][:nppc_fin]) * 1.25

            p_fibereff_tot = (p_fibereff_bar * p_fibereff_mean * p_fibereff_nppc).opts(
                fontsize={"xticks": "0pt"},
                # TODO: xlim with hvplot's bar chart does not work properly.
                # ref: https://github.com/holoviz/hvplot/issues/946
                # xlim=(0, len(obj_allo) + 10),
                ylim=(0, ymax_fibereff),
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                height=plot_height,
            )
            #"""

            # return after putting all plots into a column
            return pn.Column(
                pn.panel(p_comp_tot_fh, linked_axes=False, width=600),
                pn.panel(p_comp_tot_n, linked_axes=False, width=600),
                # pn.panel(p_fibereff_tot, linked_axes=False, width=600),
                pn.panel(p_ppc_tot, linked_axes=False, width=600),
            )

        # @pn.io.profile("ppp_res_tab1")
        def ppp_res_tab1(nppc_fin):
            hour_tot = nppc_fin * single_exptime / 3600.0  # hour
            Fhour_tot = (
                sum([len(tt) for tt in obj_allo1[:nppc_fin]["allocated_targets"]])
                * single_exptime
                / 3600.0
            )  # fiber_count*hour
            Ttot_best = calc_total_obstime(nppc_fin, single_exptime)
            fib_eff_mean = np.mean(obj_allo1["Fiber usage fraction (%)"][:nppc_fin])
            fib_eff_small = (
                sum(obj_allo1["Fiber usage fraction (%)"][:nppc_fin] < 30)
                / nppc_fin
                * 100.0
            )

            cR1 = pd.DataFrame(
                dict(zip(name[:-1], cR[0][nppc_fin - 1])),
                index=[0],
            )

            cR1 = cR1.reindex(["P_all"] + [f"P_{p}" for p in range(10)], axis="columns")

            ppc_summary = pd.DataFrame(
                {
                    "resolution": [RESmode],
                    "N_ppc": [nppc_fin],
                    "Texp (h)": [hour_tot],
                    "Texp (fiberhour)": [Fhour_tot],
                    "Request time (h)": [Ttot_best],
                    "Used fiber fraction (%)": [fib_eff_mean],
                    "Fraction of PPC < 30% (%)": [fib_eff_small],
                },
            )

            ppc_summary_fin = pd.concat([ppc_summary, cR1], axis=1)

            return ppc_summary_fin

        # @pn.io.profile("ppp_res_tab2")
        def ppp_res_tab2():
            obj_alloc = obj_allo1[
                "ppc_code",
                "ppc_ra",
                "ppc_dec",
                "ppc_pa",
                "ppc_resolution",
                "ppc_priority",
                "Fiber usage fraction (%)",
            ]
            # normalize the priority of ppc to prevent too small value
            obj_alloc["ppc_priority"] = (
                obj_alloc["ppc_priority"] / max(obj_alloc["ppc_priority"]) * 1e3
            )
            df_ = Table.to_pandas(obj_alloc)
            df_["allocated_targets"] = [list(x) for x in obj_allo1["allocated_targets"]]
            return df_

        # compose figures
        p_result_fig = pn.Column(
            f"<font size=4><u>{RESmode.capitalize():s}-resolution mode</u></font>",
            pn.bind(update_ppp_figures, nppc),
        )

        # PPP summary table
        p_result_tab = pn.widgets.Tabulator(
            pn.bind(ppp_res_tab1, nppc),
        )

        # PPC table
        p_result_ppc = pn.widgets.Tabulator(
            # pn.bind(ppp_res_tab2, nppc),
            ppp_res_tab2(),
            visible=False,
            disabled=True,
        )
        return nppc, p_result_fig, p_result_tab, p_result_ppc

    # function starts here
    logger.info("start creating PPP figures")

    # initialize output elements
    nppc_l = None
    p_result_fig_l = None
    p_result_tab_l = None
    p_result_ppc_l = None
    nppc_m = None
    p_result_fig_m = None
    p_result_tab_m = None
    p_result_ppc_m = None

    # exit if no PPP outputs
    if len(obj_allo_l) == 0 and len(obj_allo_m) == 0:
        logger.info("No PPP results [ppp_result]")
        return (None, None, None, None)

    # exit if no assignment by netflow
    if len(obj_allo_l) > 0 and sum(obj_allo_l["tel_fiber_usage_frac"] == 0) == len(
        obj_allo_l
    ):
        logger.info("No netflow assignment (L) [ppp_result]")
        return (None, None, None, None)

    if len(obj_allo_m) > 0 and sum(obj_allo_m["tel_fiber_usage_frac"] == 0) == len(
        obj_allo_m
    ):
        logger.info("No netflow assignment (M) [ppp_result]")
        return (None, None, None, None)

    # generate figures and tables for low resolution
    if len(obj_allo_l) > 0:
        nppc_l, p_result_fig_l, p_result_tab_l, p_result_ppc_l = ppp_plotFig(
            "low", cR_l, sub_l, obj_allo_l, uS_L2
        )

    # generate figures and tables for medium resolution
    if len(obj_allo_m) > 0:
        nppc_m, p_result_fig_m, p_result_tab_m, p_result_ppc_m = ppp_plotFig(
            "medium", cR_m, sub_m, obj_allo_m, uS_M2
        )

    # define rows
    nppc_fin = pn.Row(max_width=900)
    p_result_fig_fin = pn.Row(max_width=900)

    # append components if it is not None
    for slider in [nppc_l, nppc_m]:
        if slider is not None:
            nppc_fin.append(slider)
    for fig in [p_result_fig_l, p_result_fig_m]:
        if fig is not None:
            p_result_fig_fin.append(fig)

    # @pn.io.profile("p_result_tab_tot")
    def p_result_tab_tot(p_result_tab_l, p_result_tab_m):
        ppc_sum = pd.concat([p_result_tab_l, p_result_tab_m], axis=0)
        loc_total = ppc_sum.index.size
        ppc_sum.loc[loc_total] = ppc_sum.sum(numeric_only=True)
        ppc_sum.loc[loc_total, "resolution"] = "Total"
        ppc_sum.iloc[loc_total, 6:] = np.nan
        for k in ppc_sum.columns:
            if ppc_sum.loc[:, k].isna().all():
                ppc_sum.drop(columns=[k], inplace=True)
        return ppc_sum

    # @pn.io.profile("p_result_ppc_tot")
    def p_result_ppc_tot(p_result_ppc_l, p_result_ppc_m):
        ppc_lst = pd.concat([p_result_ppc_l, p_result_ppc_m], axis=0)
        return ppc_lst

    p_result_tab = pn.widgets.Tabulator(
        pn.bind(p_result_tab_tot, p_result_tab_l, p_result_tab_m),
        theme="bootstrap",
        theme_classes=["table-sm"],
        pagination=None,
        visible=True,
        layout="fit_data_table",
        hidden_columns=["index"],
        selectable=False,
        header_align="right",
        configuration={"columnDefaults": {"headerSort": False}},
        disabled=True,
        stylesheets=[tabulator_stylesheet],
        max_height=150,
        formatters=tabulator_formatters,
    )

    # currently, this table is not displayed
    p_result_ppc_fin = pn.widgets.Tabulator(
        pn.bind(p_result_ppc_tot, p_result_ppc_l, p_result_ppc_m),
        visible=False,
        disabled=True,
    )

    logger.info("creating PPP figures finished ")

    return (nppc_fin, p_result_fig_fin, p_result_ppc_fin, p_result_tab)


#
# PPP result reproduction
#
def ppp_result_reproduce(
    obj_allo,
    uS,
    tab_psl,
    tab_tac,
    d_pfi=1.38,
    box_width=1200.0,
    plot_height=400,
):
    single_exptime = uS.meta["single_exptime"] if "single_exptime" in uS.meta else 900.0

    if "ppc_code" not in obj_allo.colnames:
        pn.state.notifications.error(
            "This submission is too old and not compatible for the operation.",
            duration=5000,
        )
        logger.error("'ppc_code' not found in the ppc_list. return None")
        return (None, None, None, None)

    # exit if no PPP outputs
    if None in obj_allo["ppc_code"]:
        logger.info(
            "[Reproduce] No PPP results due to running out of time [ppp_result]"
        )
        return (None, None, None, None)

    r_pfi = d_pfi / 2.0

    def complete_ppc(sample, point_l):
        if "allocated_targets" not in point_l.colnames:
            pn.state.notifications.error(
                "This submission is too old and not compatible for the operation.",
                duration=5000,
            )
            logger.error(
                "'allocated_targets' not found in the pointing list. raise exception."
            )
            raise KeyError  #

        sample["exptime_assign"] = 0
        sub_l = sorted(list(set(sample["priority"])))
        n_sub = len(sub_l)

        if len(point_l) == 0:
            return (
                sample,
                np.array([[0] * (n_sub + 1)]),
                np.array([[0] * (n_sub + 1)]),
                np.array([[0] * (n_sub + 1)]),
                np.array([[0] * (n_sub + 1)]),
                sub_l,
            )

        point_l_pri = point_l[
            point_l.argsort(keys="ppc_priority")
        ]  # sort ppc by its total priority == sum(weights of the assigned targets in ppc)

        # sub-groups of the input sample, catagarized by the user defined priority
        count_sub_fh = [sum(sample["exptime_PPP"]) / 3600.0] + [
            sum(sample[sample["priority"] == ll]["exptime_PPP"]) / 3600.0
            for ll in sub_l
        ]  # fiber hours
        count_sub_n = [len(sample)] + [
            sum(sample["priority"] == ll) for ll in sub_l
        ]  # number count of complete targets

        completeR_fh = []  # fiber hours
        completeR_fh_ = []  # percentage

        completeR_n = []  # number count of complete targets
        completeR_n_ = []  # percentage

        for ppc in point_l_pri:
            lst = np.where(np.in1d(sample["ob_code"], ppc["allocated_targets"]))[0]
            sample["exptime_assign"].data[lst] += single_exptime

            # achieved fiber hours (in total, in P[0-9])
            comT_t_fh = [sum(sample["exptime_assign"]) / 3600.0] + [
                sum(sample[sample["priority"] == ll]["exptime_assign"]) / 3600.0
                for ll in sub_l
            ]

            comp_s = np.where(sample["exptime_PPP"] <= sample["exptime_assign"])[0]
            comT_t_n = [len(comp_s)] + [
                sum(sample["priority"].data[comp_s] == ll) for ll in sub_l
            ]

            completeR_fh.append(comT_t_fh)
            completeR_fh_.append(
                [
                    comT_t_fh[oo] / count_sub_fh[oo] * 100
                    for oo in range(len(count_sub_fh))
                ]
            )

            completeR_n.append(comT_t_n)
            completeR_n_.append(
                [comT_t_n[oo] / count_sub_n[oo] * 100 for oo in range(len(count_sub_n))]
            )

        return (
            sample,
            np.array(completeR_fh),
            np.array(completeR_fh_),
            np.array(completeR_n),
            np.array(completeR_n_),
            sub_l,
        )

    def ppp_plotFig(RESmode, cR, sub, obj_allo, uS, nppc_usr, nppc_tac=0):
        def nppc2rot(nppc_):
            return calc_total_obstime(nppc_, single_exptime)

        def rot2nppc(rot):
            return calc_nppc_from_obstime(rot, single_exptime)

        if nppc_tac > 0:
            admin_slider_ini_value = nppc_tac.data[0]
        else:
            admin_slider_ini_value = nppc_usr.data[0]

        legend_cols = 2 if len(sub) >= 6 else 1

        nppc = pn.widgets.EditableFloatSlider(
            name=(f"{RESmode.capitalize()}-resolution mode (ROT / hour)"),
            format="1[.]000",
            value=nppc2rot(admin_slider_ini_value),
            step=nppc2rot(1),
            start=0,
            fixed_start=0,
            end=nppc2rot(len(cR[0])),
            fixed_end=nppc2rot(len(cR[0])),
            bar_color="gray",
            max_width=450,
            width=400,
        )

        name = ["P_all"] + ["P_" + str(int(ii)) for ii in sub] + ["PPC_id"]
        # colors for priority 0-9
        # red + first colors from glasbey_dark colormap as strings
        colors_all = ["red"] + cc.b_glasbey_bw_minc_20_maxl_70[:9]
        colors = [colors_all[i] for i in sub]

        obj_allo1 = obj_allo[obj_allo.argsort(keys="ppc_priority")]
        obj_allo1["PPC_id"] = np.arange(0, len(obj_allo), 1) + 1
        obj_allo2 = Table.to_pandas(obj_allo1)
        uS_ = Table.to_pandas(uS)

        # add a column to indicate the color for the scatter plot
        uS_["ppc_color"] = [colors_all[i] for i in uS_["priority"]]

        cR_fh_ = np.array([list(cR[0][ii]) + [ii + 1] for ii in range(len(cR[0]))])
        cR_fh__ = pd.DataFrame(dict(zip(name, cR_fh_.T)))

        cR_n_ = np.array([list(cR[1][ii]) + [ii + 1] for ii in range(len(cR[1]))])
        cR_n__ = pd.DataFrame(dict(zip(name, cR_n_.T)))

        # create polygons for PFS FoVs for each pointing
        ppc_coord = []
        # PA=0 along y-axis, PA=90 along x-axis, PA=180 along -y-axis...
        for ii in range(len(obj_allo1["ppc_ra"])):
            ppc_ra_t, ppc_dec_t, pa_t = (
                obj_allo1["ppc_ra"][ii],
                obj_allo1["ppc_dec"][ii],
                obj_allo1["ppc_pa"][ii],
            )
            center = SkyCoord(ppc_ra_t * u.deg, ppc_dec_t * u.deg)
            hexagon = center.directional_offset_by(
                np.array([deg + pa_t for deg in [30, 90, 150, 210, 270, 330, 30]])
                * u.deg,
                r_pfi * u.deg,
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
            ppc_coord_one = []
            for a, d in zip(ra_h, dec_h):
                ppc_coord_one += [a, d]
            ppc_coord.append([np.array(ppc_coord_one)])
        ppc_coord_polygon_array = PolygonArray(ppc_coord)
        df_polygon = sp.GeoDataFrame(({"polygons": ppc_coord_polygon_array}))

        #
        # The following p_ are static plots and neeed to be created only once
        #
        # for sky distributions
        ra_min = np.min([obj_allo1["ppc_ra"].min(), uS_["ra"].min()]) - d_pfi
        ra_max = np.max([obj_allo1["ppc_ra"].max(), uS_["ra"].max()]) + d_pfi
        dec_min = np.min([obj_allo1["ppc_dec"].min(), uS_["dec"].min()]) - d_pfi
        dec_max = np.max([obj_allo1["ppc_dec"].max(), uS_["dec"].max()]) + d_pfi
        p_tgt = uS_.hvplot.scatter(
            x="ra",
            y="dec",
            by="priority",
            color="ppc_color",
            marker="o",
            # s=20,
            s=60,
            line_color="white",
            line_width=0.5,
            legend=True,
        )

        obj_allo2_for_ppcplot = obj_allo2.rename(
            columns={"ppc_ra": "RA", "ppc_dec": "Dec", "ppc_pa": "PA"}
        )

        # static part for compolation rate plot
        p_comp_rate_fh = cR_fh__.hvplot.line(
            x="PPC_id",
            y=name[:-1],
            value_label="Completion rate (%)",
            title="Achieved fiberhours / total fiberhours (%)",
            color=["k"] + colors,
            line_width=[4, 3] + [2] * (len(sub) - 1),
            line_dash=["solid"] * 2 + ["dashed"] * (len(sub) - 1),
            legend="right",
        )
        p_comp_rate_n = cR_n__.hvplot.line(
            x="PPC_id",
            y=name[:-1],
            value_label="Completion rate (%)",
            title="N(fully complete targets) / N(targets) (%)",
            color=["k"] + colors,
            line_width=[4, 3] + [2] * (len(sub) - 1),
            line_dash=["solid"] * 2 + ["dashed"] * (len(sub) - 1),
            legend="right",
        )

        """
        # static part for fiber usage fraction plot
        p_fibereff_bar = hv.Bars(
            obj_allo2,
            kdims=["PPC_id"],
            vdims=["Fiber usage fraction (%)"],
        ).opts(
            title="Fiber usage fraction by pointing",
            width=1,
            color="tomato",
            alpha=0.5,
            line_width=0,
            # xlabel="",
            tools=["hover"],
        )
        #"""

        # @pn.io.profile("update_ppp_figures")
        def update_ppp_figures(nppc_fin):
            if nppc_fin > 0:
                # update the plot of sky distribution of pointings and targets
                p_ppc_polygon = hv.Polygons(df_polygon.iloc[:nppc_fin, :]).opts(
                    fill_color="darkgray",
                    line_color="dimgray",
                    line_width=0.5,
                    alpha=0.2,
                )
                p_ppc_center = hv.Scatter(
                    obj_allo2_for_ppcplot.iloc[:nppc_fin, :],
                    kdims=["RA"],
                    vdims=["Dec", "PA"],
                ).opts(
                    tools=["hover"],
                    fill_color="lightgray",
                    line_color="gray",
                    size=10,
                    marker="s",
                    show_legend=False,
                )
                p_ppc_tot = (p_ppc_polygon * p_ppc_center * p_tgt).opts(
                    title="Distributions of targets & pointing centers",
                    xlabel="RA (deg)",
                    ylabel="Dec (deg)",
                    xlim=(ra_max, ra_min),
                    ylim=(dec_min, dec_max),
                    toolbar="left",
                    active_tools=["box_zoom"],
                    show_grid=True,
                    shared_axes=False,
                    height=int(plot_height * 0.5),
                    legend_cols=legend_cols,
                )
            else:
                p_ppc_tot = (p_tgt).opts(
                    title="Distributions of targets & pointing centers",
                    xlabel="RA (deg)",
                    ylabel="Dec (deg)",
                    xlim=(ra_max, ra_min),
                    ylim=(dec_min, dec_max),
                    toolbar="left",
                    active_tools=["box_zoom"],
                    show_grid=True,
                    shared_axes=False,
                    height=plot_height,
                    legend_cols=legend_cols,
                )

            # update completion rates as a function of PPC ID
            p_comp_nppc = hv.VLine(nppc_fin).opts(
                color="gray", line_dash="dashed", line_width=5
            )
            p_comp_nppc_usr = hv.VLine(nppc_usr).opts(
                color="gray", line_dash="dotted", line_width=3
            )
            p_comp_nppc_tac = hv.VLine(nppc_tac).opts(
                color="red", line_dash="dotted", line_width=3
            )

            p_comp_tot_fh = (
                p_comp_rate_fh * p_comp_nppc * p_comp_nppc_usr * p_comp_nppc_tac
            ).opts(
                xlim=(0.5, len(obj_allo) + 0.5),
                ylim=(0, 105),
                show_grid=True,
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                height=int(plot_height * 0.5),
                legend_cols=legend_cols,
                legend_offset=(10, -30),
            )

            p_comp_tot_n = (
                p_comp_rate_n * p_comp_nppc * p_comp_nppc_usr * p_comp_nppc_tac
            ).opts(
                xlim=(0.5, len(obj_allo) + 0.5),
                ylim=(0, 105),
                show_grid=True,
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                height=int(plot_height * 0.5),
                legend_cols=legend_cols,
                legend_offset=(10, -30),
            )

            """
            # update fiber efficiency as a function of PPC ID
            mean_FE = np.mean(obj_allo2["Fiber usage fraction (%)"][:nppc_fin])
            p_fibereff_mean = (
                hv.HLine(mean_FE).opts(
                    color="red",
                    line_width=3,
                )
            ) * (
                hv.Text(
                    int(len(cR) * 0.9),
                    mean_FE * 1.15,
                    # mean_FE * 1.5,
                    "{:.2f}%".format(mean_FE),
                ).opts(color="red")
            )
            p_fibereff_nppc = hv.VLine(nppc_fin - 0.5).opts(
                color="gray", line_dash="dashed", line_width=5
            )
            p_fibereff_nppc_usr = hv.VLine(nppc_usr - 0.5).opts(
                color="gray", line_dash="dotted", line_width=3
            )

            ymax_fibereff = max(obj_allo2["Fiber usage fraction (%)"][:nppc_fin]) * 1.25

            p_fibereff_tot = (p_fibereff_bar * p_fibereff_mean * p_fibereff_nppc * p_fibereff_nppc_usr).opts(
                fontsize={"xticks": "0pt"},
                # TODO: xlim with hvplot's bar chart does not work properly.
                # ref: https://github.com/holoviz/hvplot/issues/946
                # xlim=(0, len(obj_allo) + 10),
                ylim=(0, ymax_fibereff),
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                height=plot_height,
            )
            #"""

            # return after putting all plots into a column
            return pn.Column(
                pn.panel(p_comp_tot_fh, linked_axes=False, width=500),
                pn.panel(p_comp_tot_n, linked_axes=False, width=500),
                # pn.panel(p_fibereff_tot, linked_axes=False, width=500),
                pn.panel(p_ppc_tot, linked_axes=False, width=500),
            )

        # @pn.io.profile("ppp_res_tab1")
        def ppp_res_tab1(nppc_fin):
            hour_tot = nppc_fin * single_exptime / 3600.0  # hour
            Ttot_best = calc_total_obstime(nppc_fin, single_exptime)

            if nppc_fin > 0:
                Fhour_tot = (
                    sum([len(tt) for tt in obj_allo1[:nppc_fin]["allocated_targets"]])
                    * single_exptime
                    / 3600.0
                )  # fiber_count*hour
                fib_eff_mean = np.mean(obj_allo1["Fiber usage fraction (%)"][:nppc_fin])
                fib_eff_small = (
                    sum(obj_allo1["Fiber usage fraction (%)"][:nppc_fin] < 30)
                    / nppc_fin
                    * 100.0
                )

                cR1 = pd.DataFrame(
                    dict(zip(name[:-1], cR[0][nppc_fin - 1])),
                    index=[0],
                )
            else:
                Fhour_tot = 0
                fib_eff_mean = 0
                fib_eff_small = 0

                cR1 = pd.DataFrame(
                    dict(zip(name[:-1], [0] * 11)),
                    index=[0],
                )

            cR1 = cR1.reindex(["P_all"] + [f"P_{p}" for p in range(10)], axis="columns")

            ppc_summary = pd.DataFrame(
                {
                    "resolution": [RESmode],
                    "N_ppc": [nppc_fin],
                    "Texp (h)": [hour_tot],
                    "Texp (fiberhour)": [Fhour_tot],
                    "Request time (h)": [Ttot_best],
                    "Used fiber fraction (%)": [fib_eff_mean],
                    "Fraction of PPC < 30% (%)": [fib_eff_small],
                },
            )

            ppc_summary_fin = pd.concat([ppc_summary, cR1], axis=1)

            return ppc_summary_fin

        # @pn.io.profile("ppp_res_tab2")
        def ppp_res_tab2(nppc_fin):
            obj_alloc = obj_allo1[:nppc_fin]
            return Table.to_pandas(obj_alloc)

        # compose figures
        p_result_fig = pn.Column(
            f"<font size=4><u>{RESmode.capitalize():s}-resolution mode</u></font>",
            pn.bind(update_ppp_figures, pn.bind(rot2nppc, nppc)),
        )

        # PPP summary table
        p_result_tab = pn.widgets.Tabulator(
            pn.bind(ppp_res_tab1, pn.bind(rot2nppc, nppc)),
        )

        # PPC table
        p_result_ppc = pn.widgets.Tabulator(
            pn.bind(ppp_res_tab2, pn.bind(rot2nppc, nppc)),
            visible=False,
            disabled=True,
        )
        return nppc, p_result_fig, p_result_tab, p_result_ppc

    # function starts here
    logger.info("[Reproduce] start creating PPP figures")

    # initialize output elements
    nppc_l = None
    p_result_fig_l = None
    p_result_tab_l = None
    p_result_ppc_l = None
    nppc_m = None
    p_result_fig_m = None
    p_result_tab_m = None
    p_result_ppc_m = None

    tabulator_stylesheet = """
    .tabulator-row-odd { background-color: #ffffff !important; }
    .tabulator-row-even { background-color: #ffffff !important; }
    .tabulator-row-odd:hover { color: #000000 !important; background-color: #ffffff !important; }
    .tabulator-row-even:hover { color: #000000 !important; background-color: #ffffff !important; }
    """

    # add styling/formatting to the table
    tabulator_formatters = {
        "N_ppc": NumberFormatter(format="0", text_align="right"),
        "Texp (h)": NumberFormatter(format="0.00", text_align="right"),
        "Texp (fiberhour)": NumberFormatter(format="0.00", text_align="right"),
        "Request time (h)": NumberFormatter(format="0.00", text_align="right"),
        "Used fiber fraction (%)": NumberFormatter(format="0.000", text_align="right"),
        "Fraction of PPC < 30% (%)": NumberFormatter(format="0.0", text_align="right"),
    }
    for p in ["all"] + np.arange(10).tolist():
        tabulator_formatters[f"P_{p}"] = NumberFormatter(
            format="0.0", text_align="right"
        )

    exptime_ppp = np.ceil(uS["exptime"] / single_exptime) * single_exptime
    uS.add_column(exptime_ppp, name="exptime_PPP")

    uS_L = uS[uS["resolution"] == "L"]
    uS_M = uS[uS["resolution"] == "M"]

    obj_allo_l = obj_allo[obj_allo["ppc_resolution"] == "L"]
    obj_allo_m = obj_allo[obj_allo["ppc_resolution"] == "M"]

    # generate figures and tables for low resolution
    if len(uS_L) > 0:
        uS_L_, cR_l_fh, cR_l_fh_, cR_l_n, cR_l_n_, sub_l = complete_ppc(
            uS_L, obj_allo_l
        )
        nppc_usr_l = tab_psl[tab_psl["resolution"] == "low"]["N_ppc"]

        if len(tab_tac) > 0:
            nppc_tac_l = tab_tac[tab_tac["resolution"] == "low"]["N_ppc"]
        else:
            nppc_tac_l = 0

        nppc_l, p_result_fig_l, p_result_tab_l, p_result_ppc_l = ppp_plotFig(
            "low", [cR_l_fh_, cR_l_n_], sub_l, obj_allo_l, uS_L_, nppc_usr_l, nppc_tac_l
        )

    # generate figures and tables for medium resolution
    if len(uS_M) > 0:
        uS_M_, cR_m_fh, cR_m_fh_, cR_m_n, cR_m_n_, sub_m = complete_ppc(
            uS_M, obj_allo_m
        )
        nppc_usr_m = tab_psl[tab_psl["resolution"] == "medium"]["N_ppc"]

        if len(tab_tac) > 0:
            nppc_tac_m = tab_tac[tab_tac["resolution"] == "medium"]["N_ppc"]
        else:
            nppc_tac_m = 0

        nppc_m, p_result_fig_m, p_result_tab_m, p_result_ppc_m = ppp_plotFig(
            "medium",
            [cR_m_fh_, cR_m_n_],
            sub_m,
            obj_allo_m,
            uS_M_,
            nppc_usr_m,
            nppc_tac_m,
        )

    # define rows
    nppc_fin = pn.Row(max_width=900)
    p_result_fig_fin = pn.Row(max_width=900)

    # append components if it is not None
    for slider in [nppc_l, nppc_m]:
        if slider is not None:
            nppc_fin.append(slider)
    for fig in [p_result_fig_l, p_result_fig_m]:
        if fig is not None:
            p_result_fig_fin.append(fig)

    # @pn.io.profile("p_result_tab_tot")
    def p_result_tab_tot(p_result_tab_l, p_result_tab_m):
        ppc_sum = pd.concat([p_result_tab_l, p_result_tab_m], axis=0, ignore_index=True)
        loc_total = ppc_sum.index.size
        ppc_sum.loc[loc_total] = ppc_sum.sum(numeric_only=True)
        ppc_sum.loc[loc_total, "resolution"] = "Total"
        ppc_sum.iloc[loc_total, 6:] = np.nan
        for k in ppc_sum.columns:
            if ppc_sum.loc[:, k].isna().all():
                ppc_sum.drop(columns=[k], inplace=True)
        return ppc_sum

    # @pn.io.profile("p_result_ppc_tot")
    def p_result_ppc_tot(p_result_ppc_l, p_result_ppc_m):
        ppc_lst = pd.concat([p_result_ppc_l, p_result_ppc_m], axis=0, ignore_index=True)
        return ppc_lst

    p_result_tab = pn.widgets.Tabulator(
        pn.bind(p_result_tab_tot, p_result_tab_l, p_result_tab_m),
        theme="bootstrap",
        theme_classes=["table-sm"],
        pagination=None,
        visible=True,
        layout="fit_data_table",
        hidden_columns=["index"],
        selectable=False,
        header_align="right",
        configuration={"columnDefaults": {"headerSort": False}},
        disabled=True,
        stylesheets=[tabulator_stylesheet],
        max_height=150,
        formatters=tabulator_formatters,
    )

    # PPC list shown in the left of the main panel
    p_result_ppc_fin = pn.widgets.Tabulator(
        pn.bind(p_result_ppc_tot, p_result_ppc_l, p_result_ppc_m),
        visible=True,
        disabled=True,
        page_size=20,
        theme="bootstrap",
        theme_classes=["table-striped"],
        pagination="remote",
        header_filters=True,
        layout="fit_columns",
        selectable=False,
        hidden_columns=[
            "index",
            "Fiber usage fraction (%)",
            "allocated_targets",
            "PPC_id",
        ],
        width=650,
        height=850,
    )

    logger.info("[Reproduce] creating PPP figures finished ")

    return (nppc_fin, p_result_fig_fin, p_result_ppc_fin, p_result_tab)
