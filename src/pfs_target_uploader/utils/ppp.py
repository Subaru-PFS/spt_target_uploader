#!/usr/bin/env python3

import random
import time
import warnings
from functools import partial
from itertools import chain

import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa need to run pandas.DataFrame.hvplot
import matplotlib.pyplot as plt
import multiprocess
import numpy as np
import pandas as pd
import panel as pn
import spatialpandas as sp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from bokeh.models.widgets.tables import NumberFormatter
from logzero import logger
from matplotlib.path import Path
from sklearn.cluster import DBSCAN, AgglomerativeClustering
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


def PPPrunStart(uS, weight_para, exetime, d_pfi=1.38):
    r_pfi = d_pfi / 2.0

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

    def target_DBSCAN(sample, sep=d_pfi):
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

    def PPP_centers(sample_f, mutiPro, weight_para, starttime, exetime):
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
        Nfiber = int(2394 - 200)  # 200 for calibrators
        sample_f = count_N(sample_f)
        sample_f = weight(sample_f, conta, contb, contc)

        if (time.time() - starttime) > exetime:
            sample_f.meta["PPC"] = []
            status = 1
            logger.info("PPP stopped since the time is running out [PPP_centers s1]")
            return sample_f, status

        peak = []

        for sample in target_DBSCAN(sample_f, d_pfi):
            if (time.time() - starttime) > exetime:
                status = 1
                logger.info(
                    "PPP stopped since the time is running out [PPP_centers s2]"
                )
                continue

            sample_s = sample[sample["exptime_PPP"] > 0]  # targets not finished

            while any(sample_s["exptime_PPP"] > 0):
                if (time.time() - starttime) > exetime:
                    status = 1
                    logger.info(
                        "PPP stopped since the time is running out [PPP_centers s2_1]"
                    )
                    break

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
            np.fliplr(np.radians(ppc_xy[:, [1, 2]]))
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
            targetL.append(nf.ScienceTarget(id_, ra, dec, tm, int_, "sci"))
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

        int_ = 0
        for ii in sample:
            classdict["sci_P" + str(int_)] = {
                "nonObservationCost": ii["weight"],
                "partialObservationCost": ii["weight"] * 1.5,
                "calib": False,
            }
            int_ += 1

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

        if len(sample.meta["PPC"]) == 0:
            point_t = []
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
        sample.add_column(0, name="allocate_time")
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

    def netflow_iter(uS, obj_allo, weight_para, starttime, exetime):
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

        status = 999

        if sum(uS["allocate_time"] == uS["exptime_PPP"]) == len(uS):
            # remove ppc with no fiber assignment
            obj_allo.remove_rows(np.where(obj_allo["tel_fiber_usage_frac"] == 0)[0])
            return obj_allo, status

        elif (time.time() - starttime) > exetime:
            status = 1
            logger.info("PPP stopped since the time is running out [netflow_iter s1]")
            return obj_allo, status

        else:
            #  select non-assigned targets --> PPC determination --> netflow --> if no fibre assigned: shift PPC
            iter_m2 = 0

            while any(uS["allocate_time"] < uS["exptime_PPP"]) and iter_m2 < 10:
                if (time.time() - starttime) > exetime:
                    status = 1
                    logger.info(
                        "PPP stopped since the time is running out [netflow_iter s2]"
                    )
                    break

                uS_t1 = uS[uS["allocate_time"] < uS["exptime_PPP"]]
                uS_t1["exptime_PPP"] = (
                    uS_t1["exptime_PPP"] - uS_t1["allocate_time"]
                )  # remained exposure time
                uS_t1.remove_column("allocate_time")

                uS_t2 = PPP_centers(uS_t1, True, weight_para, starttime, exetime)[0]

                obj_allo_t = netflowRun(uS_t2)

                if len(obj_allo) > 35 * 4 or iter_m2 >= 10:
                    # stop if n_ppc>35 * 4
                    return obj_allo, status

                else:
                    obj_allo = vstack([obj_allo, obj_allo_t])
                    obj_allo.remove_rows(
                        np.where(obj_allo["tel_fiber_usage_frac"] == 0)[0]
                    )
                    uS = complete_ppc(uS_t2, obj_allo)[0]
                    iter_m2 += 1

            return obj_allo, status

    # computation starts here
    logger.info("PPP run started")
    t_ppp_start = time.time()

    exptime_ppp = np.ceil(uS["exptime"] / 900) * 900
    uS.add_column(exptime_ppp, name="exptime_PPP")

    uS_L = uS[uS["resolution"] == "L"]
    uS_M = uS[uS["resolution"] == "M"]

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

    if len(uS_L) > 0 and len(uS_M) == 0:
        uS_L_s2, status_ = PPP_centers(uS_L, True, weight_para, t_ppp_start, exetime)
        obj_allo_L = netflowRun(uS_L_s2)
        uS_L2 = complete_ppc(uS_L_s2, obj_allo_L)[0]
        obj_allo_L_fin, status_ = netflow_iter(
            uS_L2, obj_allo_L, weight_para, t_ppp_start, exetime
        )

        uS_L_s2.remove_column("allocate_time")
        uS_L2, cR_L, cR_L_, sub_l = complete_ppc(uS_L_s2, obj_allo_L_fin)

        out_uS_L2 = uS_L2
        out_cR_L = cR_L
        out_cR_L_ = cR_L_
        out_sub_l = sub_l
        out_obj_allo_L_fin = obj_allo_L_fin

    if len(uS_M) > 0 and len(uS_L) == 0:
        uS_M_s2, status_ = PPP_centers(uS_M, True, weight_para, t_ppp_start, exetime)
        obj_allo_M = netflowRun(uS_M_s2)
        uS_M2 = complete_ppc(uS_M_s2, obj_allo_M)[0]
        obj_allo_M_fin, status_ = netflow_iter(
            uS_M2, obj_allo_M, weight_para, t_ppp_start, exetime
        )

        uS_M_s2.remove_column("allocate_time")
        uS_M2, cR_M, cR_M_, sub_m = complete_ppc(uS_M_s2, obj_allo_M_fin)

        out_uS_M2 = uS_M2
        out_cR_M = cR_M
        out_cR_M_ = cR_M_
        out_sub_m = sub_m
        out_obj_allo_M_fin = obj_allo_M_fin

    if len(uS_L) > 0 and len(uS_M) > 0:
        uS_L_s2, status_ = PPP_centers(uS_L, True, weight_para, t_ppp_start, exetime)
        obj_allo_L = netflowRun(uS_L_s2)
        uS_L2 = complete_ppc(uS_L_s2, obj_allo_L)[0]
        obj_allo_L_fin, status_ = netflow_iter(
            uS_L2, obj_allo_L, weight_para, t_ppp_start, exetime
        )

        uS_L_s2.remove_column("allocate_time")
        uS_L2, cR_L, cR_L_, sub_l = complete_ppc(uS_L_s2, obj_allo_L_fin)

        uS_M_s2, status_ = PPP_centers(uS_M, True, weight_para, t_ppp_start, exetime)
        obj_allo_M = netflowRun(uS_M_s2)
        uS_M2 = complete_ppc(uS_M_s2, obj_allo_M)[0]
        obj_allo_M_fin, status_ = netflow_iter(
            uS_M2, obj_allo_M, weight_para, t_ppp_start, exetime
        )

        uS_M_s2.remove_column("allocate_time")
        uS_M2, cR_M, cR_M_, sub_m = complete_ppc(uS_M_s2, obj_allo_M_fin)

        out_uS_L2 = uS_L2
        out_cR_L = cR_L
        out_cR_L_ = cR_L_
        out_sub_l = sub_l
        out_obj_allo_L_fin = obj_allo_L_fin
        out_uS_M2 = uS_M2
        out_cR_M = cR_M
        out_cR_M_ = cR_M_
        out_sub_m = sub_m
        out_obj_allo_M_fin = obj_allo_M_fin

    t_ppp_stop = time.time()
    logger.info(f"PPP run finished in {t_ppp_stop-t_ppp_start:.1f} seconds")
    logger.info(f"PPP running status: {status_:.0f}")

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
    d_pfi=1.38,
    box_width=1200.0,
    plot_height=400,
):
    # exit if no PPP outputs
    if len(obj_allo_l) == 0 and len(obj_allo_m) == 0:
        logger.info("No PPP results due to running out of time [ppp_result]")
        return (None, None, None, None)

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

    def overheads(n_sci_frame):
        # in seconds
        t_exp_sci: float = 900.0
        t_overhead_misc: float = 60.0
        t_overhead_fiber: float = 180.0

        Toverheads_tot_best = (
            t_exp_sci + t_overhead_misc + t_overhead_fiber
        ) * n_sci_frame

        return Toverheads_tot_best / 3600.0

    def ppp_plotFig(RESmode, cR, sub, obj_allo, uS):
        nppc = pn.widgets.EditableIntSlider(
            name=(f"{RESmode.capitalize()}-resolution mode"),
            value=len(cR),
            step=1,
            start=1,
            end=len(cR),
            fixed_start=1,
            fixed_end=len(cR),
            bar_color="gray",
            max_width=450,
        )

        name = ["P_all"] + ["P_" + str(int(ii)) for ii in sub] + ["PPC_id"]
        # colors for priority 0-9
        # red + first colors from glasbey_dark colormap as strings
        colors_all = ["red"] + cc.b_glasbey_bw_minc_20_maxl_70[:9]
        colors = [colors_all[i] for i in sub]

        obj_allo1 = obj_allo[obj_allo.argsort(keys="ppc_priority")]
        obj_allo1["PPC_id"] = np.arange(0, len(obj_allo), 1) + 1
        obj_allo1["ppc_code"] = [
            "Point_" + RESmode + "_" + str(count)
            for count in (np.arange(0, len(obj_allo), 1) + 1)
        ]
        obj_allo1.rename_column("tel_fiber_usage_frac", "Fiber usage fraction (%)")
        obj_allo2 = Table.to_pandas(obj_allo1)
        uS_ = Table.to_pandas(uS)

        # add a column to indicate the color for the scatter plot
        uS_["ppc_color"] = [colors_all[i] for i in uS_["priority"]]

        cR_ = np.array([list(cR[ii]) + [ii + 1] for ii in range(len(cR))])
        cR__ = pd.DataFrame(dict(zip(name, cR_.T)))

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
        p_comp_rate = cR__.hvplot.line(
            x="PPC_id",
            y=name[:-1],
            value_label="Completion rate (%)",
            title="Progress of the completion rate",
            color=["k"] + colors,
            line_width=[4, 3] + [2] * (len(sub) - 1),
            line_dash=["solid"] * 2 + ["dashed"] * (len(sub) - 1),
            legend="right",
        )
        # p_comp_rect1 = hv.Rectangles([(30, 88, 95, 100)]).opts(
        #    color="orange", line_width=0, alpha=0.2
        # )
        # p_comp_rect2 = hv.Rectangles([(20, 43, 130, 93)]).opts(
        #    color="dodgerblue", line_width=0, alpha=0.2
        # )

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

        @pn.io.profile("update_ppp_figures")
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
                height=plot_height,
            )

            # update completion rates as a function of PPC ID
            p_comp_nppc = hv.VLine(nppc_fin).opts(
                color="gray", line_dash="dashed", line_width=5
            )
            """
            p_comp_tot = (p_comp_rate * p_comp_rect1 * p_comp_rect2 * p_comp_nppc).opts(
                xlim=(0.5, len(obj_allo) + 0.5),
                ylim=(0, 105),
                show_grid=True,
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                height=plot_height,
            )
            #"""
            p_comp_tot = (p_comp_rate * p_comp_nppc).opts(
                xlim=(0.5, len(obj_allo) + 0.5),
                ylim=(0, 105),
                show_grid=True,
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                height=plot_height,
            )

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

            # return after putting all plots into a column
            return pn.Column(
                pn.panel(p_comp_tot, linked_axes=False, width=600),
                pn.panel(p_fibereff_tot, linked_axes=False, width=600),
                pn.panel(p_ppc_tot, linked_axes=False, width=600),
            )

        @pn.io.profile("ppp_res_tab1")
        def ppp_res_tab1(nppc_fin):
            hour_tot = nppc_fin * 15.0 / 60.0  # hour
            Fhour_tot = (
                sum([len(tt) for tt in obj_allo1[:nppc_fin]["allocated_targets"]])
                * 15.0
                / 60.0
            )  # fiber_count*hour
            Ttot_best = overheads(nppc_fin)
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

        @pn.io.profile("ppp_res_tab2")
        def ppp_res_tab2():
            obj_alloc = obj_allo1[
                "ppc_code",
                "ppc_ra",
                "ppc_dec",
                "ppc_pa",
                "ppc_resolution",
                "ppc_priority",
                "Fiber usage fraction (%)",
                "allocated_targets",
            ]
            # normalize the priority of ppc to prevent too small value
            obj_alloc["ppc_priority"] = (
                obj_alloc["ppc_priority"] / max(obj_alloc["ppc_priority"]) * 1e3
            )
            return Table.to_pandas(obj_alloc)

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

    # generate figures and tables for low resolution
    if len(cR_l) > 0:
        nppc_l, p_result_fig_l, p_result_tab_l, p_result_ppc_l = ppp_plotFig(
            "low", cR_l, sub_l, obj_allo_l, uS_L2
        )

    # generate figures and tables for medium resolution
    if len(cR_m) > 0:
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

    @pn.io.profile("p_result_tab_tot")
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

    @pn.io.profile("p_result_ppc_tot")
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


## PPP result reproduction


def ppp_result_reproduce(
    obj_allo,
    uS,
    d_pfi=1.38,
    box_width=1200.0,
    plot_height=220,
):
    # exit if no PPP outputs
    if None in obj_allo["ppc_code"]:
        logger.info(
            "[Reproduce] No PPP results due to running out of time [ppp_result]"
        )
        return (None, None, None, None)

    r_pfi = d_pfi / 2.0

    def complete_ppc(sample, point_l):
        sample.add_column(0, name="allocate_time")
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

    def overheads(n_sci_frame):
        # in seconds
        t_exp_sci: float = 900.0
        t_overhead_misc: float = 60.0
        t_overhead_fiber: float = 180.0

        Toverheads_tot_best = (
            t_exp_sci + t_overhead_misc + t_overhead_fiber
        ) * n_sci_frame

        return Toverheads_tot_best / 3600.0

    def ppp_plotFig(RESmode, cR, sub, obj_allo, uS):
        def nppc2rot(nppc_):
            # in seconds
            t_exp_sci: float = 900.0
            t_overhead_misc: float = 60.0
            t_overhead_fiber: float = 180.0

            Toverheads_tot_best = (
                t_exp_sci + t_overhead_misc + t_overhead_fiber
            ) * nppc_

            return Toverheads_tot_best / 3600.0

        def rot2nppc(rot):
            # in seconds
            t_exp_sci: float = 900.0
            t_overhead_misc: float = 60.0
            t_overhead_fiber: float = 180.0

            nppc_ = rot * 3600.0 / (t_exp_sci + t_overhead_misc + t_overhead_fiber)

            return int(np.floor(nppc_))

        nppc = pn.widgets.FloatSlider(
            name=(f"{RESmode.capitalize()}-resolution mode (ROT / hour)"),
            value=nppc2rot(len(cR)),
            step=0.1,
            start=nppc2rot(1),
            end=nppc2rot(len(cR)),
            bar_color="gray",
            max_width=450,
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

        cR_ = np.array([list(cR[ii]) + [ii + 1] for ii in range(len(cR))])
        cR__ = pd.DataFrame(dict(zip(name, cR_.T)))

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
        p_comp_rate = cR__.hvplot.line(
            x="PPC_id",
            y=name[:-1],
            value_label="Completion rate (%)",
            title="Progress of the completion rate",
            color=["k"] + colors,
            line_width=[4, 3] + [2] * (len(sub) - 1),
            line_dash=["solid"] * 2 + ["dashed"] * (len(sub) - 1),
            legend="right",
        )

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

        @pn.io.profile("update_ppp_figures")
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
                height=plot_height,
            )

            # update completion rates as a function of PPC ID
            p_comp_nppc = hv.VLine(nppc_fin).opts(
                color="gray", line_dash="dashed", line_width=5
            )

            p_comp_tot = (p_comp_rate * p_comp_nppc).opts(
                xlim=(0.5, len(obj_allo) + 0.5),
                ylim=(0, 105),
                show_grid=True,
                shared_axes=False,
                toolbar="left",
                active_tools=["box_zoom"],
                height=plot_height,
            )

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

            # return after putting all plots into a column
            return pn.Column(
                pn.panel(p_comp_tot, linked_axes=False, width=500),
                pn.panel(p_fibereff_tot, linked_axes=False, width=500),
                pn.panel(p_ppc_tot, linked_axes=False, width=500),
            )

        @pn.io.profile("ppp_res_tab1")
        def ppp_res_tab1(nppc_fin):
            hour_tot = nppc_fin * 15.0 / 60.0  # hour
            Fhour_tot = (
                sum([len(tt) for tt in obj_allo1[:nppc_fin]["allocated_targets"]])
                * 15.0
                / 60.0
            )  # fiber_count*hour
            Ttot_best = overheads(nppc_fin)
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

        @pn.io.profile("ppp_res_tab2")
        def ppp_res_tab2(nppc_fin):
            obj_alloc = obj_allo1[:nppc_fin][
                "ppc_code",
                "ppc_ra",
                "ppc_dec",
                "ppc_pa",
                "ppc_resolution",
                "ppc_priority",
            ]
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
            page_size=30,
            theme="bootstrap",
            theme_classes=["table-striped"],
            pagination="remote",
            header_filters=True,
            layout="fit_columns",
            hidden_columns=["index"],
            width=650,
            height=800,
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

    exptime_ppp = np.ceil(uS["exptime"] / 900) * 900
    uS.add_column(exptime_ppp, name="exptime_PPP")

    uS_L = uS[uS["resolution"] == "L"]
    uS_M = uS[uS["resolution"] == "M"]

    obj_allo_l = obj_allo[obj_allo["ppc_resolution"] == "L"]
    obj_allo_m = obj_allo[obj_allo["ppc_resolution"] == "M"]

    # generate figures and tables for low resolution
    if len(uS_L) > 0:
        uS_L_, cR_l, cR_l_, sub_l = complete_ppc(uS_L, obj_allo_l)

        nppc_l, p_result_fig_l, p_result_tab_l, p_result_ppc_l = ppp_plotFig(
            "low", cR_l_, sub_l, obj_allo_l, uS_L_
        )

    # generate figures and tables for medium resolution
    if len(uS_M) > 0:
        uS_M_, cR_m, cR_m_, sub_m = complete_ppc(uS_M, obj_allo_m)

        nppc_m, p_result_fig_m, p_result_tab_m, p_result_ppc_m = ppp_plotFig(
            "medium", cR_m_, sub_m, obj_allo_m, uS_M_
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

    @pn.io.profile("p_result_tab_tot")
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

    @pn.io.profile("p_result_ppc_tot")
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
        visible=True,
        disabled=True,
    )

    logger.info("[Reproduce] creating PPP figures finished ")

    return (nppc_fin, p_result_fig_fin, p_result_ppc_fin, p_result_tab)
