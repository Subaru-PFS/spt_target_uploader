#!/usr/bin/env python3

import re
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from dateutil import parser, tz
from loguru import logger

# below for qplan
# isort: split
from qplan.entity import StaticTarget
from qplan.util.eph_cache import EphemerisCache
from qplan.util.site import site_subaru as observer

from . import (
    arm_values,
    filter_category,
    filter_keys,
    optional_keys,
    optional_keys_default,
    required_keys,
    target_datatype,
)

warnings.filterwarnings("ignore")


def get_semester_daterange(dt, current=False, next=True):
    if current and next:
        logger.error("current and next cannot be True at the same time")
        raise ValueError

    if (not current) and (not next):
        logger.error("current and next cannot be False at the same time")
        raise ValueError

    if current:
        if (dt.month >= 2) and (dt.month <= 7):
            semester_begin = datetime(dt.year, 2, 1)
            semester_end = datetime(dt.year, 7, 31)
        elif (dt.month >= 8) and (dt.month <= 12):
            semester_begin = datetime(dt.year, 8, 1)
            semester_end = datetime(dt.year + 1, 1, 31)
        elif dt.month == 1:
            semester_begin = datetime(dt.year - 1, 8, 1)
            semester_end = datetime(dt.year, 1, 31)

    if next:
        if (dt.month >= 2) and (dt.month <= 7):
            semester_begin = datetime(dt.year, 8, 1)
            semester_end = datetime(dt.year + 1, 1, 31)
        elif (dt.month >= 8) and (dt.month <= 12):
            semester_begin = datetime(dt.year + 1, 2, 1)
            semester_end = datetime(dt.year + 1, 7, 31)
        elif dt.month == 1:
            semester_begin = datetime(dt.year, 2, 1)
            semester_end = datetime(dt.year, 7, 31)

    return semester_begin, semester_end


def visibility_checker(uS, date_begin=None, date_end=None):
    if len(uS) == 0:
        return np.array([])

    tz_HST = tz.gettz("US/Hawaii")

    eph_cache = EphemerisCache(logger, precision_minutes=15)

    # set next semester if there is no range is defined.
    tmp_begin, tmp_end = get_semester_daterange(datetime.now(tz=tz_HST), next=True)

    if date_begin is None:
        date_begin = tmp_begin
    if date_end is None:
        date_end = tmp_end

    logger.info(f"Observation period start on {date_begin:%Y-%m-%d}")
    logger.info(f"Observation period end on {date_end:%Y-%m-%d}")

    daterange = pd.date_range(date_begin, date_end + timedelta(days=1))

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

            # observer.set_date(night_begin)

            eph_key = target

            obs_ok, t_start, t_stop = eph_cache.observable(
                eph_key,
                target,
                observer,
                night_begin,
                night_end,
                min_el,
                max_el,
                total_time,
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


def visibility_checker_vec(
    df: pd.DataFrame,
    date_begin: datetime | None = None,
    date_end: datetime | None = None,
    min_el: float = 30.0,
    max_el: float = 85.0,
) -> np.ndarray:
    if df.index.size == 0:
        return np.array([], dtype=bool)

    # set timezone to HST
    tz_HST = tz.gettz("US/Hawaii")

    # set next semester if there is no range is defined.
    tmp_begin, tmp_end = get_semester_daterange(datetime.now(tz=tz_HST), next=True)

    if date_begin is None:
        date_begin = tmp_begin
    if date_end is None:
        date_end = tmp_end

    logger.info(f"Observation period start on {date_begin:%Y-%m-%d}")
    logger.info(f"Observation period end on {date_end:%Y-%m-%d}")

    # include the last date instead of removing it
    date_middle = date_begin + (date_end - date_begin) / 2
    logger.info(f"Observation period is divided into two at {date_middle}")

    daterange_1 = pd.date_range(
        date_begin,
        date_middle,
        tz=tz_HST,
    )
    daterange_2 = pd.date_range(
        date_middle + timedelta(days=1),
        date_end + timedelta(days=1),
        tz=tz_HST,
    )

    logger.debug(
        f"Observation period is divided into two: {daterange_1} and {daterange_2}"
    )

    logger.debug(f"{len(daterange_1)} and {len(daterange_2)}")

    dates_begin_1, dates_end_1 = daterange_1[:-1], daterange_1[1:]
    dates_begin_2, dates_end_2 = daterange_2[:-1], daterange_2[1:]

    nights_begin_1 = [
        parser.parse(d.strftime("%Y-%m-%d") + " 18:30:00").replace(tzinfo=tz_HST)
        for d in dates_begin_1
    ]
    nights_end_1 = [
        parser.parse(d.strftime("%Y-%m-%d") + " 05:30:00").replace(tzinfo=tz_HST)
        for d in dates_end_1
    ]

    # reverse order for the second half
    nights_begin_2 = [
        parser.parse(d.strftime("%Y-%m-%d") + " 18:30:00").replace(tzinfo=tz_HST)
        for d in dates_begin_2[::-1]
    ]
    nights_end_2 = [
        parser.parse(d.strftime("%Y-%m-%d") + " 05:30:00").replace(tzinfo=tz_HST)
        for d in dates_end_2[::-1]
    ]

    n_dates = max(len(dates_begin_1), len(dates_begin_2))

    logger.debug(
        f"Observation period is divided into two: {nights_begin_1} and {nights_end_1}"
    )
    logger.debug(
        f"Observation period is divided into two: {nights_begin_2} and {nights_end_2}"
    )

    # # one can set start/stop times with sunset/sunrise
    # datetime_sunset = [observer.sunset(d) for d in dates_begin]
    # datetime_sunrise = [observer.sunrise(d) for d in dates_end]

    # define targets
    targets = [
        StaticTarget(name=df["ob_code"][i], ra=df["ra"][i], dec=df["dec"][i])
        for i in range(df.index.size)
    ]

    # create ephemeris cache
    eph_cache = EphemerisCache(logger, precision_minutes=15)

    def process_single_target(target: StaticTarget, exptime: float) -> bool:
        t_obs_ok_single = 0
        for dd in range(n_dates):
            try:
                # observer.set_date(nights_begin_1[dd])
                eph_key = target
                _, t_start, t_stop = eph_cache.observable(
                    eph_key,
                    target,
                    observer,
                    nights_begin_1[dd],
                    nights_end_1[dd],
                    # datetime_sunset[dd],
                    # datetime_sunrise[dd],
                    min_el,  # [deg]
                    max_el,  # [deg]
                    exptime,  # [s] TODO: This has to be a total time including overheads
                    # airmass=None,
                    # moon_sep=None,
                )
                try:
                    if t_stop > t_start:
                        t_obs_ok_single += (t_stop - t_start).seconds
                except TypeError:
                    continue
            except IndexError:
                pass

            try:
                # observer.set_date(nights_begin_2[dd])
                eph_key = target
                _, t_start, t_stop = eph_cache.observable(
                    eph_key,
                    target,
                    observer,
                    nights_begin_2[dd],
                    nights_end_2[dd],
                    # datetime_sunset[dd],
                    # datetime_sunrise[dd],
                    min_el,  # [deg]
                    max_el,  # [deg]
                    exptime,  # [s] TODO: This has to be a total time including overheads
                    # airmass=None,
                    # moon_sep=None,
                )
                try:
                    if t_stop > t_start:
                        t_obs_ok_single += (t_stop - t_start).seconds
                except TypeError:
                    continue
            except IndexError:
                pass

            # Once t_obs_ok_single exceeds the required exptime, you can exit the function
            if t_obs_ok_single >= exptime:
                return True
        # If it not ever returned before, it means that the object cannot be observed in the input period
        return False

    # make the object loop vectorized
    vec_func = np.vectorize(process_single_target, otypes=["bool"])
    # TODO: Exposure time should be the total time required to make an exposure (i.e., incl. overheads)
    is_observable = vec_func(targets, df["exptime"])

    # clear the ephemeris cache
    logger.debug("Clearing the ephemeris cache")
    eph_cache.clear_all()

    return is_observable


def visibility_checker_healpix(
    df: pd.DataFrame,
    date_begin: datetime | None = None,
    date_end: datetime | None = None,
    single_exptime: float = 900.0,
    min_el: float = 30.0,
    max_el: float = 85.0,
    nside: int = 32,
    precision_minutes: int = 15,
) -> np.ndarray:
    """
    HEALPix-based visibility checker optimized for clustered targets.

    Groups targets by HEALPix pixels and uses the maximum exptime in each pixel
    for visibility calculations, significantly reducing computation time for
    spatially clustered target lists.

    Parameters
    ----------
    df : pd.DataFrame
        Target dataframe with 'ra', 'dec', 'exptime', 'ob_code' columns
    date_begin : datetime, optional
        Observation period start date
    date_end : datetime, optional
        Observation period end date
    single_exptime : float, default 900.0
        Single exposure time in seconds (used for visibility checks)
    min_el : float, default 30.0
        Minimum elevation constraint [degrees]
    max_el : float, default 85.0
        Maximum elevation constraint [degrees]
    nside : int, default 32
        HEALPix nside parameter (higher = smaller pixels)
        nside=32 gives ~110 arcmin pixel size, good for PFS field clustering
    precision_minutes : int, default 15
        Ephemeris cache time resolution in minutes
        Larger values = coarser time grid = faster computation but less precision

    Returns
    -------
    np.ndarray
        Boolean array indicating visibility for each target
    """
    if df.index.size == 0:
        return np.array([], dtype=bool)

    # Set timezone to HST
    tz_HST = tz.gettz("US/Hawaii")

    # Set next semester if no range is defined
    tmp_begin, tmp_end = get_semester_daterange(datetime.now(tz=tz_HST), next=True)

    if date_begin is None:
        date_begin = tmp_begin
    if date_end is None:
        date_end = tmp_end

    logger.info(
        f"HEALPix visibility check: Observation period {date_begin:%Y-%m-%d} to {date_end:%Y-%m-%d}"
    )

    # Create HEALPix object
    hp = HEALPix(nside=nside, order="ring")
    pixel_res_arcmin = hp.pixel_resolution.to(u.arcmin).value
    logger.info(
        f"Using HEALPix nside={nside} (~{pixel_res_arcmin:.1f} arcmin pixel size)"
    )

    # Convert target coordinates to SkyCoord
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)

    # Get HEALPix pixel indices for all targets
    pixel_indices = hp.lonlat_to_healpix(coords.ra, coords.dec)

    # Group targets by pixel and find maximum exptime in each pixel
    df_with_pixels = df.copy()
    df_with_pixels["healpix_pixel"] = pixel_indices

    pixel_groups = df_with_pixels.groupby("healpix_pixel")
    pixel_max_exptime = pixel_groups["exptime"].max()
    pixel_coords = pixel_groups[
        ["ra", "dec"]
    ].first()  # Use first target coords as representative

    logger.info(
        f"Grouped {len(df)} targets into {len(pixel_max_exptime)} HEALPix pixels"
    )

    # print first ten pixels for debugging
    logger.info(
        f"First 10 HEALPix pixels and coordinates used: {pixel_coords.head(10)}"
    )

    logger.info(
        f"Pixel exptime range: {pixel_max_exptime.min():.1f}s - {pixel_max_exptime.max():.1f}s"
    )

    # Prepare observation time periods (same logic as original function)
    date_middle = date_begin + (date_end - date_begin) / 2

    daterange_1 = pd.date_range(date_begin, date_middle, tz=tz_HST)
    daterange_2 = pd.date_range(
        date_middle + timedelta(days=1), date_end + timedelta(days=1), tz=tz_HST
    )

    dates_begin_1, dates_end_1 = daterange_1[:-1], daterange_1[1:]
    dates_begin_2, dates_end_2 = daterange_2[:-1], daterange_2[1:]

    nights_begin_1 = [
        parser.parse(d.strftime("%Y-%m-%d") + " 18:30:00").replace(tzinfo=tz_HST)
        for d in dates_begin_1
    ]
    nights_end_1 = [
        parser.parse(d.strftime("%Y-%m-%d") + " 05:30:00").replace(tzinfo=tz_HST)
        for d in dates_end_1
    ]

    # Reverse order for second half
    nights_begin_2 = [
        parser.parse(d.strftime("%Y-%m-%d") + " 18:30:00").replace(tzinfo=tz_HST)
        for d in dates_begin_2[::-1]
    ]
    nights_end_2 = [
        parser.parse(d.strftime("%Y-%m-%d") + " 05:30:00").replace(tzinfo=tz_HST)
        for d in dates_end_2[::-1]
    ]

    n_dates = max(len(dates_begin_1), len(dates_begin_2))

    # Create ephemeris cache
    eph_cache = EphemerisCache(logger, precision_minutes=precision_minutes)

    # Check visibility for each unique pixel
    pixel_visibility = {}

    for pixel_id, max_exptime in pixel_max_exptime.items():
        # Use representative coordinates for this pixel
        ra_rep = pixel_coords.loc[pixel_id, "ra"]
        dec_rep = pixel_coords.loc[pixel_id, "dec"]

        logger.debug(f"Checking visibility for pixel {pixel_id}: ({ra_rep}, {dec_rep})")

        # Create target for this pixel
        target = StaticTarget(name=f"pixel_{pixel_id}", ra=ra_rep, dec=dec_rep)

        # Check visibility with maximum exptime in this pixel
        t_obs_ok_total = 0

        for dd in range(n_dates):
            # Check first half of observation period
            try:
                eph_key = target
                _, t_start, t_stop = eph_cache.observable(
                    eph_key,
                    target,
                    observer,
                    nights_begin_1[dd],
                    nights_end_1[dd],
                    min_el,  # [deg]
                    max_el,  # [deg]
                    single_exptime,  # [s]
                )
                if t_stop is not None and t_start is not None and t_stop > t_start:
                    t_obs_ok_total += (t_stop - t_start).seconds
            except (IndexError, TypeError):
                pass

            # Check second half of observation period
            try:
                eph_key = target
                _, t_start, t_stop = eph_cache.observable(
                    eph_key,
                    target,
                    observer,
                    nights_begin_2[dd],
                    nights_end_2[dd],
                    min_el,  # [deg]
                    max_el,  # [deg]
                    single_exptime,
                )
                if t_stop is not None and t_start is not None and t_stop > t_start:
                    t_obs_ok_total += (t_stop - t_start).seconds
            except (IndexError, TypeError):
                pass

            logger.debug(
                f"Pixel {pixel_id} observable from {t_start} to {t_stop}, total {t_obs_ok_total}s"
            )
            # Early exit if we have enough observing time
            if t_obs_ok_total >= max_exptime:
                break

        # Store visibility result for this pixel
        pixel_visibility[pixel_id] = t_obs_ok_total >= max_exptime

    logger.info(
        f"Pixel visibility results: {sum(pixel_visibility.values())}/{len(pixel_visibility)} pixels observable"
    )

    # Apply pixel visibility to individual targets based on their exptime
    is_observable = np.zeros(len(df), dtype=bool)

    for i, (pixel_id, exptime) in enumerate(zip(pixel_indices, df["exptime"])):
        if pixel_visibility[pixel_id]:
            # If pixel is observable with max_exptime, check if this target's exptime is achievable
            # Since we used max_exptime for the pixel check, any target with smaller exptime
            # in the same pixel should also be observable
            is_observable[i] = True
        else:
            # If pixel failed with max_exptime, this target with smaller/equal exptime also fails
            is_observable[i] = False

    # Clear the ephemeris cache
    logger.debug("Clearing the ephemeris cache")
    eph_cache.clear_all()

    logger.info(
        f"Final visibility: {is_observable.sum()}/{len(is_observable)} targets observable"
    )

    return is_observable


def check_keys(
    df,
    required_keys=required_keys,
    optional_keys=optional_keys,
    logger=logger,
):
    required_status = []
    optional_status = []

    required_desc_success = []
    required_desc_error = []
    optional_desc_success = []
    optional_desc_warning = []

    for k in required_keys:
        if k in df.columns:
            desc = f"Required column `{k}` is found."
            required_status.append(True)
            required_desc_success.append(desc)
            logger.info(desc)
        else:
            desc = f"Required column `{k}` is missing."
            required_status.append(False)
            required_desc_error.append(desc)
            logger.error(desc)

    for k in optional_keys:
        if k in df.columns:
            desc = f"Optional column `{k}` is found."
            optional_status.append(True)
            optional_desc_success.append(desc)
            logger.info(desc)
        else:
            desc = f"Optional column `{k}` is missing. The default value, `{optional_keys_default[k]}`, will be used."
            optional_status.append(False)
            optional_desc_warning.append(desc)
            logger.warning(desc)

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
        try:
            return bool(re.match(pattern, element))
        except TypeError:
            return False

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
            if np.all(is_match):
                logger.info(f"[{k}] validation for string values in {k} is successful")
            else:
                logger.error(f"[{k}] validation for string values in {k} is failed")

    for k in optional_keys:
        if (k in df.columns) and (dtype[k] is str):
            is_match = vectorized_check(df[k].to_numpy())
            # True for good value; False for violation
            dict_str[f"status_{k}"] = np.all(is_match)
            dict_str[f"success_{k}"] = is_match
            success_optional_keys = np.logical_and(success_optional_keys, is_match)
            is_optional_success = is_optional_success and np.all(is_match)
            if np.all(is_match):
                logger.info(f"[{k}] validation for string values is {k} is successful")
            else:
                logger.warning(f"[{k}] validation for string values in {k} if failed")

    dict_str["status"] = is_success
    dict_str["status_optional"] = is_optional_success
    dict_str["success_required"] = success_required_keys
    dict_str["success_optional"] = success_optional_keys

    return dict_str


def check_values(df, logger=logger):
    # Required keys
    is_ra = np.logical_and(df["ra"] >= 0.0, df["ra"] <= 360.0)
    is_dec = np.logical_and(df["dec"] >= -90.0, df["dec"] <= 90.0)

    is_priority = np.logical_and(df["priority"] >= 0.0, df["priority"] <= 9.0)
    is_exptime = df["exptime"] > 0.0
    is_resolution = np.logical_or(df["resolution"] == "L", df["resolution"] == "M")

    is_refarm = np.full(df.index.size, False, dtype=bool)
    for arm in arm_values:
        is_refarm = np.logical_or(is_refarm, df["reference_arm"] == arm)

    # refarm shouldn't be 'medium' for the low resolution mode
    is_wrong_refarm_lr = np.logical_and(
        df["resolution"] == "L", df["reference_arm"] == "m"
    )
    # refarm shouldn't be 'red' for the low resolution mode
    is_wrong_refarm_mr = np.logical_and(
        df["resolution"] == "M", df["reference_arm"] == "r"
    )
    is_refarm = np.logical_and(is_refarm, np.all(~is_wrong_refarm_lr))
    is_refarm = np.logical_and(is_refarm, np.all(~is_wrong_refarm_mr))

    dict_values = {}
    is_success = True

    success_all = np.ones(df.index.size, dtype=bool)  # True if success

    for k, v in zip(
        ["ra", "dec", "priority", "exptime", "resolution", "reference_arm"],
        [is_ra, is_dec, is_priority, is_exptime, is_resolution, is_refarm],
    ):
        dict_values[f"status_{k}"] = np.all(v)
        dict_values[f"success_{k}"] = v
        is_success = is_success and np.all(v)
        success_all = np.logical_and(success_all, v)

        if np.all(v):
            logger.info(f"[{k}] validation for values in {k} successful")
        else:
            logger.error(f"[{k}] validation for values in {k} failed")

    dict_values["status"] = is_success
    dict_values["success"] = success_all

    # shall we check values for optional fields?

    return dict_values


def check_fluxcolumns(df, filter_category=filter_category, logger=logger):
    # initialize filter/flux columns
    logger.info("Initialize flux columns")
    for band in filter_category.keys():
        df[f"filter_{band}"] = None
        df[f"flux_{band}"] = np.nan
        df[f"flux_error_{band}"] = np.nan

    def assign_filter_category(k):
        for band in filter_category.keys():
            if k in filter_category[band]:
                return band
        return None

    t_start = time.time()

    def detect_fluxcolumns(s):
        filters_found_one = []
        is_found_filter = False
        for c in s.keys():
            b = assign_filter_category(c)
            if b is not None:
                if np.isfinite(s[c]):
                    if s[f"filter_{b}"] is not None:
                        logger.warning(
                            f"filter_{b} has already been filled. {c} filter for {s['ob_code']} is skipped."
                        )
                        continue

                    flux = s[c]
                    logger.debug(
                        f"{b} band filter column ({c}) found for OB {s['ob_code']} as {flux}"
                    )
                    is_found_filter = True
                    filters_found_one.append(c)
                    s[f"filter_{b}"] = c
                    s[f"flux_{b}"] = flux

                    try:
                        if np.isfinite(s[f"{c}_error"]):
                            flux_error = s[f"{c}_error"]
                            s[f"flux_error_{b}"] = flux_error
                            logger.debug(
                                f"{b} band flux error ({c}_error) found as {flux_error}"
                            )
                    except KeyError:
                        pass
                    except TypeError:
                        pass

        return s, is_found_filter, filters_found_one

    vfunc_fluxcolumns = np.vectorize(
        detect_fluxcolumns, otypes=[dict, bool, np.ndarray]
    )
    input_list_of_dicts = df.to_dict(orient="records")
    output_list_of_dicts, is_found, filters_found = vfunc_fluxcolumns(
        input_list_of_dicts
    )
    dfout = pd.DataFrame.from_records(output_list_of_dicts)
    t_stop = time.time()

    filters_found_flatten = [item for sublist in filters_found for item in sublist]
    filters_found_unique = np.unique(filters_found_flatten)

    logger.info(f"Flux column detection finished in {t_stop - t_start:.2f} [s]")

    dict_flux = {}
    dict_flux["success"] = is_found
    dict_flux["filters"] = filters_found_unique

    logger.info(f"Unique filters {filters_found_unique}")

    if not np.all(is_found):
        dict_flux["status"] = False
        logger.error(
            f"Flux columns are missing for objects: {dfout.loc[~is_found,'ob_code'].to_numpy()}"
        )
    else:
        logger.info("Flux columns are detected for all objects")
        dict_flux["status"] = True

    # cleaning
    logger.info("dropping columns with NA values for all rows.")
    for k, v in filter_category.items():
        if dfout.loc[:, f"filter_{k}"].isna().all():
            dfout.drop(columns=[f"filter_{k}"], inplace=True)
            dfout.drop(columns=[f"flux_{k}"], inplace=True)
            dfout.drop(columns=[f"flux_error_{k}"], inplace=True)
        elif dfout.loc[:, f"flux_{k}"].isna().all():
            dfout.drop(columns=[f"flux_{k}"], inplace=True)
        elif dfout.loc[:, f"flux_error_{k}"].isna().all():
            dfout.drop(columns=[f"flux_error_{k}"], inplace=True)

    logger.info(f"{dfout}")

    return dict_flux, dfout


def check_visibility(
    df,
    date_begin=None,
    date_end=None,
    single_exptime=900,
    vectorized=False,
    healpix=True,
    nside=32,
    logger=logger,
):
    dict_visibility = {}

    if healpix:
        logger.info("Using HEALPix-optimized visibility checker")
        is_visible = visibility_checker_healpix(
            df,
            date_begin=date_begin,
            date_end=date_end,
            single_exptime=single_exptime,
            nside=nside,
        )
    elif vectorized:
        is_visible = visibility_checker_vec(
            df, date_begin=date_begin, date_end=date_end
        )
    else:
        is_visible = visibility_checker(df, date_begin=date_begin, date_end=date_end)
        # print(is_visible)

    if np.all(is_visible):
        logger.info("All objects are visible in the input period")
        dict_visibility["status"] = True
    elif np.any(is_visible):
        logger.warning(
            f"Objects are not visible in the input period: {df.loc[~is_visible,'ob_code'].to_list()}"
        )
        dict_visibility["status"] = True
    else:
        # None of targets are visible in the input observation period
        logger.error("None of objects is visible in the input period")
        dict_visibility["status"] = False

    dict_visibility["success"] = is_visible

    return dict_visibility


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

    # find unique elements for a pair of ('obj_id', 'resolution')
    # unique_elements, unique_counts = np.unique(
    #     df.loc[:, ["obj_id", "resolution"]].to_numpy(), return_counts=True
    # )
    is_duplicated = df.duplicated(subset=["obj_id", "resolution"], keep="first")

    # If the number of duplicated elements is zero, 'success' status is returned.
    if np.sum(is_duplicated) == 0:
        unique_status = unique_status and True
        description += " All ('ob_code', 'resolution') pairs are unique."
        logger.info("All ('ob_code', 'resolution') are unique.")
    else:
        for i in np.arange(df.index.size)[is_duplicated]:
            flag_duplicate[
                np.logical_and(
                    df["obj_id"] == df["obj_id"][i],
                    df["resolution"] == df["resolution"][i],
                )
            ] = True
        unique_status = False
        description += " Duplicate ('obj_id', 'resolution') pair found. ('obj_id', 'resolution') must be unique."
        logger.error("Duplicates in ('obj_id', 'resolution') detected!")
        logger.error(
            f"""Duplicates by flag:\n{df.loc[flag_duplicate,['ob_code', 'obj_id', 'resolution']]}"""
        )

    return dict(status=unique_status, flags=flag_duplicate, description=description)


def validate_input(
    df,
    date_begin=None,
    date_end=None,
    single_exptime=900,
    healpix=True,
    nside=32,
    logger=logger,
):
    logger.info("Validation of the input list starts")
    t_validate_start = time.time()

    def msg_t_stop():
        t_validate_stop = time.time()
        logger.info(
            f"Validation of the input list finished in {t_validate_stop-t_validate_start:.1f} seconds"
        )

    validation_status = {}

    # Validation status
    # - None: not reached to the step
    # - True: success
    # - False: fail

    validation_status["status"] = False

    validation_status["str"] = {"status": None}
    validation_status["values"] = {"status": None}
    validation_status["flux"] = {"status": None}
    validation_status["visibility"] = {"status": None}
    validation_status["unique"] = {"status": None}

    # check mandatory columns
    logger.info("[Column names] Checking column names")
    dict_required_keys, dict_optional_keys = check_keys(df)
    logger.info(
        f"[Columns] required_keys status: {dict_required_keys['status']} (Success if True)"
    )
    logger.info(
        f"[Columns] optional_keys status: {dict_optional_keys['status']} (Success if True)"
    )
    validation_status["required_keys"] = dict_required_keys
    validation_status["optional_keys"] = dict_optional_keys

    if not dict_required_keys["status"]:
        msg_t_stop()
        return validation_status, df

    # check string values
    logger.info("[Strings] Checking string values")
    dict_str = check_str(df)
    logger.info(f"[Strings] status: {dict_str['status']} (Success if True)")
    validation_status["str"] = dict_str
    if not dict_str["status"]:
        msg_t_stop()
        return validation_status, df

    # check value against allowed ranges
    logger.info("[Values] Checking whether values are in allowed ranges")
    dict_values = check_values(df)
    logger.info(f"[Values] status: {dict_values['status']} (Success if True)")
    validation_status["values"] = dict_values
    if not dict_values["status"]:
        msg_t_stop()
        return validation_status, df

    # check columns for flux
    logger.info("[Fluxes] Checking flux information")
    dict_flux, df = check_fluxcolumns(df)
    logger.info(f"[Fluxes] status: {dict_flux['status']} (Success if True)")
    validation_status["flux"] = dict_flux

    # check columns for visibility
    logger.info("[Visibility] Checking target visibility")
    dict_visibility = check_visibility(
        df,
        date_begin=date_begin,
        date_end=date_end,
        single_exptime=single_exptime,
        vectorized=False,
        healpix=healpix,
        nside=nside,
    )
    logger.info(f"[Visibility] status: {dict_visibility['status']} (Success if True)")
    validation_status["visibility"] = dict_visibility

    # check unique constraint for `ob_code`
    logger.info("[Uniqueness] Checking whether all ob_code are unique")
    dict_unique = check_unique(df)
    logger.info(f"[Uniqueness] status: {dict_unique['status']} (Success if True)")
    validation_status["unique"] = dict_unique

    if (
        validation_status["required_keys"]["status"]
        and validation_status["str"]["status"]
        and validation_status["values"]["status"]
        and validation_status["flux"]["status"]
        and validation_status["visibility"]["status"]
        and validation_status["unique"]["status"]
    ):
        logger.info("[Summary] succeeded to meet all validation criteria")
        validation_status["status"] = True
    else:
        logger.info("[Summary] failed to meet all validation criteria")

    msg_t_stop()

    # remove unregistered columns from the dataframe
    logger.info("Dropping columns not in the required, optional, and filter keys")
    for k in df.columns:
        if k not in required_keys + optional_keys + filter_keys:
            logger.info(f'"{k}" is dropped')
            df.drop(columns=[k], inplace=True)

    return validation_status, df
