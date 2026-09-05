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
from qplan.util.site import site_subaru as observer
from spot.util.eph_cache import EphemerisCache

from . import (
    arm_values,
    filter_category,
    filter_keys,
    optional_keys,
    optional_keys_default,
    required_keys,
    target_datatype,
)
from .internal_duplication import dupcheck_internal

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

    This is the visibility checker used by validate_input().
    Groups targets by HEALPix pixels and uses the maximum exptime in each pixel
    for visibility calculations, significantly reducing computation time for
    spatially clustered target lists.

    Performance: reduces N per-target ephemeris calculations to one per occupied HEALPix pixel.

    Algorithm Details
    -----------------
    For each HEALPix pixel, the function:
    1. Uses the first target's coordinates as representative (conservative approximation)
    2. Calculates total observable time across the observation period
    3. Compares each target's exptime against its pixel's total observable time

    This is conservative because all targets within a pixel (~110 arcmin) have similar
    but not identical observability. Using a single representative position ensures
    we don't overestimate observability.

    Parameters
    ----------
    df : pd.DataFrame
        Target dataframe with 'ra', 'dec', 'exptime', 'ob_code' columns
    date_begin : datetime, optional
        Observation period start date
    date_end : datetime, optional
        Observation period end date
    single_exptime : float, default 900.0
        Minimum contiguous observation window in seconds. This parameter filters
        out short observability windows during ephemeris queries. The value
        (900s = 15min) aligns with ephemeris cache precision and should be
        significantly smaller than typical target exptime values.
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

    t_start_total = time.time()

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
    # Store total observable time (in seconds) for each pixel
    pixel_observable_time = {}  # dict[int, float]

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
                    window_time = (t_stop - t_start).seconds
                    t_obs_ok_total += window_time
                    logger.debug(
                        f"Pixel {pixel_id} day {dd} (1st half): observable window {window_time}s "
                        f"({t_start} to {t_stop}), cumulative {t_obs_ok_total}s"
                    )
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
                    window_time = (t_stop - t_start).seconds
                    t_obs_ok_total += window_time
                    logger.debug(
                        f"Pixel {pixel_id} day {dd} (2nd half): observable window {window_time}s "
                        f"({t_start} to {t_stop}), cumulative {t_obs_ok_total}s"
                    )
            except (IndexError, TypeError):
                pass

            # Early exit if we have enough observing time
            if t_obs_ok_total >= max_exptime:
                logger.debug(
                    f"Pixel {pixel_id}: early exit after {dd+1}/{n_dates} days "
                    f"({t_obs_ok_total}s >= {max_exptime}s required)"
                )
                break

        # Store total observable time for this pixel (not boolean!)
        pixel_observable_time[pixel_id] = t_obs_ok_total

    # Count how many pixels meet their max_exptime requirement
    pixels_meet_max = sum(
        1
        for pid, obs_time in pixel_observable_time.items()
        if obs_time >= pixel_max_exptime[pid]
    )
    logger.info(
        f"Pixel visibility results: {pixels_meet_max}/{len(pixel_observable_time)} "
        f"pixels meet max_exptime requirement"
    )
    if pixel_observable_time:
        logger.debug(
            f"Observable time range: {min(pixel_observable_time.values()):.1f}s - "
            f"{max(pixel_observable_time.values()):.1f}s"
        )

    # Apply pixel visibility to individual targets based on their exptime
    is_observable = np.zeros(len(df), dtype=bool)

    for i, (pixel_id, exptime) in enumerate(zip(pixel_indices, df["exptime"])):
        # Compare target's exptime against pixel's total observable time
        # This is the CORRECT logic: target is observable if pixel has enough total time
        is_observable[i] = pixel_observable_time[pixel_id] >= exptime

    # Clear the ephemeris cache
    logger.debug("Clearing the ephemeris cache")
    eph_cache.clear_all()

    t_elapsed = time.time() - t_start_total
    logger.info(
        f"HEALPix visibility check completed in {t_elapsed:.2f}s "
        f"({t_elapsed/len(df)*1000:.1f}ms per target)"
    )
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


def _warn_unusable_entries(column, series, finite, mask, logger):
    """Report cells that hold something, but not a finite number.

    ``pd.to_numeric(errors="coerce")`` turns unparseable text into NaN, which
    is then indistinguishable from an empty cell.  Without this warning the
    object either falls through to the band's next column or is reported as
    "flux missing", and in neither case does the user learn which value was
    discarded.  ``mask`` restricts the report to a subset of rows (used for
    error columns, where only the objects that actually adopted the column
    matter).
    """
    unusable = ~finite & series.notna().to_numpy()
    if mask is not None:
        unusable &= mask
    n_bad = int(np.count_nonzero(unusable))
    if n_bad == 0:
        return
    examples = series.to_numpy()[unusable][:3].tolist()
    logger.warning(
        f"{column}: {n_bad} entry(ies) hold a value that is not a finite "
        f"number and are read as missing, e.g. {examples}"
    )


def check_fluxcolumns(df, filter_category=filter_category, logger=logger):
    logger.info("Detecting flux columns")
    t_start = time.time()

    dfout = df.copy(deep=True)
    n_rows = len(dfout)

    # One "did this object get any flux at all" flag per row, OR-ed across bands.
    is_found = np.zeros(n_rows, dtype=bool)
    filters_found = []

    for band, band_filters in filter_category.items():
        # DataFrame column order sets the priority, not the order the band's
        # filters happen to be listed in filter_category: the old row loop
        # iterated over the record dict's keys, which follow the frame.
        candidates = [c for c in df.columns if c in band_filters]

        filter_of_band = np.full(n_rows, None, dtype=object)
        flux_of_band = np.full(n_rows, np.nan)
        flux_error_of_band = np.full(n_rows, np.nan)
        assigned = np.zeros(n_rows, dtype=bool)

        for column in candidates:
            # errors="coerce" rather than a bare np.isfinite: a stray
            # non-numeric entry used to raise an uncaught TypeError and kill
            # validation with an opaque traceback. Reading it as "no flux"
            # lets the object be reported through dict_flux["status"] below,
            # which already names the objects that came up empty.
            flux = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(flux)
            _warn_unusable_entries(column, df[column], finite, None, logger)

            n_skipped = int(np.count_nonzero(finite & assigned))
            if n_skipped > 0:
                # Aggregated on purpose. The old code warned once per row,
                # which meant tens of thousands of identical lines on a large
                # list -- and cost real time writing them.
                logger.warning(
                    f"filter_{band} has already been filled by an earlier column; "
                    f"{column} is skipped for {n_skipped} object(s)."
                )

            take = finite & ~assigned
            if not take.any():
                continue

            filter_of_band[take] = column
            flux_of_band[take] = flux[take]

            error_column = f"{column}_error"
            if error_column in df.columns:
                flux_error = pd.to_numeric(df[error_column], errors="coerce").to_numpy(
                    dtype=float
                )
                error_finite = np.isfinite(flux_error)
                _warn_unusable_entries(
                    error_column, df[error_column], error_finite, take, logger
                )
                has_error = take & error_finite
                flux_error_of_band[has_error] = flux_error[has_error]

            assigned |= take
            filters_found.append(column)

        dfout[f"filter_{band}"] = filter_of_band
        dfout[f"flux_{band}"] = flux_of_band
        dfout[f"flux_error_{band}"] = flux_error_of_band
        is_found |= assigned

    filters_found_unique = np.unique(filters_found)
    t_stop = time.time()

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

    return dict_flux, dfout


def check_fluxvalues(
    df: pd.DataFrame,
    filter_category: dict = filter_category,
    min_flux: float = 10.0,
    max_flux: float = 30.0,
    thresh_frac_suspicious: float = 0.9,
    logger=logger,
):
    bands = filter_category.keys()

    dict_flux_values = {}

    for band in bands:
        if f"flux_{band}" in df.columns:
            is_flux_finite = np.isfinite(df[f"flux_{band}"])
            is_flux_suspicious = (
                (df[f"flux_{band}"] >= min_flux)
                & (df[f"flux_{band}"] <= max_flux)
                & is_flux_finite
            )

            frac_suspicious = (
                np.sum(is_flux_suspicious) / np.sum(is_flux_finite)
                if np.sum(is_flux_finite) > 0
                else np.nan
            )

            num_total = np.sum(is_flux_finite)
            num_suspicious = np.sum(is_flux_suspicious)

            dict_flux_values[f"frac_suspicious_flux_{band}"] = frac_suspicious
            dict_flux_values[f"status_flux_{band}"] = (
                True if frac_suspicious < thresh_frac_suspicious else False
            )
            dict_flux_values[f"num_total_flux_{band}"] = num_total
            dict_flux_values[f"num_suspicious_flux_{band}"] = num_suspicious

            # Log the results for each band
            if num_total > 0:
                if dict_flux_values[f"status_flux_{band}"]:
                    logger.info(
                        f"[Flux {band}] {frac_suspicious:.1%} suspicious values "
                        f"({num_suspicious}/{num_total}) - OK"
                    )
                else:
                    logger.warning(
                        f"[Flux {band}] {frac_suspicious:.1%} suspicious values "
                        f"({num_suspicious}/{num_total}) - WARNING: possible magnitude instead of nJy"
                    )

    status_flux_values = True
    num_total_flux = 0
    num_suspicious_flux = 0

    for band in bands:
        if f"status_flux_{band}" in dict_flux_values:
            status_flux_values = (
                status_flux_values and dict_flux_values[f"status_flux_{band}"]
            )
            num_total_flux += dict_flux_values[f"num_total_flux_{band}"]
            num_suspicious_flux += dict_flux_values[f"num_suspicious_flux_{band}"]
    dict_flux_values["status"] = status_flux_values
    dict_flux_values["thresh_frac_suspicious"] = thresh_frac_suspicious
    dict_flux_values["min_flux"] = min_flux
    dict_flux_values["max_flux"] = max_flux
    dict_flux_values["num_total_flux"] = num_total_flux
    dict_flux_values["num_suspicious_flux"] = num_suspicious_flux
    dict_flux_values["frac_suspicious_flux_all"] = (
        num_suspicious_flux / num_total_flux if num_total_flux > 0 else np.nan
    )

    # Log overall summary
    if num_total_flux > 0:
        frac_all = num_suspicious_flux / num_total_flux
        if status_flux_values:
            logger.info(
                f"[Flux overall] {frac_all:.1%} suspicious values "
                f"({num_suspicious_flux}/{num_total_flux}) - OK"
            )
        else:
            logger.warning(
                f"[Flux overall] {frac_all:.1%} suspicious values "
                f"({num_suspicious_flux}/{num_total_flux}) - WARNING: "
                f"threshold exceeded ({thresh_frac_suspicious:.0%})"
            )
    else:
        logger.warning("[Flux overall] No flux values found to check")

    return dict_flux_values


def check_fluxrange(
    df: pd.DataFrame,
    filter_category: dict = filter_category,
    min_mag: float | None = None,
    max_mag: float | None = None,
    logger=logger,
):
    """
    Check if flux values are within specified AB magnitude range.

    Parameters
    ----------
    df : pd.DataFrame
        Target dataframe with flux_{band} columns (values in nJy)
    filter_category : dict
        Dictionary of filter bands (default from __init__.py)
    min_mag : float or None
        Minimum AB magnitude (brightest allowed limit).
        None means no bright limit.
    max_mag : float or None
        Maximum AB magnitude (faintest allowed limit).
        None means no faint limit.
    logger : loguru.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary with validation results:
        - status: bool (True if all in range, False if any out of range)
        - min_mag, max_mag: input magnitude limits
        - min_flux_nJy, max_flux_nJy: converted flux limits
        - success_flux_{band}: per-row boolean arrays
        - status_flux_{band}: per-band overall status
        - num_total_flux_{band}, num_out_of_range_flux_{band}: counts
        - success: overall per-row boolean array
        - num_total_flux, num_out_of_range_flux: overall counts

    Notes
    -----
    AB magnitude to nJy conversion uses astropy.units for accuracy.

    A brighter magnitude (lower number) corresponds to higher flux in nJy.
    Therefore:
    - min_mag (bright limit) -> max_flux_nJy (upper flux bound)
    - max_mag (faint limit) -> min_flux_nJy (lower flux bound)
    """
    bands = filter_category.keys()

    # Validate magnitude range consistency
    if min_mag is not None and max_mag is not None:
        if min_mag > max_mag:
            raise ValueError(
                f"Invalid magnitude range: min_mag ({min_mag}) > max_mag ({max_mag}). "
                f"min_mag should be brighter (smaller value) than max_mag."
            )

    # Convert AB magnitude limits to nJy using astropy.units
    # Note: brighter mag (smaller number) = higher flux
    min_flux_nJy = None
    max_flux_nJy = None

    if max_mag is not None:
        # Faint limit -> lower flux bound
        min_flux_nJy = (max_mag * u.ABmag).to_value(u.nJy)

    if min_mag is not None:
        # Bright limit -> upper flux bound
        max_flux_nJy = (min_mag * u.ABmag).to_value(u.nJy)

    dict_fluxval = {
        "min_mag": min_mag,
        "max_mag": max_mag,
        "min_flux_nJy": min_flux_nJy,
        "max_flux_nJy": max_flux_nJy,
    }

    # Track overall success per row
    success_all = np.ones(df.index.size, dtype=bool)
    num_total_flux = 0
    num_out_of_range_flux = 0

    for band in bands:
        col_name = f"flux_{band}"
        if col_name not in df.columns:
            continue

        is_flux_finite = np.isfinite(df[col_name])
        num_total = np.sum(is_flux_finite)

        # Initialize per-row success as True for finite values
        is_in_range = is_flux_finite.copy()

        # Apply lower bound (faint limit)
        if min_flux_nJy is not None:
            is_in_range = is_in_range & (
                (df[col_name] >= min_flux_nJy) | ~is_flux_finite
            )

        # Apply upper bound (bright limit)
        if max_flux_nJy is not None:
            is_in_range = is_in_range & (
                (df[col_name] <= max_flux_nJy) | ~is_flux_finite
            )

        # For non-finite values, mark as True (not checked)
        is_in_range = is_in_range | ~is_flux_finite

        num_out_of_range = np.sum(is_flux_finite & ~is_in_range)
        status_band = num_out_of_range == 0

        dict_fluxval[f"success_flux_{band}"] = is_in_range
        dict_fluxval[f"status_flux_{band}"] = status_band
        dict_fluxval[f"num_total_flux_{band}"] = int(num_total)
        dict_fluxval[f"num_out_of_range_flux_{band}"] = int(num_out_of_range)

        # Update overall tracking
        success_all = success_all & is_in_range
        num_total_flux += num_total
        num_out_of_range_flux += num_out_of_range

        # Log results for each band
        if num_total > 0:
            if status_band:
                logger.info(
                    f"[Flux {band}] All {num_total} values within magnitude range - OK"
                )
            else:
                logger.warning(
                    f"[Flux {band}] {num_out_of_range}/{num_total} values out of range "
                    f"(AB mag {min_mag} to {max_mag}) - WARNING"
                )

    # Overall status
    if num_total_flux == 0:
        status_overall = None  # No flux values to check
    else:
        status_overall = num_out_of_range_flux == 0
    dict_fluxval["status"] = status_overall
    dict_fluxval["success"] = success_all
    dict_fluxval["num_total_flux"] = int(num_total_flux)
    dict_fluxval["num_out_of_range_flux"] = int(num_out_of_range_flux)

    # Log overall summary
    if num_total_flux > 0:
        if status_overall:
            logger.info(
                f"[Flux range check] All {num_total_flux} flux values within "
                f"AB magnitude range [{min_mag}, {max_mag}] - OK"
            )
        else:
            logger.warning(
                f"[Flux range check] {num_out_of_range_flux}/{num_total_flux} flux values "
                f"out of AB magnitude range [{min_mag}, {max_mag}] - WARNING"
            )
    else:
        logger.warning("[Flux range check] No flux values found to check")

    return dict_fluxval


def check_visibility(
    df,
    date_begin=None,
    date_end=None,
    single_exptime=900,
    nside=32,
    logger=logger,
):
    """
    Check target visibility during the observation period.

    Parameters
    ----------
    df : pd.DataFrame
        Target dataframe
    date_begin : datetime, optional
        Observation period start date
    date_end : datetime, optional
        Observation period end date
    single_exptime : float, default 900
        Single exposure time in seconds. Controls the time resolution of the
        observability-window checks.
    nside : int, default 32
        HEALPix nside parameter
    logger : loguru.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary with 'status' (bool) and 'success' (array) keys
    """
    dict_visibility = {}

    is_visible = visibility_checker_healpix(
        df,
        date_begin=date_begin,
        date_end=date_end,
        single_exptime=single_exptime,
        nside=nside,
    )

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


def check_internal_duplicate(
    df: pd.DataFrame, sep: u.Quantity = 1.0 * u.arcsec, logger=logger
) -> dict:
    """
    Check for internal duplicate or clustered targets within a single input table.

    This function identifies exact and near-duplicate targets (within ``sep``)
    based on sky position and returns a boolean mask of duplicated targets
    together with the nearest-neighbour separation for each.

    Parameters
    ----------
    df : pandas.DataFrame
        Input target table. ``"ob_code"`` is not required to be unique here
        -- duplicate ``"ob_code"`` values are themselves reported by
        :func:`check_unique` and must not prevent this check from running.
    sep : astropy.units.Quantity, optional
        Maximum angular separation used to define near-duplicates.
        Default is 1.0 arcsec (PFS fiber diameter).
    logger : loguru.Logger, optional
        Logger instance used for reporting.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"status"``: ``True`` if no internal duplicates or clusters are
          found, otherwise ``False``.
        - ``"flags"``: a boolean array of the same length as ``df`` where
          ``True`` marks duplicated targets and ``False`` marks isolated ones.
        - ``"nn_sep"``: a float array giving the nearest-neighbour separation
          (in arcsec) for duplicated targets, and ``NaN`` for isolated ones.
    """
    # Work on a positionally-indexed copy so the row alignment below never
    # depends on `ob_code` being unique (a list with duplicate `ob_code`
    # still needs to reach validation results -- via check_unique() --
    # instead of raising).
    df_positional = df.reset_index(drop=True)

    df_isolated, df_dups_exact, df_dups_near = dupcheck_internal(
        df_positional,
        sep=sep,
        max_cluster_diameter=1.0 * u.arcsec,  # PFS fiber diameter
        max_points_for_agglomerative=None,
    )

    # Combine all duplicates (exact + near) with nn_sep information
    df_dups_all = pd.concat([df_dups_exact, df_dups_near], ignore_index=False)

    # Initialize output arrays (use len(df) to avoid index assumptions)
    is_duplicated = np.zeros(len(df), dtype=bool)
    nn_sep_array = np.full(len(df), np.nan)

    # dupcheck_internal() preserves the row position (0..len(df)-1) of the
    # rows it returns, so align by position rather than by `ob_code` value.
    if not df_dups_all.empty:
        dup_positions = df_dups_all.index.to_numpy()
        is_duplicated[dup_positions] = True
        nn_sep_array[dup_positions] = df_dups_all["nn_sep"].to_numpy()

    if len(df) == len(df_isolated):
        logger.info("No duplicated or clustered targets found internally.")
        status = True
    else:
        logger.warning("Duplicated or clustered targets found internally.")
        status = False

    return dict(status=status, flags=is_duplicated, nn_sep=nn_sep_array)


def validate_input(
    df,
    date_begin=None,
    date_end=None,
    single_exptime=900,
    nside=32,
    min_mag=None,
    max_mag=None,
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

    validation_status["empty"] = {"status": None}
    validation_status["str"] = {"status": None}
    validation_status["values"] = {"status": None}
    validation_status["flux_columns"] = {"status": None}
    validation_status["flux_values"] = {"status": None}
    validation_status["flux_range"] = {"status": None}
    validation_status["visibility"] = {"status": None}
    validation_status["unique"] = {"status": None}
    validation_status["internal_duplication"] = {"status": None}

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

    # Every check below indexes the frame by column name, and a duplicated
    # name makes df[col] a DataFrame rather than a Series.  Each consumer then
    # fails in its own opaque way -- check_values() with "Cannot apply ufunc
    # <ufunc 'logical_and'> to mixed DataFrame and Series inputs",
    # check_fluxcolumns() with "arg must be a list, tuple, 1-d array, or
    # Series" -- and neither names the column.  In the web app that reaches
    # the catch-all handler in FileInputWidgets.validate(); in the CLI it is a
    # bare traceback.
    #
    # Unreachable from a file: pd.read_csv mangles CSV duplicates to `ra.1`
    # and astropy Tables forbid duplicate names.  Only a caller building a
    # frame in memory can get here, so name the offenders rather than quietly
    # repairing the frame.  (check_fluxcolumns() used to round-trip through
    # to_dict(orient="records"), where the last duplicate silently won; the
    # vectorized version has no such accidental tolerance.)
    duplicated_columns = sorted(set(df.columns[df.columns.duplicated()]))
    if duplicated_columns:
        raise ValueError(
            "validate_input() cannot read a frame with duplicate column "
            f"name(s): {duplicated_columns}"
        )

    if df.empty:
        desc = "The file contains no data rows."
        logger.error(desc)
        validation_status["empty"] = {"status": False, "desc_error": desc}
        msg_t_stop()
        return validation_status, df
    validation_status["empty"] = {"status": True}

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
    logger.info("[Flux columns] Checking flux information")
    dict_flux_columns, df = check_fluxcolumns(df)
    validation_status["flux_columns"] = dict_flux_columns
    logger.info(
        f"[Flux columns] status: {dict_flux_columns['status']} (Success if True)"
    )

    logger.info("[Flux values] Checking flux values for suspicious entries")
    dict_flux_values = check_fluxvalues(df)
    validation_status["flux_values"] = dict_flux_values
    logger.info(f"[Flux values] status: {dict_flux_values['status']} (Success if True)")

    # check flux value range (AB magnitude based)
    if min_mag is not None or max_mag is not None:
        logger.info("[Flux range] Checking flux values against AB magnitude range")
        dict_flux_range = check_fluxrange(df, min_mag=min_mag, max_mag=max_mag)
        validation_status["flux_range"] = dict_flux_range
        logger.info(
            f"[Flux range] status: {dict_flux_range['status']} (Success if True)"
        )
    else:
        logger.info("[Flux range] Skipping flux range check (no limits specified)")
        validation_status["flux_range"] = {
            "status": None,
            "min_mag": None,
            "max_mag": None,
            "min_flux_nJy": None,
            "max_flux_nJy": None,
            "success": np.ones(len(df), dtype=bool),
            "num_total_flux": 0,
            "num_out_of_range_flux": 0,
        }

    # check columns for visibility
    logger.info("[Visibility] Checking target visibility")
    dict_visibility = check_visibility(
        df,
        date_begin=date_begin,
        date_end=date_end,
        single_exptime=single_exptime,
        nside=nside,
    )
    logger.info(f"[Visibility] status: {dict_visibility['status']} (Success if True)")
    validation_status["visibility"] = dict_visibility

    # check unique constraint for `ob_code`
    logger.info("[Uniqueness] Checking whether all ob_code are unique")
    dict_unique = check_unique(df)
    logger.info(f"[Uniqueness] status: {dict_unique['status']} (Success if True)")
    validation_status["unique"] = dict_unique

    # check internal duplication by coordinates
    logger.info("[Internal duplication] Checking internal duplication by coordinates")
    dict_internal_dup = check_internal_duplicate(df)
    logger.info(
        f"[Internal duplication] status: {dict_internal_dup['status']} (Success if True)"
    )
    validation_status["internal_duplication"] = dict_internal_dup

    if (
        validation_status["required_keys"]["status"]
        and validation_status["str"]["status"]
        and validation_status["values"]["status"]
        and validation_status["flux_columns"]["status"]
        and validation_status["visibility"]["status"]
        and validation_status["unique"]["status"]
    ):
        logger.info("[Summary] succeeded to meet all validation criteria")
        validation_status["status"] = True
    else:
        logger.warning("[Summary] failed to meet all validation criteria")

    msg_t_stop()

    # remove unregistered columns from the dataframe
    logger.info("Dropping columns not in the required, optional, and filter keys")
    for k in df.columns:
        if k not in required_keys + optional_keys + filter_keys:
            logger.info(f'"{k}" is dropped')
            df.drop(columns=[k], inplace=True)

    return validation_status, df
