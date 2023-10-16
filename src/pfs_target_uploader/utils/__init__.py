#!/usr/bin/env python3

import numpy as np

__all__ = ["required_keys", "optional_keys", "target_datatype", "filter_category"]


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


optional_keys = ["pmra", "pmdec", "parallax", "tract", "patch"]


target_datatype = {
    # required keys
    "ob_code": str,
    "obj_id": np.int64,
    "ra": float,  # deg
    "dec": float,  # deg
    "equinox": str,
    "exptime": float,  # s
    "priority": float or int,
    "resolution": str,
    # optional keys
    "pmra": float,  # mas/yr
    "pmdec": float,  # mas/yr
    "parallax": float,  # mas
    "tract": int,
    "patch": int,
    # # filter keys
    # "filter_g": str,
    # "filter_r": str,
    # "filter_i": str,
    # "filter_z": str,
    # "filter_y": str,
    # "filter_j": str,
    # "flux_g": float,  # nJy
    # "flux_r": float,  # nJy
    # "flux_i": float,  # nJy
    # "flux_z": float,  # nJy
    # "flux_y": float,  # nJy
    # "flux_j": float,  # nJy
    # "flux_error_g": float,  # nJy
    # "flux_error_r": float,  # nJy
    # "flux_error_i": float,  # nJy
    # "flux_error_z": float,  # nJy
    # "flux_error_y": float,  # nJy
    # "flux_error_j": float,  # nJy
}


filter_category = {
    "g": ["g_hsc", "g_ps1", "g_sdss", "bp_gaia"],
    "r": ["r_old_hsc", "r2_hsc", "r_ps1", "r_sdss", "g_gaia"],
    "i": ["i_old_hsc", "i2_hsc", "i_ps1", "i_sdss", "rp_gaia"],
    "z": ["z_hsc", "z_ps1", "z_sdss"],
    "y": ["y_hsc", "y_ps1"],
    "j": [],
}


# filter_names = [
#     "g_hsc",
#     "r_old_hsc",
#     "r2_hsc",
#     "i_old_hsc",
#     "i2_hsc",
#     "z_hsc",
#     "y_hsc",
#     "g_ps1",
#     "r_ps1",
#     "i_ps1",
#     "z_ps1",
#     "y_ps1",
#     "bp_gaia",
#     "rp_gaia",
#     "g_gaia",
#     "u_sdss",
#     "g_sdss",
#     "r_sdss",
#     "i_sdss",
#     "z_sdss",
# ]


# filter_keys = [
#     # TODO: filters must be in the filter_name table in targetDB
#     "filter_g",
#     "filter_r",
#     "filter_i",
#     "filter_z",
#     "filter_y",
#     "filter_j",
#     # TODO: fluxes can be fiber, psf, total, etc.
#     # Let's assume it is total (still ambiguous, though)
#     "flux_g",
#     "flux_r",
#     "flux_i",
#     "flux_z",
#     "flux_y",
#     "flux_j",
#     # errors are optional
#     "flux_error_g",
#     "flux_error_r",
#     "flux_error_i",
#     "flux_error_z",
#     "flux_error_y",
#     "flux_error_j",
# ]
