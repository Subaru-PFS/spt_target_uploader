#!/usr/bin/env python3

import numpy as np

__all__ = [
    "required_keys",
    "optional_keys",
    "optional_keys_default",
    "filter_keys",
    "target_datatype",
    "ppc_datatype",
    "filter_category",
]


required_keys = [
    "obj_id",
    "ob_code",
    "ra",
    "dec",
    "priority",
    "exptime",
    "resolution",
    "reference_arm",
]


optional_keys = ["pmra", "pmdec", "parallax", "tract", "patch"]
optional_keys_default = {
    "pmra": 0,
    "pmdec": 0,
    "parallax": 1e-7,
    "tract": None,
    "patch": None,
}


target_datatype = {
    # required keys
    "ob_code": str,
    "obj_id": np.int64,
    "ra": float,  # deg
    "dec": float,  # deg
    "exptime": float,  # s
    "priority": float or int,
    "resolution": str,
    # optional keys
    "pmra": float,  # mas/yr
    "pmdec": float,  # mas/yr
    "parallax": float,  # mas
    "tract": int,
    "patch": int,
    "reference_arm": str,
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

ppc_datatype = {
    "ppc_code": str,
    "ppc_ra": float,
    "ppc_dec": float,
    "ppc_pa": float,
    "ppc_resolution": str,
    "ppc_priority": float,
}


# The mapping must stay one-to-one: no filter may appear under two bands.
# check_fluxcolumns() works one band at a time and would populate *every* band
# that lists a filter, where the row loop it replaced returned the first
# matching band and populated only that one.
#
# The Gaia entries are the ones most likely to look like a typo worth
# "correcting" by adding a second entry.  They are deliberate, and follow
# effective wavelength rather than the band letter: BP (~330-680 nm) sits with
# g, G is broad (~330-1050 nm) but centred near r, and RP (~630-1050 nm) sits
# with i.  Adding g_gaia to "g" would break the invariant below.
filter_category = {
    "g": ["g_hsc", "g_ps1", "g_sdss", "bp_gaia"],
    "r": ["r_old_hsc", "r2_hsc", "r_ps1", "r_sdss", "g_gaia"],
    "i": ["i_old_hsc", "i2_hsc", "i_ps1", "i_sdss", "rp_gaia"],
    "z": ["z_hsc", "z_ps1", "z_sdss"],
    "y": ["y_hsc", "y_ps1"],
    "j": [],
}


def _check_bands_disjoint(categories: dict) -> None:
    """Raise if any filter is listed under more than one band.

    Enforced at import rather than in a test: the test suite does not run in
    CI, so a bad edit to the literal above would otherwise reach the app and
    show up as a filter quietly counted twice.  ``raise`` rather than
    ``assert`` so ``python -O`` cannot strip it.
    """
    band_of_filter = {}
    for band, band_filters in categories.items():
        for name in band_filters:
            if name in band_of_filter:
                raise ValueError(
                    f"filter_category must map each filter to one band, but "
                    f"{name!r} is listed under both {band_of_filter[name]!r} "
                    f"and {band!r}"
                )
            band_of_filter[name] = band


_check_bands_disjoint(filter_category)


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


filter_keys = [
    # TODO: filters must be in the filter_name table in targetDB
    "filter_g",
    "filter_r",
    "filter_i",
    "filter_z",
    "filter_y",
    "filter_j",
    # TODO: fluxes can be fiber, psf, total, etc.
    # Let's assume it is total (still ambiguous, though)
    "flux_g",
    "flux_r",
    "flux_i",
    "flux_z",
    "flux_y",
    "flux_j",
    # errors are optional
    "flux_error_g",
    "flux_error_r",
    "flux_error_i",
    "flux_error_z",
    "flux_error_y",
    "flux_error_j",
]

arm_values = ["b", "r", "n", "m"]
