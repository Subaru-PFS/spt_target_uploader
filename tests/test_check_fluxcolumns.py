"""Tests for the vectorized flux-column detector.

check_fluxcolumns() used to walk every row in Python via np.vectorize, which
cost 5.3 s on a 30,000-row list.  The rewrite works per band on whole columns.
The contract it must keep is subtle -- the winning filter for a band is the
first *DataFrame column* holding a finite value, which is column order, not
the order the band's filters are listed in filter_category -- so the tests
below pin it against a frozen copy of the old implementation.
"""

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from pfs_target_uploader.utils import filter_category
from pfs_target_uploader.utils.checker import check_fluxcolumns


def reference_check_fluxcolumns(df, filter_category=filter_category, logger=logger):
    """Verbatim copy of the pre-vectorization implementation.

    Kept here as the oracle for the equivalence test.  Do not "improve" it:
    its value is that it is exactly what shipped before.
    """
    for band in filter_category.keys():
        df[f"filter_{band}"] = None
        df[f"flux_{band}"] = np.nan
        df[f"flux_error_{band}"] = np.nan

    def assign_filter_category(k):
        for band in filter_category.keys():
            if k in filter_category[band]:
                return band
        return None

    def detect_fluxcolumns(s):
        filters_found_one = []
        is_found_filter = False
        for c in s.keys():
            b = assign_filter_category(c)
            if b is not None:
                if np.isfinite(s[c]):
                    if s[f"filter_{b}"] is not None:
                        continue
                    flux = s[c]
                    is_found_filter = True
                    filters_found_one.append(c)
                    s[f"filter_{b}"] = c
                    s[f"flux_{b}"] = flux
                    try:
                        if np.isfinite(s[f"{c}_error"]):
                            s[f"flux_error_{b}"] = s[f"{c}_error"]
                    except KeyError:
                        pass
                    except TypeError:
                        pass
        return s, is_found_filter, filters_found_one

    vfunc = np.vectorize(detect_fluxcolumns, otypes=[dict, bool, np.ndarray])
    out, is_found, filters_found = vfunc(df.to_dict(orient="records"))
    dfout = pd.DataFrame.from_records(out)

    filters_found_flatten = [item for sub in filters_found for item in sub]
    dict_flux = {
        "success": is_found,
        "filters": np.unique(filters_found_flatten),
        "status": bool(np.all(is_found)),
    }

    for k in filter_category.keys():
        if dfout.loc[:, f"filter_{k}"].isna().all():
            dfout.drop(
                columns=[f"filter_{k}", f"flux_{k}", f"flux_error_{k}"], inplace=True
            )
        elif dfout.loc[:, f"flux_{k}"].isna().all():
            dfout.drop(columns=[f"flux_{k}"], inplace=True)
        elif dfout.loc[:, f"flux_error_{k}"].isna().all():
            dfout.drop(columns=[f"flux_error_{k}"], inplace=True)

    return dict_flux, dfout


def assert_same_result(df):
    """Run both implementations on independent copies and compare."""
    expected_flux, expected_df = reference_check_fluxcolumns(df.copy(deep=True))
    actual_flux, actual_df = check_fluxcolumns(df.copy(deep=True))

    np.testing.assert_array_equal(actual_flux["success"], expected_flux["success"])
    np.testing.assert_array_equal(actual_flux["filters"], expected_flux["filters"])
    assert actual_flux["status"] == expected_flux["status"]

    assert list(actual_df.columns) == list(expected_df.columns)
    # check_dtype=False: the reference round-trips through dicts and lets
    # from_records re-infer dtypes; the rewrite copies the frame and keeps the
    # originals.  Values must match, dtypes need not.
    pd.testing.assert_frame_equal(
        actual_df, expected_df, check_dtype=False, check_like=False
    )


BASE_COLUMNS = {
    "ob_code": ["a", "b", "c"],
    "obj_id": [1, 2, 3],
    "ra": [150.0, 150.1, 150.2],
    "dec": [2.0, 2.1, 2.2],
    "exptime": [900.0, 900.0, 900.0],
    "priority": [1.0, 1.0, 1.0],
    "resolution": ["L", "L", "L"],
    "reference_arm": ["r", "r", "r"],
}


def test_single_filter_column_matches_reference():
    df = pd.DataFrame({**BASE_COLUMNS, "g_hsc": [10.0, 20.0, np.nan]})
    assert_same_result(df)


def test_flux_error_column_matches_reference():
    df = pd.DataFrame(
        {
            **BASE_COLUMNS,
            "g_hsc": [10.0, 20.0, np.nan],
            "g_hsc_error": [1.0, np.nan, 3.0],
        }
    )
    assert_same_result(df)


def test_first_column_of_a_band_wins_not_the_filter_category_order():
    """g_ps1 is listed after g_hsc in filter_category, but comes first here."""
    df = pd.DataFrame(
        {
            **BASE_COLUMNS,
            "g_ps1": [10.0, np.nan, 30.0],
            "g_hsc": [11.0, 21.0, 31.0],
        }
    )
    _, out = check_fluxcolumns(df.copy(deep=True))
    assert list(out["filter_g"]) == ["g_ps1", "g_hsc", "g_ps1"]
    assert list(out["flux_g"]) == [10.0, 21.0, 30.0]
    assert_same_result(df)


def test_multiple_bands_match_reference():
    df = pd.DataFrame(
        {
            **BASE_COLUMNS,
            "g_hsc": [10.0, np.nan, 30.0],
            "r_ps1": [np.nan, 20.0, 30.0],
            "i_sdss": [15.0, 25.0, np.nan],
            "i_sdss_error": [1.5, np.nan, 3.5],
        }
    )
    assert_same_result(df)


def test_row_without_any_finite_flux_is_reported_unsuccessful():
    df = pd.DataFrame({**BASE_COLUMNS, "g_hsc": [10.0, np.nan, 30.0]})
    dict_flux, _ = check_fluxcolumns(df.copy(deep=True))
    assert list(dict_flux["success"]) == [True, False, True]
    assert dict_flux["status"] is False
    assert_same_result(df)


def test_no_filter_columns_at_all_matches_reference():
    df = pd.DataFrame(BASE_COLUMNS)
    assert_same_result(df)


@pytest.mark.parametrize(
    "path",
    [
        "docs/docs/examples/example_perseus_cluster_r60arcmin.csv",
        "docs/docs/examples/example_targetlist.csv",
    ],
)
def test_real_example_lists_match_reference(path):
    """The shipped example lists are the closest thing to production input."""
    from pfs_target_uploader.utils.io import load_input

    df, dict_load = load_input(path, format="csv")
    assert dict_load["status"]
    assert_same_result(df)
