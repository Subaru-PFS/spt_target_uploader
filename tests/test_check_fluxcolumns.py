"""Tests for the vectorized flux-column detector.

check_fluxcolumns() used to walk every row in Python via np.vectorize, which
cost 5.3 s on a 30,000-row list.  The rewrite works per band on whole columns.
The contract it must keep is subtle -- the winning filter for a band is the
first *DataFrame column* holding a finite value, which is column order, not
the order the band's filters are listed in filter_category -- so the tests
below pin it against a frozen copy of the old implementation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from pfs_target_uploader.utils import filter_category
from pfs_target_uploader.utils.checker import check_fluxcolumns

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "docs" / "docs" / "examples"


def reference_check_fluxcolumns(df, filter_category=filter_category, logger=logger):
    """Copy of the pre-vectorization implementation, verbatim modulo logging.

    Kept here as the oracle for the equivalence test.  Do not "improve" it:
    its value is that it is otherwise exactly what shipped before.  Only the
    logger.debug/logger.warning calls are dropped, because they had no bearing
    on the returned values.
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


def test_fallback_within_a_band_is_decided_per_object():
    """Two columns of the same band, each missing for different objects.

    The band's winner is chosen per object, not once for the whole column:
    an object whose leftmost column is NaN falls through to the next one.
    The flux *error* has to follow the column that actually won -- picking
    g_hsc_error for an object served by g_sdss would attach the wrong
    uncertainty to the flux.
    """
    df = pd.DataFrame(
        {
            "ob_code": ["r0", "r1", "r2", "r3"],
            "obj_id": [1, 2, 3, 4],
            "ra": [150.0] * 4,
            "dec": [2.0] * 4,
            "exptime": [900.0] * 4,
            "priority": [1.0] * 4,
            "resolution": ["L"] * 4,
            "reference_arm": ["r"] * 4,
            "g_hsc": [10.0, np.nan, 11.0, np.nan],
            # An error is present even where the flux is missing, so picking
            # the wrong column's error would go unnoticed without this.
            "g_hsc_error": [1.0, 0.9, 1.1, 0.8],
            "g_sdss": [np.nan, 20.0, 21.0, np.nan],
            "g_sdss_error": [2.0, 2.1, 2.2, 2.3],
        }
    )

    dict_flux, out = check_fluxcolumns(df.copy(deep=True))

    assert out["filter_g"].tolist() == ["g_hsc", "g_sdss", "g_hsc", None]
    assert out["flux_g"].tolist()[:3] == [10.0, 20.0, 11.0]
    assert np.isnan(out["flux_g"].tolist()[3])
    assert out["flux_error_g"].tolist()[:3] == [1.0, 2.1, 1.1]
    assert np.isnan(out["flux_error_g"].tolist()[3])
    assert list(dict_flux["success"]) == [True, True, True, False]
    assert sorted(dict_flux["filters"]) == ["g_hsc", "g_sdss"]

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
    "name",
    [
        "example_perseus_cluster_r60arcmin.csv",
        "example_targetlist.csv",
        # Several columns per band with _error companions on both, so this
        # one exercises real per-object fallback across g_ps1/g_sdss,
        # i2_hsc/i_sdss, z_hsc/z_ps1 and y_hsc/y_ps1.
        "example_targetlist_random10.csv",
    ],
)
def test_real_example_lists_match_reference(name):
    """The shipped example lists are the closest thing to production input."""
    from pfs_target_uploader.utils.io import load_input

    df, dict_load = load_input(str(EXAMPLES / name), format="csv")
    assert dict_load["status"]
    assert_same_result(df)


def test_text_in_an_error_column_no_longer_discards_the_whole_column():
    """The one shipped list where the rewrite deliberately changes the output.

    example_targetlist_random100.csv has a truncated final row ending
    "...,10752.32,875.41>". That stray ">" makes pandas read the entire
    i2_hsc_error column as strings. The old code then hit
    np.isfinite(str) -> TypeError on every row, swallowed it with a bare
    `except TypeError: pass`, and so threw away all 66 perfectly good error
    values along with the one corrupt cell.

    pd.to_numeric(errors="coerce") recovers them and drops only the bad cell.
    The column itself is present either way -- other i-band error columns keep
    it alive -- so what changes is 66 objects gaining a real flux_error_i in
    the target_<id>.ecsv that goes to targetDB.
    """
    from pfs_target_uploader.utils.io import load_input

    df, dict_load = load_input(
        str(EXAMPLES / "example_targetlist_random100.csv"), format="csv"
    )
    assert dict_load["status"]
    # Precondition: the corruption is still in the file. If someone fixes the
    # CSV this assertion fires and the test can simply be deleted.
    assert df["i2_hsc_error"].dtype == object

    _, out = check_fluxcolumns(df.copy(deep=True))
    _, ref = reference_check_fluxcolumns(df.copy(deep=True))

    # Same columns, same everything else.
    assert list(out.columns) == list(ref.columns)
    differing = [c for c in out.columns if not out[c].equals(ref[c])]
    assert differing == ["flux_error_i"]

    recovered = out["flux_error_i"].notna() & ref["flux_error_i"].isna()
    assert recovered.sum() == 66
    # And nothing the old code had was lost.
    assert not (ref["flux_error_i"].notna() & out["flux_error_i"].isna()).any()


def test_flux_and_filter_columns_come_back_with_usable_dtypes():
    """check_fluxrange() runs np.isfinite and >=/<= on flux_<band>, so it has
    to be float64 even when the source column was object dtype."""
    df = pd.DataFrame(
        {
            **BASE_COLUMNS,
            "g_hsc": [10.0, "bogus", 30.0],
            "g_hsc_error": [1.0, 2.0, "junk"],
        }
    )
    assert df["g_hsc"].dtype == object
    assert df["g_hsc_error"].dtype == object

    _, out = check_fluxcolumns(df)

    assert out["flux_g"].dtype == np.float64
    assert out["flux_error_g"].dtype == np.float64
    assert out["filter_g"].dtype == object
    # The text cells became missing rather than poisoning the whole column.
    assert out["flux_g"].tolist()[0] == 10.0
    assert out["flux_error_g"].tolist()[0] == 1.0
    assert np.isnan(out["flux_error_g"].tolist()[2])


def test_the_callers_frame_is_left_untouched():
    """The old code appended 18 columns to the input in place; the rewrite
    works on a copy. Several tests here pass an uncopied frame, which is only
    safe because of that."""
    df = pd.DataFrame({**BASE_COLUMNS, "g_hsc": [10.0, 20.0, np.nan]})
    before = df.copy(deep=True)

    check_fluxcolumns(df)

    pd.testing.assert_frame_equal(df, before)


def test_the_index_of_the_input_is_preserved():
    """from_records() used to hand back a fresh RangeIndex; copying keeps
    whatever the caller had. Every consumer masks positionally, so preserving
    it is the less surprising contract -- pinned here so it stays a decision."""
    df = pd.DataFrame({**BASE_COLUMNS, "g_hsc": [10.0, 20.0, 30.0]})
    df.index = [10, 20, 30]

    _, out = check_fluxcolumns(df)

    assert list(out.index) == [10, 20, 30]


def test_an_empty_frame_does_not_crash():
    """The old code raised KeyError: 'filter_g' out of the cleanup, because
    from_records([]) produces a frame with no columns at all."""
    df = pd.DataFrame({k: [] for k in BASE_COLUMNS})

    dict_flux, out = check_fluxcolumns(df)

    assert len(out) == 0
    assert list(dict_flux["success"]) == []
    assert dict_flux["status"] is True


def test_fallback_to_a_column_without_an_error_companion():
    """g_sdss has no g_sdss_error here: an object served by it must get NaN,
    not the g_hsc_error sitting on the same row."""
    df = pd.DataFrame(
        {
            **BASE_COLUMNS,
            "g_hsc": [10.0, np.nan, np.nan],
            "g_hsc_error": [1.0, 0.9, 0.8],
            "g_sdss": [np.nan, 20.0, np.nan],
        }
    )

    _, out = check_fluxcolumns(df.copy(deep=True))

    assert out["filter_g"].tolist() == ["g_hsc", "g_sdss", None]
    assert out["flux_error_g"].tolist()[0] == 1.0
    assert np.isnan(out["flux_error_g"].tolist()[1])
    assert np.isnan(out["flux_error_g"].tolist()[2])

    assert_same_result(df)


def _captured_warnings(df):
    messages = []
    handler_id = logger.add(
        lambda m: messages.append(m.record["message"]), level="WARNING"
    )
    try:
        check_fluxcolumns(df)
    finally:
        logger.remove(handler_id)
    return messages


def test_duplicate_filter_warning_is_emitted_once_per_column():
    """The old code warned once per row; on a 30k list that was 30k lines.

    Only 3 of the 50 objects carry both filters, so this also pins the count
    in the aggregated message -- the part most likely to drift silently.
    """
    n = 50
    g_ps1 = np.full(n, np.nan)
    g_ps1[:3] = 20.0
    df = pd.DataFrame(
        {
            "ob_code": [f"ob{i}" for i in range(n)],
            "obj_id": np.arange(n),
            "ra": np.full(n, 150.0),
            "dec": np.full(n, 2.0),
            "exptime": np.full(n, 900.0),
            "priority": np.full(n, 1.0),
            "resolution": ["L"] * n,
            "reference_arm": ["r"] * n,
            "g_hsc": np.full(n, 10.0),
            "g_ps1": g_ps1,
        }
    )

    skipped = [m for m in _captured_warnings(df) if "g_ps1 is skipped" in m]
    assert len(skipped) == 1
    assert "3 object(s)" in skipped[0]


def test_entries_coerced_to_missing_are_reported():
    """An object silently falling through to the band's next column, or being
    reported as "flux missing", is untraceable unless the discarded value is
    named."""
    df = pd.DataFrame({**BASE_COLUMNS, "g_hsc": [10.0, "n/a", 30.0]})

    reported = [m for m in _captured_warnings(df) if m.startswith("g_hsc:")]

    assert len(reported) == 1
    assert "1 entry(ies)" in reported[0]
    assert "n/a" in reported[0]


def test_an_empty_cell_is_not_reported_as_unusable():
    """A genuinely blank flux is normal input, not something to warn about."""
    df = pd.DataFrame({**BASE_COLUMNS, "g_hsc": [10.0, np.nan, 30.0]})

    assert [m for m in _captured_warnings(df) if m.startswith("g_hsc:")] == []


def test_non_numeric_flux_reads_as_missing_instead_of_raising():
    """It used to raise an uncaught TypeError out of np.isfinite."""
    df = pd.DataFrame({**BASE_COLUMNS, "g_hsc": [10.0, "bogus", 30.0]})

    dict_flux, out = check_fluxcolumns(df)

    assert list(dict_flux["success"]) == [True, False, True]
    assert dict_flux["status"] is False
    assert out["filter_g"].tolist() == ["g_hsc", None, "g_hsc"]
