"""Tests for validation edge cases and internal-duplication detection.

check_internal_duplicate() used to align its per-row results back onto the
input DataFrame by matching on the *value* of `ob_code`
(`df["ob_code"].map(...)` against a Series indexed by `ob_code`). That breaks
the moment `ob_code` itself is duplicated -- exactly the case `check_unique()`
is meant to catch and report to the user -- because a non-uniquely-indexed
Series cannot be used as a `.map()` mapper. Instead of surfacing the
"duplicate ob_code" validation error, the whole validate step crashed with
`pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued
Index objects` (see GitHub issue #475).
"""

from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd

from pfs_target_uploader.utils.checker import check_internal_duplicate, validate_input
from pfs_target_uploader.utils.io import load_input


def test_validate_input_rejects_header_only_csv_with_clear_error():
    df, load_status = load_input(
        BytesIO(
            b"obj_id,ob_code,ra,dec,exptime,priority,resolution,reference_arm,g_hsc\n"
        ),
        format="csv",
    )

    assert load_status["status"] is True

    validation_status, df_output = validate_input(
        df,
        date_begin=date(2026, 9, 1),
        date_end=date(2027, 2, 28),
        single_exptime=900.0,
    )

    assert df_output.empty
    assert validation_status["status"] is False
    # check_keys() reports status via np.all(), i.e. a numpy bool -- test for
    # truthiness rather than identity to `True`.
    assert validation_status["required_keys"]["status"]
    assert validation_status["empty"] == {
        "status": False,
        "desc_error": "The file contains no data rows.",
    }
    assert validation_status["str"]["status"] is None


def test_check_internal_duplicate_does_not_crash_on_duplicate_ob_code():
    """Two rows sharing the same `ob_code` and identical coordinates must not
    raise -- they must be reported as duplicated, like any other exact
    coordinate match."""
    df = pd.DataFrame(
        {
            "ob_code": ["dup", "dup", "iso1", "iso2"],
            "ra": [150.0, 150.0, 160.0, 170.0],
            "dec": [2.0, 2.0, -10.0, 30.0],
            "resolution": ["L", "L", "L", "L"],
        }
    )

    result = check_internal_duplicate(df)

    assert result["status"] is False
    assert list(result["flags"]) == [True, True, False, False]
    assert len(result["nn_sep"]) == len(df)
    assert result["nn_sep"][0] == 0.0
    assert result["nn_sep"][1] == 0.0
    assert np.isnan(result["nn_sep"][2])
    assert np.isnan(result["nn_sep"][3])


def test_check_internal_duplicate_aligns_flags_to_row_position():
    """The duplicated pair sits away from the start of the frame (positions 1
    and 3), unlike every other test here where duplicates happen to occupy
    positions 0 and 1. That shape is indistinguishable from a compacted
    `0..n_dups-1` index, so it alone pins the invariant the fix relies on:
    dupcheck_internal() must return results indexed by input row position,
    not just by row order."""
    df = pd.DataFrame(
        {
            "ob_code": ["iso1", "dup", "iso2", "dup"],
            "ra": [160.0, 150.0, 170.0, 150.0],
            "dec": [-10.0, 2.0, 30.0, 2.0],
            "resolution": ["L", "L", "L", "L"],
        }
    )

    result = check_internal_duplicate(df)

    assert list(result["flags"]) == [False, True, False, True]
    assert result["nn_sep"][1] == 0.0
    assert result["nn_sep"][3] == 0.0
    assert np.isnan(result["nn_sep"][0])
    assert np.isnan(result["nn_sep"][2])


def test_check_internal_duplicate_all_isolated_when_ob_code_is_unique():
    """Non-regression: widely separated targets with unique `ob_code` are
    still reported as fully isolated."""
    df = pd.DataFrame(
        {
            "ob_code": ["a", "b", "c"],
            "ra": [150.0, 160.0, 170.0],
            "dec": [2.0, -10.0, 30.0],
            "resolution": ["L", "L", "L"],
        }
    )

    result = check_internal_duplicate(df)

    assert result["status"] is True
    assert list(result["flags"]) == [False, False, False]
    assert np.all(np.isnan(result["nn_sep"]))


def test_check_internal_duplicate_flags_only_the_near_pair():
    """Non-regression: a near-duplicate pair (well within 1 arcsec) is
    flagged with a matching nearest-neighbour separation while an unrelated
    isolated target is not, when `ob_code` is unique throughout."""
    near_offset_deg = 0.5 / 3600.0  # 0.5 arcsec
    df = pd.DataFrame(
        {
            "ob_code": ["x", "y", "z"],
            "ra": [150.0, 150.0 + near_offset_deg, 170.0],
            "dec": [2.0, 2.0, 30.0],
            "resolution": ["L", "L", "L"],
        }
    )

    result = check_internal_duplicate(df)

    assert result["status"] is False
    assert list(result["flags"]) == [True, True, False]
    assert result["nn_sep"][0] == result["nn_sep"][1]
    assert 0.0 < result["nn_sep"][0] < 1.0
    assert np.isnan(result["nn_sep"][2])


def test_validate_input_reports_duplicate_ob_code_without_crashing():
    """End-to-end: a target list with a duplicate `ob_code` must fail
    validation with the existing 'unique' error instead of crashing before
    that error can reach the caller."""
    df = pd.DataFrame(
        {
            "obj_id": [1, 2, 3, 3],
            "ob_code": ["a", "b", "c", "c"],
            "ra": [150.0, 150.1, 150.2, 150.2],
            "dec": [2.0, 2.1, 2.2, 2.2],
            "priority": [1, 1, 1, 1],
            "exptime": [900.0, 900.0, 900.0, 900.0],
            "resolution": ["L", "L", "L", "L"],
            "reference_arm": ["r", "r", "r", "r"],
        }
    )

    validation_status, _df_output = validate_input(df)

    assert validation_status["unique"]["status"] is False
    assert validation_status["status"] is False
    assert "internal_duplication" in validation_status
