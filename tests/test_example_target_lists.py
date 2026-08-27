"""Pins what each fixture in tests/data/ actually triggers in validate_input().

The fixtures exist so that other tests can reach a specific validation
outcome without hand-building a DataFrame.  That only works while each file
still triggers the failure mode its name claims, and a one-character edit is
enough to break that silently -- e.g. fixing the typo in an `ob_code` turns
invalid_string.csv into just another valid list, and every test relying on it
starts asserting nothing.  The table below is the guard against that.

The four fixtures whose expectations trail off into None are the ones that
matter most: they are the inputs that make validate_input() take each of its
early returns, leaving the checks below it unreached.  Anything consuming a
validation_status has to cope with those None statuses, so keeping one input
per early-return path available is the point of the set.

The pinned dates, the check-name list and the validation itself come from
conftest.py, which caches each fixture's result for the session -- the
visibility check is slow enough that validating the same 13 files once per
module is worth avoiding.
"""

from pathlib import Path

import pandas as pd
import pytest

from pfs_target_uploader.utils.checker import validate_input
from tests.conftest import CHECKS, DATA, DATE_BEGIN, DATE_END

PUBLISHED = Path(__file__).resolve().parents[1] / "docs" / "docs" / "examples"

# Everything in docs/docs/examples/ that is a target list. The other two files
# there are a pointing list and an admin proposal-ID list, neither of which
# validate_input() takes.
NOT_TARGET_LISTS = {"example_ppclist.csv", "example_admin_pslID.csv"}

# True = passed, False = failed, None = never reached (or, for flux_range,
# not applicable because no magnitude limits were supplied).
_PASS_THROUGH = dict(
    required_keys=True,
    empty=True,
    str=True,
    values=True,
    flux_columns=True,
    flux_values=True,
    flux_range=None,
    visibility=True,
    unique=True,
    internal_duplication=True,
)


def _expect(**overrides):
    return {**_PASS_THROUGH, **overrides}


# (filename, expected per-check status, overall status).  Any extra
# validate_input() arguments a fixture needs live in conftest.EXTRA_KWARGS.
FIXTURES = [
    (
        "valid_minimal.csv",
        _expect(),
        True,
    ),
    # --- inputs that make validate_input() return early ---------------------
    (
        "missing_required_column.csv",
        dict.fromkeys(CHECKS) | dict(required_keys=False),
        False,
    ),
    (
        "empty_header_only.csv",
        dict.fromkeys(CHECKS) | dict(required_keys=True, empty=False),
        False,
    ),
    (
        "invalid_string.csv",
        dict.fromkeys(CHECKS) | dict(required_keys=True, empty=True, str=False),
        False,
    ),
    (
        "invalid_value_range.csv",
        dict.fromkeys(CHECKS)
        | dict(required_keys=True, empty=True, str=True, values=False),
        False,
    ),
    # --- inputs that run every check, failing one of the later ones ---------
    (
        "missing_flux.csv",
        _expect(flux_columns=False),
        False,
    ),
    (
        "suspicious_flux_values.csv",
        _expect(flux_values=False),
        # flux_values is warning-only and is deliberately left out of the
        # overall success criteria.
        True,
    ),
    (
        "flux_out_of_ab_range.csv",
        _expect(flux_range=False),
        True,
    ),
    (
        "not_visible.csv",
        _expect(visibility=False),
        False,
    ),
    (
        "duplicate_ob_code.csv",
        _expect(unique=False),
        False,
    ),
    (
        "duplicate_obj_id_resolution.csv",
        _expect(unique=False),
        False,
    ),
    (
        "internal_duplication.csv",
        _expect(internal_duplication=False),
        True,
    ),
    (
        "internal_duplication_lm_pair.csv",
        # Same coordinates, but L and M are never duplicates of each other,
        # so this one must come back clean.
        _expect(),
        True,
    ),
]


@pytest.mark.parametrize(
    "filename, expected, expected_overall",
    FIXTURES,
    ids=[f[0] for f in FIXTURES],
)
def test_fixture_triggers_expected_outcome(
    validate_fixture, filename, expected, expected_overall
):
    validation_status, _ = validate_fixture(filename)

    actual = {k: validation_status[k]["status"] for k in CHECKS}
    # Cast away numpy bools so a mismatch prints as True/False, not np.True_.
    actual = {k: (None if v is None else bool(v)) for k, v in actual.items()}

    assert actual == expected
    assert bool(validation_status["status"]) is expected_overall

    # optional_keys is outside CHECKS -- it never gates an early return, so it
    # has no place in the ordered chain (see conftest).  Pinned here so it is
    # covered somewhere: the fixtures are deliberately minimal and carry none
    # of pmra/pmdec/parallax/tract/patch, which is what makes this False.  A
    # fixture that grew one would flip it and change the warning pane
    # show_results() renders.
    assert not validation_status["optional_keys"]["status"]
    assert len(validation_status["optional_keys"]["desc_warning"]) == 5


def test_every_early_return_path_is_covered():
    """The fixture set must keep one input per early return in validate_input().

    validate_input() has four of them today.  If a fifth is added, the check
    it guards will start coming back None for some input, and nothing here
    would notice unless a fixture reaches it -- so this asserts the set stays
    complete as the list of early returns grows.
    """
    early_return_checks = set()
    for _filename, expected, _overall in FIXTURES:
        unreached = [k for k in CHECKS if expected[k] is None]
        if not unreached:
            continue
        # flux_range comes back None when no magnitude limits are given, which
        # is "not applicable", not "not reached".
        if unreached == ["flux_range"]:
            continue
        # The check that failed is the last one before the unreached tail.
        first_unreached = CHECKS.index(unreached[0])
        early_return_checks.add(CHECKS[first_unreached - 1])

    assert early_return_checks == {"required_keys", "empty", "str", "values"}


@pytest.mark.parametrize(
    "path",
    sorted(p for p in PUBLISHED.glob("*.csv") if p.name not in NOT_TARGET_LISTS),
    ids=lambda p: p.name,
)
def test_published_example_passes_validation(path):
    """Every example list the docs site offers must survive validation.

    These are what users download and copy, so one that the uploader rejects
    is worse than no example at all.  Three of them had gone stale exactly
    that way: `reference_arm` became a required column and the files, which
    predate it, were never updated -- so they had been failing on a missing
    required column, silently, for as long as that column has existed.
    """
    df = pd.read_csv(path)
    validation_status, _ = validate_input(df, date_begin=DATE_BEGIN, date_end=DATE_END)

    failed = [
        k
        for k in CHECKS
        if validation_status[k]["status"] is not None
        and not validation_status[k]["status"]
        # Warning-only checks; a published example is allowed to trip them.
        and k not in ("flux_values", "internal_duplication")
    ]
    assert not failed, f"{path.name} fails: {failed}"
