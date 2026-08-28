"""Shared setup for the tests that drive validate_input().

The pinned dates and the check-name list live here because more than one
module needs them and they are easy to get subtly wrong: a second, hand-kept
copy of the check list can lose an entry without anything failing, which
silently narrows whatever it is used to parametrize.
"""

from copy import deepcopy
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pfs_target_uploader.utils.checker import validate_input

DATA = Path(__file__).resolve().parent / "data"

# Pinned rather than left to default: validate_input() falls back to "the next
# semester relative to now", which would make every visibility expectation
# depend on the day the suite runs.
DATE_BEGIN = date(2026, 2, 1)
DATE_END = date(2026, 7, 31)

# The checks validate_input() runs in sequence, in order.  Nine of them are
# pre-seeded with {"status": None} at checker.py:1371-1379; required_keys is
# assigned at 1390, before any return can happen.  Single source of truth --
# derive subsets from it rather than retyping.
#
# optional_keys is deliberately not here.  It is a check, and show_results()
# reads its fields unguarded, but it is assigned at checker.py:1391 alongside
# required_keys and never gates an early return, so it has no position in the
# chain.  Giving it one would break the inference in
# test_every_early_return_path_is_covered, which reads the entry before the
# first unreached check as the one that failed: with optional_keys sitting
# between required_keys and empty -- and never None, because it always runs --
# that inference would name optional_keys instead of required_keys.
# test_example_target_lists.py asserts its status separately.
CHECKS = [
    "required_keys",
    "empty",
    "str",
    "values",
    "flux_columns",
    "flux_values",
    "flux_range",
    "visibility",
    "unique",
    "internal_duplication",
]

# Fixtures that need more than the default arguments. The flux range check is
# skipped entirely unless a magnitude limit is supplied.
EXTRA_KWARGS = {"flux_out_of_ab_range.csv": dict(min_mag=12.0, max_mag=30.0)}

FIXTURE_FILES = sorted(p.name for p in DATA.glob("*.csv"))


@pytest.fixture(scope="session")
def validate_fixture():
    """Return a callable that validates a tests/data/ file, once per session.

    The visibility check dominates the suite's runtime, and the modules that
    use this each want the same set of validations; caching keeps that cost
    to one pass. Results are handed out as deep copies, so a test that edits
    a status dict -- test_show_results.py does -- cannot reach another test.
    """
    cache = {}

    def _validate(filename):
        if filename not in cache:
            df = pd.read_csv(DATA / filename)
            cache[filename] = validate_input(
                df,
                date_begin=DATE_BEGIN,
                date_end=DATE_END,
                **EXTRA_KWARGS.get(filename, {}),
            )
        return deepcopy(cache[filename])

    return _validate
