"""show_results() must survive checks that validate_input() never reached.

validate_input() seeds every check with ``{"status": None}`` and returns early
at the first failure, so anything below that failure keeps the None -- and,
crucially, has none of the other fields the renderer reads (``success``,
``flags``, ``filters``).  Touching one raises KeyError, and cb_validate() in
pn_app.py calls show_results() with no try/except: the exception escapes the
callback, panel_timer.timer(on=False) never runs, and the user is left on a
spinner that never stops.  A silent hang, not a visible error.

The failure is one early return away at all times.  Today validate_input()
returns early at four places and every section show_results() renders
unguarded happens to sit above all of them; add a fifth in the wrong place and
the renderer starts reading fields that are not there.  These tests fix the
contract in place rather than the accident: whatever the control flow above
does, show_results() has to cope with any check coming back None.
"""

from copy import deepcopy

import pandas as pd
import pytest

from pfs_target_uploader.utils.checker import validate_input
from pfs_target_uploader.widgets.ValidationResultWidgets import ValidationResultWidgets
from tests.conftest import CHECKS, DATA, DATE_BEGIN, DATE_END, FIXTURE_FILES

# Derived, not retyped: required_keys is the first thing validate_input()
# checks, so it is the one entry that can never come back unreached.
SKIPPABLE = [c for c in CHECKS if c != "required_keys"]


@pytest.fixture(scope="module")
def clean_result():
    """A validation_status in which every check ran, plus the frame it read.

    Validated here rather than through conftest's cache because this one needs
    magnitude limits, which make flux_range produce a real result instead of
    the "no limits configured" None.  The baseline has to have every field
    populated for the parametrization below to strip them off one tail at a
    time.  It costs one extra validation, and the visibility results for this
    patch of sky are already cached by the time it runs.
    """
    df = pd.read_csv(DATA / "valid_minimal.csv")
    validation_status, df_validated = validate_input(
        df, date_begin=DATE_BEGIN, date_end=DATE_END, min_mag=10.0, max_mag=30.0
    )
    assert validation_status["status"], "fixture must pass validation"
    for check in SKIPPABLE:
        assert (
            validation_status[check]["status"] is not None
        ), f"baseline must have run {check}"
    return validation_status, df_validated


@pytest.mark.parametrize("first_skipped", SKIPPABLE, ids=SKIPPABLE)
def test_renders_when_a_check_never_ran(clean_result, first_skipped):
    """Simulate an early return before `first_skipped` and render the result.

    Parametrized over every position an early return could be added to
    validate_input(), not just the four that exist -- the point is that
    show_results() stops depending on where they happen to be.
    """
    validation_status, df = deepcopy(clean_result)
    for key in SKIPPABLE[SKIPPABLE.index(first_skipped) :]:
        validation_status[key] = {"status": None}
    validation_status["status"] = False

    # Must not raise.  A KeyError here is the spinner-that-never-stops bug.
    ValidationResultWidgets().show_results(df, validation_status)


@pytest.mark.parametrize("filename", FIXTURE_FILES)
def test_renders_every_fixture_outcome(validate_fixture, filename):
    """Render each real validation outcome, including the four early returns.

    Complements the synthetic test above: those states are reached through
    validate_input() itself rather than assembled by hand, so a status dict
    this file gets wrong cannot hide a break here.
    """
    validation_status, df_validated = validate_fixture(filename)

    ValidationResultWidgets().show_results(df_validated, validation_status)
