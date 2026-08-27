"""Tests for the validation memoization in FileInputWidgets.

cb_validate, cb_PPP and cb_submit each call validate() on the same inputs,
so a 30,000-row list used to pay for a full validation up to three times.
validate() now returns its previous result when nothing it reads has changed.

The subtle part is secret_token: validate() stamps its first 7 characters
onto every ob_code, and cb_submit assigns a fresh token after an upload. A
cache that ignored the token would hand back ob_codes carrying the old one.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from pfs_target_uploader.widgets.FileInputWidgets import FileInputWidgets

DATE_BEGIN = date(2026, 9, 1)
DATE_END = date(2027, 2, 28)

CSV = b"""obj_id,ob_code,ra,dec,exptime,priority,resolution,reference_arm,g_hsc
1,a,150.0,2.0,900.0,1,L,r,10.0
2,b,150.1,2.1,900.0,1,L,r,20.0
"""


@pytest.fixture
def widget(monkeypatch, tmp_path):
    """A FileInputWidgets with a loaded file and a fixed upload ID."""
    import panel as pn

    class _Notifications:
        def error(self, message, **kwargs):
            raise AssertionError(f"unexpected error notification: {message}")

        def info(self, message, **kwargs):
            pass

    monkeypatch.setattr(
        type(pn.state), "notifications", property(lambda self: _Notifications())
    )

    w = FileInputWidgets()
    w.use_db = False
    # Swapping the file mid-test sends validate() down the "new file" branch,
    # which mints a fresh upload ID by globbing this directory.
    w.output_dir = str(tmp_path)
    w.file_input.filename = "targets.csv"
    w.file_input.mime_type = "text/csv"
    w.file_input.value = CSV
    w.secret_token = "0123456789abcdef"
    # Match what validate() would have recorded, so it takes the
    # "identical to the previous validation" path and keeps the token.
    w.previous_filename = "targets.csv"
    w.previous_mime_type = "text/csv"
    w.previous_value = CSV
    return w


def _validate(w, **overrides):
    kwargs = dict(
        date_begin=DATE_BEGIN,
        date_end=DATE_END,
        single_exptime=900.0,
        min_mag=12.0,
        max_mag=25.0,
    )
    kwargs.update(overrides)
    return w.validate(**kwargs)


def _has_cache(w, **overrides):
    kwargs = dict(
        date_begin=DATE_BEGIN,
        date_end=DATE_END,
        single_exptime=900.0,
        min_mag=12.0,
        max_mag=25.0,
    )
    kwargs.update(overrides)
    return w.has_cached_validation(**kwargs)


def test_second_identical_validation_is_served_from_cache(widget, monkeypatch):
    status_1, df_1 = _validate(widget)
    assert status_1 is not None

    # Any real work would go through validate_input; make it explode.
    # Reached via sys.modules, not `import ...widgets.FileInputWidgets`:
    # widgets/__init__.py rebinds that name on the package to the *class*.
    import sys

    module = sys.modules["pfs_target_uploader.widgets.FileInputWidgets"]

    def _boom(*args, **kwargs):
        raise AssertionError("validate_input must not run on a cache hit")

    monkeypatch.setattr(module, "validate_input", _boom)

    status_2, df_2 = _validate(widget)

    assert status_2["status"] == status_1["status"]
    pd.testing.assert_frame_equal(df_2, df_1)


def test_cache_hands_out_independent_copies(widget):
    _, df_1 = _validate(widget)
    df_1["exptime"] = -999.0

    _, df_2 = _validate(widget)

    assert (df_2["exptime"] == 900.0).all()


def test_changing_the_observation_period_misses_the_cache(widget):
    _validate(widget)
    assert not _has_cache(widget, date_end=DATE_END + timedelta(days=30))


def test_changing_single_exptime_misses_the_cache(widget):
    _validate(widget)
    assert not _has_cache(widget, single_exptime=1800.0)


def test_changing_min_mag_misses_the_cache(widget):
    """min_mag is derived from the observation type, so this covers a
    queue -> filler switch as well."""
    _validate(widget)
    assert not _has_cache(widget, min_mag=14.0)


def test_a_new_upload_id_misses_the_cache(widget):
    """cb_submit assigns a fresh token after an upload; the cached frame
    carries the old one stamped onto every ob_code."""
    _, df_1 = _validate(widget)
    assert df_1["ob_code"].iloc[0].endswith("_0123456")

    widget.secret_token = "fedcba9876543210"
    assert not _has_cache(widget)

    _, df_2 = _validate(widget)
    assert df_2["ob_code"].iloc[0].endswith("_fedcba9")


def test_changing_the_file_content_misses_the_cache(widget):
    _validate(widget)
    widget.file_input.value = CSV.replace(b"150.0", b"151.0")
    assert not _has_cache(widget)


def test_reset_clears_the_cache(widget):
    _validate(widget)
    widget.reset()
    assert widget.last_validation_key is None
    assert not _has_cache(widget)


def test_invalidate_cache_drops_the_entry(widget):
    """cb_submit calls this after assigning a fresh upload ID: the entry can
    never be hit again, but would pin its frames until the session ends."""
    _validate(widget)
    assert _has_cache(widget)

    widget.invalidate_cache()

    assert widget._cached_result is None
    assert widget.last_validation_key is None
    assert not _has_cache(widget)


@pytest.mark.parametrize(
    "break_it, expected_error",
    [
        # date_begin >= date_end
        (lambda w: None, "dates"),
        # unreadable content
        # Raw bytes: pandas happily parses most malformed text, so this has
        # to be something read_csv genuinely refuses.
        (lambda w: setattr(w.file_input, "value", bytes(range(256))), "load"),
        # nothing selected
        (lambda w: setattr(w.file_input, "filename", None), "no file"),
    ],
    ids=["bad-dates", "unreadable", "no-file"],
)
def test_a_failed_validation_clears_the_key(
    widget, monkeypatch, break_it, expected_error
):
    """last_validation_key must describe the *most recent* call.

    pn_app's render gate reads it to decide whether the panels already show
    the current result. If a failure left the previous run's key in place, the
    gate would suppress every later attempt to render that result and the user
    could only recover by reloading the page.
    """
    import panel as pn

    class _Quiet:
        def error(self, message, **kwargs):
            pass

        def info(self, message, **kwargs):
            pass

    _validate(widget)
    assert widget.last_validation_key is not None

    monkeypatch.setattr(
        type(pn.state), "notifications", property(lambda self: _Quiet())
    )
    break_it(widget)

    if expected_error == "dates":
        status, _ = widget.validate(
            date_begin=DATE_END, date_end=DATE_BEGIN, single_exptime=900.0
        )
    else:
        status, _ = _validate(widget)

    assert status is None
    assert widget.last_validation_key is None


def test_a_cache_hit_re_raises_the_large_list_notice(widget, monkeypatch):
    """Every callback clears notifications before validating, so the notice
    raised on the first run is gone by the time the user starts the
    multi-minute simulation it warns about."""
    import panel as pn

    seen = []

    class _Recording:
        def error(self, message, **kwargs):
            raise AssertionError(message)

        def info(self, message, **kwargs):
            seen.append(message)

    monkeypatch.setattr(
        type(pn.state), "notifications", property(lambda self: _Recording())
    )

    # Threshold of 1 makes the two-row fixture "very large".
    widget.validate(
        date_begin=DATE_BEGIN,
        date_end=DATE_END,
        single_exptime=900.0,
        min_mag=12.0,
        max_mag=25.0,
        warn_threshold=1,
    )
    first = len([m for m in seen if "very large" in m])
    assert first == 1

    widget.validate(
        date_begin=DATE_BEGIN,
        date_end=DATE_END,
        single_exptime=900.0,
        min_mag=12.0,
        max_mag=25.0,
        warn_threshold=1,
    )
    assert len([m for m in seen if "very large" in m]) == 2


def test_a_file_swapped_in_mid_validation_does_not_poison_the_cache(
    widget, monkeypatch
):
    """validate() runs in a worker thread while the browser can still deliver
    a new upload. Reading the widget again at the end would file this run's
    result under the *next* file's key, and every later validation of that
    file would return the wrong targets -- for the rest of the session.
    """
    # sys.modules, not a plain import: widgets/__init__.py rebinds this name
    # on the package to the *class*.
    import sys

    module = sys.modules["pfs_target_uploader.widgets.FileInputWidgets"]
    original = module.validate_input
    other = CSV.replace(b"150.0", b"170.0")

    def swap_then_validate(*args, **kwargs):
        # Stand in for the upload landing while the worker thread is busy.
        widget.file_input.value = other
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "validate_input", swap_then_validate)
    _validate(widget)
    monkeypatch.setattr(module, "validate_input", original)

    # The result belongs to the ORIGINAL bytes, so the swapped-in file must
    # not be served from cache.
    assert widget.file_input.value == other
    assert not _has_cache(widget)
