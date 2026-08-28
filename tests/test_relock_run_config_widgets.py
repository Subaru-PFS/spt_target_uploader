"""Tests for relock_run_config_widgets() in pn_app.

cb_validate, cb_PPP and cb_submit each re-enable the whole sidebar when they
finish. relock_run_config_widgets() re-applies the locks that must outlast
that blanket re-enable:

- queue / filler always pin single_exptime and the user pointing list;
  filler also keeps the Simulate button off.
- Whenever a submittable simulation result is on screen (the Submit button is
  enabled), single_exptime, the pointing list and both date pickers freeze --
  all four feed cb_submit's re-validation, so moving any of them between
  Simulate and Submit desyncs target_<id>.ecsv from the psl_*/ppc_* products
  of the run being submitted (issue #486).

The function only ever disables; each callback owns the matching re-enable.
"""

import pytest

from pfs_target_uploader.pn_app import relock_run_config_widgets


class FakeWidget:
    def __init__(self, disabled=False):
        self.disabled = disabled


@pytest.fixture
def widgets():
    """The six widgets the function touches, all starting enabled."""
    return {
        "single_exptime": FakeWidget(),
        "pointing_list": FakeWidget(),
        "date_begin": FakeWidget(),
        "date_end": FakeWidget(),
        "run_ppp_button": FakeWidget(),
        "submit_button": FakeWidget(disabled=True),
    }


def call(obs_type, widgets):
    relock_run_config_widgets(
        obs_type,
        single_exptime=widgets["single_exptime"],
        pointing_list=widgets["pointing_list"],
        date_begin=widgets["date_begin"],
        date_end=widgets["date_end"],
        run_ppp_button=widgets["run_ppp_button"],
        submit_button=widgets["submit_button"],
    )


def disabled_names(widgets):
    return {
        name for name, w in widgets.items() if w.disabled and name != "submit_button"
    }


def test_queue_without_a_result_pins_exptime_and_pointing_list(widgets):
    call("queue", widgets)
    assert disabled_names(widgets) == {"single_exptime", "pointing_list"}


def test_queue_with_a_result_also_freezes_the_date_pickers(widgets):
    widgets["submit_button"].disabled = False
    call("queue", widgets)
    assert disabled_names(widgets) == {
        "single_exptime",
        "pointing_list",
        "date_begin",
        "date_end",
    }


def test_filler_without_a_result_pins_exptime_pointing_list_and_simulate(widgets):
    call("filler", widgets)
    assert disabled_names(widgets) == {
        "single_exptime",
        "pointing_list",
        "run_ppp_button",
    }


def test_filler_with_a_result_also_freezes_the_date_pickers(widgets):
    widgets["submit_button"].disabled = False
    call("filler", widgets)
    assert disabled_names(widgets) == {
        "single_exptime",
        "pointing_list",
        "run_ppp_button",
        "date_begin",
        "date_end",
    }


def test_classical_without_a_result_locks_nothing(widgets):
    call("classical", widgets)
    assert disabled_names(widgets) == set()


def test_classical_with_a_result_freezes_the_four_visibility_inputs(widgets):
    widgets["submit_button"].disabled = False
    call("classical", widgets)
    assert disabled_names(widgets) == {
        "single_exptime",
        "pointing_list",
        "date_begin",
        "date_end",
    }


def test_classical_with_a_result_leaves_the_simulate_button_alone(widgets):
    widgets["submit_button"].disabled = False
    call("classical", widgets)
    assert widgets["run_ppp_button"].disabled is False


def test_never_re_enables_an_already_disabled_widget(widgets):
    for w in widgets.values():
        w.disabled = True
    call("classical", widgets)  # the branch that would lock nothing
    assert all(w.disabled for w in widgets.values())


def test_does_not_touch_the_submit_button(widgets):
    widgets["submit_button"].disabled = False
    call("queue", widgets)
    assert widgets["submit_button"].disabled is False


def test_wiring_matches_the_real_widget_classes():
    """The attribute paths pn_app's closure forwards must stay valid.

    Guards against a widget class renaming e.g. single_exptime without the
    closure in target_uploader_app being updated in step.
    """
    from pfs_target_uploader.widgets import (
        DatePickerWidgets,
        ObsTypeWidgets,
        PPCInputWidgets,
        RunPppButtonWidgets,
        SubmitButtonWidgets,
    )

    obs_type = ObsTypeWidgets()
    ppcinput = PPCInputWidgets()
    dates = DatePickerWidgets()
    ppp_button = RunPppButtonWidgets()
    submit_button = SubmitButtonWidgets()
    submit_button.submit.disabled = False

    relock_run_config_widgets(
        "classical",
        single_exptime=obs_type.single_exptime,
        pointing_list=ppcinput.file_input,
        date_begin=dates.date_begin,
        date_end=dates.date_end,
        run_ppp_button=ppp_button.PPPrun,
        submit_button=submit_button.submit,
    )

    assert obs_type.single_exptime.disabled is True
    assert ppcinput.file_input.disabled is True
    assert dates.date_begin.disabled is True
    assert dates.date_end.disabled is True
