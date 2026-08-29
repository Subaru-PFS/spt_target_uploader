"""Tests for the run-configuration locking policy in pn_app."""

from types import SimpleNamespace

import pytest

from pfs_target_uploader.pn_app import relock_run_config_widgets


class FakeWidget:
    def __init__(self, *, disabled=False, value=None):
        self.disabled = disabled
        self.value = value


@pytest.fixture
def widgets():
    obs_type = SimpleNamespace(
        obs_type=FakeWidget(value="classical"),
        single_exptime=FakeWidget(),
    )
    ppc_input = SimpleNamespace(file_input=FakeWidget())
    dates = SimpleNamespace(date_begin=FakeWidget(), date_end=FakeWidget())
    ppp_button = SimpleNamespace(PPPrun=FakeWidget())
    return obs_type, ppc_input, dates, ppp_button


def call(widgets, observation_type, *, has_pending_run):
    obs_type, ppc_input, _, _ = widgets
    obs_type.obs_type.value = observation_type
    relock_run_config_widgets(*widgets, has_pending_run=has_pending_run)


def disabled_names(widgets):
    obs_type, ppc_input, dates, ppp_button = widgets
    return {
        name
        for name, widget in {
            "obs_type": obs_type.obs_type,
            "single_exptime": obs_type.single_exptime,
            "pointing_list": ppc_input.file_input,
            "date_begin": dates.date_begin,
            "date_end": dates.date_end,
            "run_ppp_button": ppp_button.PPPrun,
        }.items()
        if widget.disabled
    }


@pytest.mark.parametrize(
    ("observation_type", "has_pending_run", "expected"),
    [
        ("classical", False, set()),
        (
            "classical",
            True,
            {
                "obs_type",
                "single_exptime",
                "pointing_list",
                "date_begin",
                "date_end",
            },
        ),
        ("queue", False, {"single_exptime", "pointing_list"}),
        (
            "queue",
            True,
            {
                "obs_type",
                "single_exptime",
                "pointing_list",
                "date_begin",
                "date_end",
            },
        ),
        ("filler", False, {"single_exptime", "pointing_list", "run_ppp_button"}),
        (
            "filler",
            True,
            {
                "obs_type",
                "single_exptime",
                "pointing_list",
                "date_begin",
                "date_end",
                "run_ppp_button",
            },
        ),
    ],
)
def test_locks_match_observation_type_and_pending_run(
    widgets, observation_type, has_pending_run, expected
):
    call(widgets, observation_type, has_pending_run=has_pending_run)
    assert disabled_names(widgets) == expected


def test_pending_run_does_not_disable_simulate_in_classical_mode(widgets):
    call(widgets, "classical", has_pending_run=True)
    assert widgets[3].PPPrun.disabled is False


def test_never_re_enables_an_already_disabled_widget(widgets):
    for widget in (
        widgets[0].obs_type,
        widgets[0].single_exptime,
        widgets[1].file_input,
        widgets[2].date_begin,
        widgets[2].date_end,
        widgets[3].PPPrun,
    ):
        widget.disabled = True

    call(widgets, "queue", has_pending_run=True)

    assert disabled_names(widgets) == {
        "obs_type",
        "single_exptime",
        "pointing_list",
        "date_begin",
        "date_end",
        "run_ppp_button",
    }


def test_real_widget_holders_lock_all_pending_run_inputs():
    from pfs_target_uploader.widgets import (
        DatePickerWidgets,
        ObsTypeWidgets,
        PPCInputWidgets,
        RunPppButtonWidgets,
    )

    obs_type = ObsTypeWidgets()
    ppc_input = PPCInputWidgets()
    dates = DatePickerWidgets()
    ppp_button = RunPppButtonWidgets()
    obs_type.single_exptime.disabled = False

    relock_run_config_widgets(
        obs_type,
        ppc_input,
        dates,
        ppp_button,
        has_pending_run=True,
    )

    assert obs_type.obs_type.disabled is True
    assert obs_type.single_exptime.disabled is True
    assert ppc_input.file_input.disabled is True
    assert dates.date_begin.disabled is True
    assert dates.date_end.disabled is True
