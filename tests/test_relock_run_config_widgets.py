"""Tests for the run-configuration locking policy in pn_app."""

from types import SimpleNamespace

import pytest

from pfs_target_uploader.pn_app import (
    relock_run_config_widgets,
    run_config_matches,
)
from pfs_target_uploader.widgets.buttons import (
    SubmitButtonWidgets,
    stylesheet,
    stylesheet_warning,
)


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


def call(widgets, observation_type):
    obs_type, ppc_input, _, _ = widgets
    obs_type.obs_type.value = observation_type
    relock_run_config_widgets(*widgets)


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
    ("observation_type", "expected"),
    [
        ("classical", set()),
        ("queue", {"single_exptime", "pointing_list"}),
        ("filler", {"single_exptime", "pointing_list", "run_ppp_button"}),
    ],
)
def test_locks_match_observation_type(widgets, observation_type, expected):
    call(widgets, observation_type)
    assert disabled_names(widgets) == expected


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

    call(widgets, "queue")

    assert disabled_names(widgets) == {
        "obs_type",
        "single_exptime",
        "pointing_list",
        "date_begin",
        "date_end",
        "run_ppp_button",
    }


def test_real_widget_holders_leave_classical_config_editable():
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
    obs_type.obs_type.value = "classical"
    obs_type.single_exptime.disabled = False

    relock_run_config_widgets(obs_type, ppc_input, dates, ppp_button)

    assert obs_type.obs_type.disabled is False
    assert obs_type.single_exptime.disabled is False
    assert ppc_input.file_input.disabled is False
    assert dates.date_begin.disabled is False
    assert dates.date_end.disabled is False


def test_completed_run_requires_an_exact_current_config_snapshot():
    completed_config = {
        "observation_type": "classical",
        "date_begin": "2026-02-01",
        "date_end": "2026-07-31",
        "single_exptime": 900,
        "min_mag": 10.0,
        "max_mag": 30.0,
        "target_input": ("targets.csv", "text/csv", b"targets"),
        "pointing_input": ("pointings.csv", b"pointings"),
    }

    assert run_config_matches(completed_config, completed_config.copy())

    for key, changed_value in {
        "observation_type": "queue",
        "date_begin": "2026-02-02",
        "date_end": "2026-08-01",
        "single_exptime": 1200,
        "min_mag": 11.0,
        "max_mag": 31.0,
        "target_input": ("other.csv", "text/csv", b"targets"),
        "pointing_input": ("other-pointings.csv", b"pointings"),
    }.items():
        current_config = completed_config | {key: changed_value}
        assert not run_config_matches(completed_config, current_config)


def test_submit_warning_style_remains_enabled_and_can_be_cleared():
    submit_button = SubmitButtonWidgets()

    submit_button.set_warning(True)

    assert submit_button.submit.disabled is False
    assert submit_button.submit.stylesheets == [stylesheet_warning]

    submit_button.set_warning(False)

    assert submit_button.submit.disabled is False
    assert submit_button.submit.stylesheets == [stylesheet]
