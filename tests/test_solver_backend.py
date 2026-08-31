"""Tests for selecting the ILP solver backend used by the pointing simulation.

The backend is named in three places -- SOLVER_BACKEND in .env.shared, --solver
on the CLI, and the solver_backend argument of PPPrunStart -- and each one has
to reject a name the next one downstream would not understand.
"""

import numpy as np
import pandas as pd
import pytest

from pfs_target_uploader.cli.cli_main import SolverBackend
from pfs_target_uploader.utils.config import SOLVER_BACKENDS, load_app_config


def _write_env(tmp_path, **settings):
    """Write a minimal .env.shared and return its path."""
    lines = [f"OUTPUT_DIR={tmp_path}"]
    lines += [f"{key}={value}" for key, value in settings.items()]
    env_file = tmp_path / ".env.shared"
    env_file.write_text("\n".join(lines) + "\n")
    return str(env_file)


def _load(tmp_path, **settings):
    return load_app_config(
        _write_env(tmp_path, **settings),
        create_output_dir=False,
        validate_db=False,
    )


def test_cli_enum_matches_the_shared_backend_list():
    """The typer Enum is spelled out by hand; keep it honest."""
    assert tuple(member.value for member in SolverBackend) == SOLVER_BACKENDS


@pytest.mark.parametrize("written", ["highs", "HiGHS", "HIGHS", "  highs  "])
def test_solver_backend_is_read_case_and_space_insensitively(tmp_path, written):
    """A site without a Gurobi licence must not be sent to Gurobi over casing."""
    assert _load(tmp_path, SOLVER_BACKEND=written).solver_backend == "highs"


def test_unknown_solver_backend_falls_back_to_gurobi(tmp_path):
    assert _load(tmp_path, SOLVER_BACKEND="bogus").solver_backend == "gurobi"


def test_absent_solver_backend_defaults_to_gurobi(tmp_path):
    assert _load(tmp_path).solver_backend == "gurobi"


def test_empty_solver_backend_defaults_to_gurobi(tmp_path):
    """dotenv yields "" for `SOLVER_BACKEND=`; it must not crash the parse."""
    assert _load(tmp_path, SOLVER_BACKEND="").solver_backend == "gurobi"


def test_ppprunstart_rejects_an_unknown_backend():
    """The guard sits at the entry, before the minutes of clustering."""
    from pfs_target_uploader.utils.ppp import PPPrunStart

    with pytest.raises(ValueError, match="Unknown solver_backend"):
        PPPrunStart(None, None, None, solver_backend="bogus")


def test_ppprunstart_refuses_highs_when_netflow_lacks_it(monkeypatch):
    """ets-fiber-assigner reports 3.8.0 with or without the HiGHS backend, so
    PPPrunStart probes for the class rather than trusting the version."""
    from pfs_target_uploader.utils import ppp

    class _NetflowWithoutHighs:
        pass

    monkeypatch.setattr(ppp, "nf", _NetflowWithoutHighs())

    with pytest.raises(RuntimeError, match="no HiGHS backend"):
        ppp.PPPrunStart(None, None, None, solver_backend="highs")


def test_ppprunstart_accepts_highs_when_netflow_has_it():
    """Guard must not fire against the pinned ets-fiber-assigner."""
    from pfs_target_uploader.utils import ppp

    assert hasattr(ppp.nf, "HighsProblem")

    # Gets past both guards and fails later on the None inputs instead.
    with pytest.raises(Exception) as excinfo:
        ppp.PPPrunStart(None, None, None, solver_backend="highs")
    assert not isinstance(excinfo.value, (ValueError, RuntimeError))


def test_run_ppp_reports_failure_when_the_child_produces_nothing(monkeypatch):
    """A child that dies before queueing anything must not be unpacked.

    PPPrunStart rejecting a bad backend kills the child immediately, which is
    the same empty-queue state a crash leaves behind. run_ppp has to report it
    the way ppp_result() reports its own failures -- nppc None -- because that
    is what show_results() and pn_app read as "simulation failed".
    """
    import panel as pn
    from loguru import logger

    from pfs_target_uploader.widgets.PppResultWidgets import PppResultWidgets

    class _Notifications:
        def __init__(self):
            self.errors = []

        def error(self, message, **kwargs):
            self.errors.append(message)

        def info(self, message, **kwargs):
            pass

    notifications = _Notifications()
    monkeypatch.setattr(
        type(pn.state), "notifications", property(lambda self: notifications)
    )

    df = pd.DataFrame(
        {
            "obj_id": [1, 2],
            "ob_code": ["a", "b"],
            "ra": [150.0, 150.1],
            "dec": [2.0, 2.1],
            "exptime": [900.0, 900.0],
            "priority": [1, 1],
            "resolution": ["M", "M"],
        }
    )
    validation_status = {
        "visibility": {"success": np.array([True, True])},
        "status": True,
    }

    widget = PppResultWidgets()
    widget.run_ppp(
        df,
        pd.DataFrame(),
        validation_status,
        single_exptime=900,
        logger=logger,
        max_exetime=120,
        solver_backend="bogus",
    )

    assert widget.nppc is None
    assert widget.p_result_fig is None
    assert widget.p_result_ppc is None
    assert widget.p_result_tab is None
    # 2 renders as "no fiber assigned"; 1 is reserved for a timeout.
    assert widget.status_ == 2
    assert notifications.errors


def test_reset_discards_all_previous_ppp_outputs():
    """A new validation must not package products from a discarded run."""
    from pfs_target_uploader.widgets.PppResultWidgets import PppResultWidgets

    widget = PppResultWidgets()
    widget.nppc = object()
    widget.p_result_fig = object()
    widget.p_result_ppc = object()
    widget.p_result_tab = object()
    widget.single_exptime = 1800

    widget.reset()

    assert widget.nppc is None
    assert widget.p_result_fig is None
    assert widget.p_result_ppc is None
    assert widget.p_result_tab is None
    assert widget.single_exptime == 900


def test_terminate_active_ppp_stops_the_registered_process_once(monkeypatch):
    import importlib

    from pfs_target_uploader.widgets import PppResultWidgets

    ppp_widgets = importlib.import_module(
        "pfs_target_uploader.widgets.PppResultWidgets"
    )

    widget = PppResultWidgets()
    active_process = object()
    terminated = []
    widget._ppp_process = active_process
    monkeypatch.setattr(
        ppp_widgets,
        "terminate_process_group",
        lambda process: terminated.append(process),
    )

    widget.terminate_active_ppp()
    widget.terminate_active_ppp()

    assert terminated == [active_process]
    assert widget._ppp_process is None


def test_server_cleanup_terminates_each_active_ppp_widget_once(monkeypatch):
    from pfs_target_uploader import pn_app

    class _PppWidget:
        def __init__(self):
            self.terminations = 0

        def terminate_active_ppp(self):
            self.terminations += 1

    first = _PppWidget()
    second = _PppWidget()
    monkeypatch.setattr(pn_app, "_active_ppp_widgets", {first, second})

    pn_app.terminate_active_ppp_runs()

    assert first.terminations == 1
    assert second.terminations == 1


def test_start_app_cleans_active_ppp_when_panel_server_stops(monkeypatch):
    import panel as pn

    from pfs_target_uploader import pn_app
    from pfs_target_uploader.cli import cli_main

    cleanup_calls = []
    monkeypatch.setattr(pn, "serve", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main.logger, "remove", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main.logger, "add", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pn_app,
        "terminate_active_ppp_runs",
        lambda: cleanup_calls.append(True),
    )

    cli_main.start_app(cli_main.PanelAppName.uploader, port=0)

    assert cleanup_calls == [True]
