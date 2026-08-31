"""Contract tests for ``TimerWidgets`` / ``ElapsedTimeDisplay``.

The per-second counting now lives in the browser (``ElapsedTimeDisplay._scripts``,
driven by ``performance.now()``), so pytest cannot see the displayed string tick.
What these tests pin instead is the Python-side contract the browser depends on:

- which parameters ``timer()`` sets, and in which direction;
- that ``run_id`` -- not ``running`` -- is the restart signal, so a restart while
  already running still re-zeros the readout in the browser;
- that ``limit_seconds == 0`` means unlimited, including the ``total_seconds=0``
  ("MAX_EXETIME=0") case the original loop-based implementation got wrong
  (``range(0, 0)`` is empty -> instant "Time's up!");
- that the spinner tracks ``running`` from a single watcher, so the browser-side
  stop at the time limit (which writes ``running = False`` back to the server)
  also turns the spinner off.
"""

from pfs_target_uploader.widgets.TimerWidgets import TimerWidgets


def test_unlimited_start_sets_running_and_no_limit():
    timer = TimerWidgets()

    timer.timer(on=True, time_limit=False)

    assert timer.elapsed_time.running is True
    assert timer.elapsed_time.limit_seconds == 0
    assert timer.elapsed_time.run_id == 1
    assert timer.loading_spinner.value is True


def test_limited_start_passes_total_seconds_as_limit():
    timer = TimerWidgets(total_seconds=60)

    timer.timer(on=True, time_limit=True)

    assert timer.elapsed_time.limit_seconds == 60
    assert timer.elapsed_time.running is True


def test_zero_total_seconds_means_unlimited():
    # Regression guard: the old implementation looped over range(0, total_seconds),
    # so total_seconds=0 showed "Time's up!" immediately instead of counting on.
    timer = TimerWidgets(total_seconds=0)

    timer.timer(on=True, time_limit=True)

    assert timer.elapsed_time.limit_seconds == 0
    assert timer.elapsed_time.running is True


def test_restart_while_running_bumps_run_id():
    timer = TimerWidgets()

    timer.timer(on=True, time_limit=False)
    timer.timer(on=True, time_limit=False)

    assert timer.elapsed_time.run_id == 2
    assert timer.elapsed_time.running is True


def test_switching_time_limit_on_restart_updates_limit_before_run_id():
    timer = TimerWidgets(total_seconds=90)

    timer.timer(on=True, time_limit=False)
    assert timer.elapsed_time.limit_seconds == 0

    timer.timer(on=True, time_limit=True)
    assert timer.elapsed_time.limit_seconds == 90
    assert timer.elapsed_time.run_id == 2


def test_stop_clears_running_and_spinner():
    timer = TimerWidgets()

    timer.timer(on=True, time_limit=False)
    timer.timer(on=False)

    assert timer.elapsed_time.running is False
    assert timer.loading_spinner.value is False


def test_browser_side_stop_turns_spinner_off():
    # When the time limit is reached the browser writes running=False back to the
    # server; the spinner must follow it off through the same watcher.
    timer = TimerWidgets(total_seconds=60)

    timer.timer(on=True, time_limit=True)
    timer.elapsed_time.running = False

    assert timer.loading_spinner.value is False
