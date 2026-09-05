"""Elapsed-time counter for validation and simulation operations."""

import panel as pn
import param


class ElapsedTimeDisplay(pn.reactive.ReactiveHTML):
    """Client-side elapsed-time readout for the Validate / Simulate operations.

    The counter is driven entirely in the browser from ``performance.now()``.
    The server only flips ``running`` and bumps ``run_id``; it never pushes a
    per-second update. That is deliberate (issue #500): validation runs in a
    worker thread that holds the GIL, so a server-driven readout stalls or
    drifts while the ``LoadingSpinner`` next to it -- a pure CSS animation once
    started -- keeps spinning. Moving the readout client-side makes it just as
    independent of the busy server.

    - ``run_id``: bump to (re)start. Restarting a timer that is already running
      is a bump with ``running`` left true -- several ``pn_app.py`` paths do
      exactly that.
    - ``running``: false stops the browser interval. The browser also sets it
      false itself when a time limit is reached, which is how the stop is
      handed back to Python (and, via ``TimerWidgets``, to the spinner).
    - ``limit_seconds``: stop and show "Time's up!" at this many seconds.
      ``0`` means unlimited.
    """

    run_id = param.Integer(default=0)
    running = param.Boolean(default=False)
    limit_seconds = param.Integer(default=0, bounds=(0, None))

    _template = '<div id="display" style="font-weight: bold">00:00</div>'

    _scripts = {
        "run_id": """
            if (state.interval !== undefined) {
                clearInterval(state.interval)
                state.interval = undefined
            }
            display.textContent = '00:00'
            if (!data.running) return

            const startedAt = performance.now()
            const update = () => {
                const elapsedSeconds = Math.floor((performance.now() - startedAt) / 1000)
                if (data.limit_seconds > 0 && elapsedSeconds >= data.limit_seconds) {
                    clearInterval(state.interval)
                    state.interval = undefined
                    display.textContent = "Time's up!"
                    data.running = false
                    return
                }
                const minutes = Math.floor(elapsedSeconds / 60)
                const seconds = elapsedSeconds % 60
                display.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
            }
            update()
            state.interval = setInterval(update, 250)
        """,
        "running": """
            if (!data.running && state.interval !== undefined) {
                clearInterval(state.interval)
                state.interval = undefined
            }
        """,
    }


class TimerWidgets:
    def __init__(self, total_seconds=15 * 60):
        self.total_seconds = total_seconds
        self.loading_spinner = pn.indicators.LoadingSpinner(
            value=False, size=40, margin=(10, 0, 0, -10), color="secondary"
        )
        # Panel 1.9 wraps ReactiveHTML in a `display: contents` element, so the
        # readout no longer gets the old wrapper's vertical extent and the text
        # rendered ~10px higher than on 1.8 -- level with the spinner's centre
        # instead of its lower-right. The larger top margin puts it back.
        self.elapsed_time = ElapsedTimeDisplay(
            width=80, height=30, margin=(30, 0, 0, -10)
        )
        # The spinner is on exactly while the readout is running. Driving it
        # from this one signal covers both stops: the server-side timer(on=False)
        # and the browser-side stop when limit_seconds is reached.
        self.elapsed_time.param.watch(self._sync_spinner, "running")

        self.pane = pn.Row(
            self.loading_spinner,
            self.elapsed_time,
            width=90,
            height=50,
        )

    def _sync_spinner(self, event):
        self.loading_spinner.value = event.new

    def timer(self, on=False, time_limit=True):
        if on:
            self.elapsed_time.limit_seconds = (
                max(self.total_seconds, 0) if time_limit else 0
            )
            self.elapsed_time.running = True
            self.elapsed_time.run_id += 1
        else:
            self.elapsed_time.running = False
