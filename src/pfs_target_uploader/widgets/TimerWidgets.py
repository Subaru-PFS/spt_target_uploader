#!/usr/bin/env python3

import panel as pn
import asyncio


class TimerWidgets:
    def __init__(self, total_seconds=15 * 60):
        self.total_seconds = total_seconds
        self.loading_spinner = pn.indicators.LoadingSpinner(
            value=False, size=40, margin=(10, 0, 0, -10), color="secondary"
        )
        self.countdown_text = pn.pane.Markdown(
            "**00:00**", width=80, margin=(20, 0, 0, -10)
        )

        self.pane = pn.Row(
            self.loading_spinner,
            self.countdown_text,
            width=90,
            height=50,
        )
        self._stop_flag = asyncio.Event()
        self._timer_task = None

    async def _run_countdown(self):
        self.loading_spinner.value = True
        for seconds_left in range(0, self.total_seconds, 1):
            if self._stop_flag.is_set():
                self.loading_spinner.value = False
                return
            mins, secs = divmod(seconds_left, 60)
            self.countdown_text.object = f"**{mins:02d}:{secs:02d}**"
            await asyncio.sleep(1)
        self.loading_spinner.value = False
        self.countdown_text.object = "**Time's up!**"

    def timer(self, on=False):
        if on:
            # Stop any previous task
            if self._timer_task is not None and not self._timer_task.done():
                self._stop_flag.set()
            self._stop_flag = asyncio.Event()  # Reset Event (new instance)
            self.countdown_text.object = f"**00:00**"  # Reset display
            self._timer_task = asyncio.create_task(self._run_countdown())
        else:
            if self._timer_task is not None and not self._timer_task.done():
                self._stop_flag.set()
            self.loading_spinner.value = False
