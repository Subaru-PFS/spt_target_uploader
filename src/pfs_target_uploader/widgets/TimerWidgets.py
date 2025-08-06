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
        self._stop_flag_c = asyncio.Event()
        self._stop_flag_cd = asyncio.Event()
        self._timer_task_c = None
        self._timer_task_cd = None

    async def _run_countdown(self):
        await asyncio.sleep(1)
        self.loading_spinner.value = True
        for seconds_left in range(0, self.total_seconds, 1):
            if self._stop_flag_cd.is_set():
                self.loading_spinner.value = False
                return
            mins, secs = divmod(seconds_left, 60)
            self.countdown_text.object = f"**{mins:02d}:{secs:02d}**"
            await asyncio.sleep(1)
        self.loading_spinner.value = False
        self.countdown_text.object = "**Time's up!**"

    async def _run_count(self):
        self.loading_spinner.value = True
        sec_passing = 0
        while not self._stop_flag_c.is_set():
            mins, secs = divmod(sec_passing, 60)
            self.countdown_text.object = f"**{mins:02d}:{secs:02d}**"
            sec_passing += 1
            await asyncio.sleep(1)
        self.loading_spinner.value = False

    def timer(self, on=False, time_limit=True):
        if on:
            if time_limit:
                # Stop any previous task
                if self._timer_task_cd is not None and not self._timer_task_cd.done():
                    self._stop_flag_cd.set()
                self._stop_flag_cd.clear()  # Reset Event (clear existing)
                self.countdown_text.object = f"**00:00**"  # Reset display
                self._timer_task_cd = asyncio.create_task(self._run_countdown())
            else:
                # Stop any previous task
                if self._timer_task_c is not None and not self._timer_task_c.done():
                    self._stop_flag_c.set()
                self._stop_flag_c.clear()  # Reset Event (clear existing)
                self.countdown_text.object = f"**00:00**"  # Reset display
                self._timer_task_c = asyncio.create_task(self._run_count())
        else:
            if self._timer_task_c is not None and not self._timer_task_c.done():
                self._stop_flag_c.set()
            if self._timer_task_cd is not None and not self._timer_task_cd.done():
                self._stop_flag_cd.set()
            # self.loading_spinner.value = False
