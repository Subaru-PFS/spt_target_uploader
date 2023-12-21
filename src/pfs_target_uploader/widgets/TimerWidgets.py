#!/usr/bin/env python3

import time

import panel as pn


class TimerWidgets:
    def __init__(self):
        self.loading_spinner = pn.indicators.LoadingSpinner(
            value=False, size=40, margin=(10, 0, 0, -10), color="secondary"
        )

        # self.timer_init = pn.pane.Markdown("00:00")

        self.pane = pn.Column(self.loading_spinner, width=40, height=40)

    def timer(self, on=False):
        if on is True:
            """
            x = 0
            while x < 20:#15 * 60:
                self.loading_spinner.value = True
                mins, secs = divmod(x, 60)
                self.timer_init.object = '{:02d}:{:02d}'.format(mins, secs)
                x += 1
                time.sleep(1)
            #"""
            self.loading_spinner.value = True
        elif on is False:
            self.loading_spinner.value = False
