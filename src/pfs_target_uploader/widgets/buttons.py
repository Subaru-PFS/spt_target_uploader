#!/usr/bin/env python3

import panel as pn
from logzero import logger
from param import descendents


class ValidateButtonWidgets:
    stylesheet = """
        .bk-btn-primary {
            border-color: #3A7D7E !important;
            // border-color: #d2e7de !important;
        }

        .bk-btn-primary:hover {
            color: #ffffff !important;
            background-color: #008899 !important;
        }"""

    def __init__(self):
        self.validate = pn.widgets.Button(
            name="Validate",
            button_style="outline",
            button_type="primary",
            icon="stethoscope",
            height=60,
            max_width=130,
            stylesheets=[self.stylesheet],
        )
        self.pane = self.validate


class RunPppButtonWidgets:
    stylesheet = """
        .bk-btn-primary {
            border-color: #3A7D7E !important;
        }

        .bk-btn-primary:hover {
            color: #ffffff !important;
            background-color: #008899 !important;
        }"""

    def __init__(self):
        self.PPPrun = pn.widgets.Button(
            name="Simulate",
            button_style="outline",
            button_type="primary",
            icon="player-play-filled",
            height=60,
            max_width=130,
            stylesheets=[self.stylesheet],
        )
        # loading spinner
        self.gif_pane = pn.pane.GIF(
            "https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif",
            width=20,
        )

        # placeholder for loading spinner
        self.PPPrunStats = pn.Column("", width=100)

        self.pane = self.PPPrun

    def start(self):
        self.PPPrunStats.append(self.gif_pane)

    def stop(self):
        self.PPPrunStats.remove(self.gif_pane)


class SubmitButtonWidgets:
    stylesheet = """
        .bk-btn-primary {
            border-color: #3A7D7E !important;
        }

        .bk-btn-primary:hover {
            color: #ffffff !important;
            background-color: #008899 !important;
        }"""

    stylesheet_warning = """
        .bk-btn-primary {
            border-color: #3A7D7E !important;
        }

        .bk-btn-primary:hover {
            color: #000000 !important;
            // background-color: salmon !important;
            background-color: #fdf3d1 !important;
        }"""

    def __init__(self):
        self.submit = pn.widgets.Button(
            name="Submit",
            button_style="outline",
            button_type="primary",
            icon="send",
            disabled=True,
            height=60,
            max_width=130,
            stylesheets=[self.stylesheet],
        )
        self.pane = self.submit

    def enable_button(self, ppp_status):
        if ppp_status:
            self.submit.stylesheets = []
            self.submit.stylesheets = [self.stylesheet]
            self.submit.disabled = False
        else:
            self.submit.stylesheets = []
            self.submit.stylesheets = [self.stylesheet_warning]
            self.submit.disabled = False
