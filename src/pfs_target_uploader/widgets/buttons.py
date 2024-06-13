#!/usr/bin/env python3

import panel as pn

stylesheet = """
    .bk-btn-primary {
        // background-color: #eaf4fc   !important;
        background-color: #C7E2D6 !important;
        border-color: #008899;
        border-width: 2px;
    }

    .bk-btn-primary:hover {
        color: #ffffff !important;
        background-color: #008899 !important;
        border-color: #008899;
    }

    .bk-btn-primary:disabled {
        border-color: #d2e7de !important;
        background-color: #ffffff !important;
        border-width: 1px;
    }
    """

stylesheet_warning = """
    .bk-btn-primary {
        background-color: #fdf3d1 !important;
        border-color: #866109;
        border-width: 2px;
    }

    .bk-btn-primary:hover {
        color: #ffffff !important;
        background-color: #b8860b  !important;
        border-color: #866109;
    }

    .bk-btn-primary:disabled {
        border-color: #d2e7de !important;
        background-color: #ffffff !important;
        border-width: 1px;
    }
    """


class ValidateButtonWidgets:
    def __init__(self):
        self.validate = pn.widgets.Button(
            name="Validate",
            button_style="outline",
            button_type="primary",
            icon="stethoscope",
            height=60,
            max_width=130,
            stylesheets=[stylesheet],
        )
        self.pane = self.validate


class RunPppButtonWidgets:
    def __init__(self):
        self.PPPrun = pn.widgets.Button(
            name="Simulate",
            button_style="outline",
            button_type="primary",
            icon="player-play-filled",
            height=60,
            max_width=130,
            stylesheets=[stylesheet],
        )

        self.pane = self.PPPrun


class SubmitButtonWidgets:
    def __init__(self):
        self.submit = pn.widgets.Button(
            name="Submit",
            button_style="outline",
            button_type="primary",
            icon="send",
            disabled=True,
            height=60,
            max_width=130,
            stylesheets=[stylesheet],
        )
        self.pane = self.submit

    def enable_button(self, ppp_status):
        if ppp_status:
            self.submit.stylesheets = []
            self.submit.stylesheets = [stylesheet]
            self.submit.disabled = False
        else:
            self.submit.stylesheets = []
            self.submit.stylesheets = [stylesheet_warning]
            self.submit.disabled = False
