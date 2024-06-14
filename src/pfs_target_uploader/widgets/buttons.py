#!/usr/bin/env python3

import panel as pn

stylesheet = """
    .bk-btn {
        color: var(--success-text-color) !important;
        background-color: #C7E2D6 !important;
        border-color: var(--success-border-subtle) !important;
        border-width: 1px;
        // color: #145B33;
        // background-color: #eaf4fc;
        // background-color: var(--success-bg-color);
        // border-color: #008899;
    }

    .bk-btn:hover {
        color: #ffffff !important;
        background-color: #008899 !important;
        // border-color: #008899;
        // background-color: var(--success-text-color) !important;
        border-color: var(--success-border-subtle) !important;
    }

    .bk-btn:disabled {
        color: var(--secondary-text-color) !important;
        border-color: #d2e7de !important;
        background-color: #ffffff !important;
        border-width: 1px;
    }
    """

stylesheet_warning = """
    .bk-btn {
        color: var(--warning-text-color) !important;
        background-color: #fdf3d1 !important;
        border-color: var(--warning-border-subtle);
        border-width: 1px;
        // border-color: #866109;
    }

    .bk-btn:hover {
        color: #ffffff !important;
        background-color: #b8860b  !important;
        border-color: var(--warning-border-subtle);
        // border-color: #866109;
    }

    .bk-btn:disabled {
        color: var(--secondary-text-color) !important;
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
