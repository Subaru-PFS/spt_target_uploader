#!/usr/bin/env python3

import panel as pn
from logzero import logger


class ValidateButtonWidgets:
    def __init__(self):
        self.validate = pn.widgets.Button(
            name="Validate",
            button_style="outline",
            button_type="primary",
            icon="stethoscope",
        )
        self.pane = pn.Column(
            """# Step 2:
## Check the uploaded list""",
            pn.Row(self.validate),
        )


class RunPppButtonWidgets:
    def __init__(self):
        self.PPPrun = pn.widgets.Button(
            name="Start (takes a few minutes ~ half hour)",
            button_style="outline",
            button_type="primary",
            icon="player-play-filled",
        )
        # loading spinner
        self.gif_pane = pn.pane.GIF(
            "https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif",
            width=20,
        )

        # placeholder for loading spinner
        self.PPPrunStats = pn.Column("", width=100)

        self.pane = pn.Column(
            """# Step 3:
## Estimate the total required time""",
            self.PPPrun,
            self.PPPrunStats,
        )

    def start(self):
        self.PPPrunStats.append(self.gif_pane)

    def stop(self):
        self.PPPrunStats.remove(self.gif_pane)


class SubmitButtonWidgets:
    def __init__(self):
        self.submit = pn.widgets.Button(
            name="Submit Results",
            button_type="primary",
            icon="send",
            disabled=True,
        )
        self.pane = pn.Column(
            """# Step 4:
## Submit results""",
            self.submit,
        )
