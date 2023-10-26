#!/usr/bin/env python3

import panel as pn


class DocLinkWidgets:
    def __init__(self):
        self.doc = pn.pane.Markdown(
            "<font size='4'><i class='fa-solid fa-circle-info fa-lg' style='color: #3A7D7E;'></i> <a href='doc/index.html' target='_blank'>Manual</a></font>",
            # "<font size='4'><i class='fa-solid fa-circle-info fa-lg' style='color: #3A7D7E;'></i> <a href='doc/index.html' target='_blank'>Documentation</a></font>",
            # styles={"text-align": "right"},
        )
        self.pane = pn.Column(self.doc)
