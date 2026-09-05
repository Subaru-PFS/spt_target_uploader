"""Sidebar link to the user guide."""

import panel as pn


class DocLinkWidgets:
    def __init__(self):
        self.doc = pn.pane.Markdown(
            "<font size='4'><i class='fa-solid fa-circle-info fa-lg' style='color: #3A7D7E;'></i> <a href='doc/index.html' target='_blank'>User Guide</a></font>",
        )
        self.pane = pn.Column(self.doc)
