"""Floating panel that shows the operator's announcement text on start-up."""

import panel as pn


class AnnouncementNoteWidgets:
    def __init__(self, ann_file=None):
        if ann_file is not None:

            with open(ann_file, "r") as f:
                message = f.read()

            self.floatpanel = pn.layout.FloatPanel(
                message,
                name="Important Announcements for Users",
                config={
                    "headerLogo": "<i style='margin-left: 0.5em;' class='fa-solid fa-circle-info fa-lg'></i>",
                },
                contained=False,
                position="center",
                # theme="#3A7D7E",
                theme="#DB2955",
                margin=20,
                width=720,
            )
