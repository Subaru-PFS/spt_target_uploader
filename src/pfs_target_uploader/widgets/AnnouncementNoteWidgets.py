#!/usr/bin/env python3

import os

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
                # theme="danger",
                # theme="#3A7D7E",
                theme="#DB2955",
                margin=20,
                # margin=100,
                width=720,
                # height=350,
            )
