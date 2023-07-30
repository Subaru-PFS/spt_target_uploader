#!/usr/bin/env python3

import panel as pn
from pfs_target_uploader.pn_app import target_uploader_app

pn.extension(
    "mathjax",
    "tabulator",
    notifications=True,
    sizing_mode="stretch_width",
    # sizing_mode="scale_width",
    # js_files={
    #     "font-awesome": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js"
    # },
    # css_files=[
    #     "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    #     # "https://fonts.googleapis.com/css?family=Lato&subset=latin,latin-ext",
    # ],
    layout_compatibility="error",
)

pn.state.notifications.position = "bottom-left"

target_uploader_app()
