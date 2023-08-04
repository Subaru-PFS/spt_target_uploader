#!/usr/bin/env python3

import panel as pn
from pfs_target_uploader.pn_app import list_files_app

pn.extension(
    "floatpanel",
    "mathjax",
    "tabulator",
    notifications=True,
    sizing_mode="stretch_width",
    # sizing_mode="scale_width",
    js_files={
        "font-awesome": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js",
        # "bootstrap": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css",
    },
    css_files=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
        # "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js",
        # "https://fonts.googleapis.com/css?family=Lato&subset=latin,latin-ext",
    ],
    layout_compatibility="error",
)

pn.state.notifications.position = "bottom-left"

list_files_app()
