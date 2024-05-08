#!/usr/bin/env python3

import pandas as pd
import panel as pn


class TargetWidgets:
    stylesheet = """
    .tabulator-row-even { background-color: #f9f9f9 !important; }
    .tabulator-row-odd:hover { color: #000000 !important; }
    .tabulator-row-even:hover { color: #000000 !important; }
    """

    def __init__(self):
        self.table_all = pn.widgets.Tabulator(
            None,
            page_size=50,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=[],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
            disabled=True,
            stylesheets=[self.stylesheet],
            titles={"obj_id_str": "obj_id"},
            hidden_columns=["obj_id"],
        )

        self.pane = pn.Column(self.table_all)

    def show_results(self, df):
        # it seems that frozen_columns must be empty when replacing its value
        self.table_all.frozen_columns = []
        if self.table_all.value is not None:
            self.table_all.value[0:0]
        self.table_all.value = df
        self.table_all.frozen_columns = ["index"]
        self.table_all.visible = True

    def reset(self):
        self.table_all.frozen_columns = []
        if self.table_all.value is not None:
            self.table_all.value[0:0]
        self.table_all.visible = False
