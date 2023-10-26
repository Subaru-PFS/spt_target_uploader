#!/usr/bin/env python3

import pandas as pd
import panel as pn
from logzero import logger


class TargetWidgets:
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
        )

        self.pane = pn.Column(self.table_all)

    def show_results(self, df):
        # it seems that frozen_columns must be empty when replacing its value
        self.table_all.frozen_columns = []
        # self.table_all.value = pd.DataFrame()
        self.table_all.value = df
        self.table_all.frozen_columns = ["index"]
        self.table_all.visible = True

    def reset(self):
        self.table_all.frozen_columns = []
        self.table_all.value = pd.DataFrame()
        self.table_all.visible = False
