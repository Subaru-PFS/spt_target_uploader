import panel as pn
import param


class ObsTypeWidgets(param.Parameterized):
    stylesheet = """
        .bk-btn-primary {
            border-color: #3A7D7E !important;
            // border-color: #d2e7de !important;
        }

        .bk-btn-primary:hover, .bk-btn-primary.bk-active {
            color: #ffffff !important;
            background-color: #008899 !important;
        }
        """

    def __init__(self):

        def _activate_exptime(event):
            if self.obs_type.value == "classical":
                self.single_exptime.disabled = False
            else:
                self.single_exptime.disabled = True
                self.single_exptime.value = 900

        self.single_exptime = pn.widgets.IntInput(
            # name="Individual Exposure Time (s) in [10, 7200]",
            value=900,
            step=10,
            start=10,
            end=7200,
            disabled=True,
        )

        # self.obs_type = pn.widgets.RadioButtonGroup(
        #     options={"Queue": "queue", "Classical": "classical", "Filler": "filler"},
        #     value="queue",
        #     button_style="outline",
        #     button_type="primary",
        #     stylesheets=[self.stylesheet],
        # )
        self.obs_type = pn.widgets.Select(
            options={"Queue": "queue", "Classical": "classical", "Filler": "filler"},
            value="Queue",
        )

        self.obstype_pane = pn.Column(self.obs_type)

        i_activate_exptime = pn.bind(_activate_exptime, self.obs_type)

        self.exptime_pane = pn.Column(
            "### Individual Exposure Time (s)",
            self.single_exptime,
            i_activate_exptime,
        )
