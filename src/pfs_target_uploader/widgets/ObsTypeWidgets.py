import panel as pn
import param


class ObsTypeWidgets(param.Parameterized):
    stylesheet = """
        .bk-btn-primary {
            border-color: #3A7D7E !important;
            // border-color: #d2e7de !important;
        }

        .bk-btn-primary.bk-active {
            color: #ffffff !important;
            background-color: #008899 !important;
        }

        .bk-btn-primary:hover {
            background-color: #008899 !important;
            opacity: 0.8; !important;
        }
        """

    stylesheet_radiobox = """
        .bk-input-group span {
            font-size: 1.25em;
            font-weight: 500;
            vertical-align: middle !important;
            margin-left: 0.25em;
            margin-right: 1.25em;
        }

        .bk-input-group label {
            min-width: 100px !important;
            max-width: 33% !important;
        }

        input[type='radio'] {
            accent-color: #3A7D7E !important;
        }
        """

    def __init__(self):
        # single exposure time widget
        self.single_exptime = pn.widgets.IntInput(
            # name="Individual Exposure Time (s) in [10, 7200]",
            value=900,
            step=10,
            start=10,
            end=7200,
            disabled=True,
        )

        #
        # observation type widget
        #
        # self.obs_type = pn.widgets.RadioButtonGroup(
        #     options={"Queue": "queue", "Classical": "classical", "Filler": "filler"},
        #     value="queue",
        #     button_style="outline",
        #     button_type="primary",
        #     stylesheets=[self.stylesheet],
        # )
        self.obs_type = pn.widgets.RadioBoxGroup(
            options={"Queue": "queue", "Classical": "classical", "Filler": "filler"},
            value="queue",
            inline=True,
            stylesheets=[self.stylesheet_radiobox],
        )

        self.obstype_pane = pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    "<font size=4><i class='fas fa-binoculars'></i> **Observation type**</font>",
                    width=400,
                ),
                pn.widgets.TooltipIcon(
                    value="(Optional for Classical) Set **individual exposure time** and **pointing centers** in the **Config** tab.",
                    margin=(0, 0, 0, -230),
                ),
            ),
            self.obs_type,
        )

        self.exptime_pane = pn.Column(
            "<font size=3><i class='far fa-clock'></i> **Individual exposure time (s)**</font>",
            self.single_exptime,
        )
