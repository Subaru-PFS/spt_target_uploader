#!/usr/bin/env python3

import os

import panel as pn
from logzero import logger


class UploadNoteWidgets:
    # TODO: perhaps I can refactor to make it simple...
    def __init__(self, secret_token, uploaded_time, ppp_status, outdir, outfile_zip):
        if ppp_status:
            self.floatpanel = pn.layout.FloatPanel(
                None,
                # pn.pane.Markdown(message),
                name="Info",
                # config={"headerLogo": "<i class='fa-regular fa-thumbs-up fa-lg'></i>"},
                contained=False,
                position="center",
                # theme="none",
                theme="#3A7D7E",
                margin=20,
                width=720,
                height=350,
            )

            # JS on-click actions
            # https://github.com/awesome-panel/awesome-panel/blob/master/examples/js_actions.py
            # so far not working...
            stylesheet = """
        :host {
            --font-size: 2.5em;
            --color: darkcyan;
        }
        .bk-btn-light {
            color: darkcyan;
        }
        """

            self.copy_source_button = pn.widgets.Button(
                name=f"{secret_token}",
                icon="copy",
                # width=500,
                height=96,
                icon_size="1.5em",
                # button_style="outline",
                button_type="light",
                stylesheets=[stylesheet],
            )

            copy_source_code = "navigator.clipboard.writeText(source);"

            self.copy_source_button.js_on_click(
                args={"source": f"{secret_token}"},
                code=copy_source_code,
            )

            messages = [
                pn.pane.Markdown(
                    "<i class='fa-regular fa-thumbs-up fa-2xl'></i><font size='4'>  Upload successful! Your **Upload ID** is the following. </font>"
                ),
                self.copy_source_button,
                pn.pane.Markdown(
                    f"<a href='{os.path.join(outdir, outfile_zip)}'><i class='fa-solid fa-download fa-2xl'></i><font size=4>  Download the results as a zip file</a></font>"
                ),
                pn.pane.Markdown(
                    f"<font size='4'>Uploaded at {uploaded_time.isoformat(timespec='seconds')}</font>"
                ),
                pn.pane.Markdown(
                    """
                - Please keep the Upload ID for the observation planning.
                - You can copy the Upload ID to the clipboard by clicking it.
                """
                ),
            ]

        elif not ppp_status:
            self.floatpanel = pn.layout.FloatPanel(
                None,
                # pn.pane.Markdown(message),
                name="Warning",
                # config={"headerLogo": "<i class='fa-regular fa-thumbs-up fa-lg'></i>"},
                contained=False,
                position="center",
                # theme="none",
                # theme="#FFF1C2",
                theme="#98741E fillcolor #FFF3D0",
                margin=20,
                width=720,
                # config={"theme": {"colorContent": "#866208"}},
            )

            # JS on-click actions
            # https://github.com/awesome-panel/awesome-panel/blob/master/examples/js_actions.py
            # so far not working...
            stylesheet = """
        :host {
            --font-size: 2.5em;
            --color: #98741E;
        }
        .bk-btn-light {
            color: #98741E;
            background-color: #FFF3D0;
        }
        .bk-btn-light:hover {
            color: #98741E;
            background-color: #fffacd;
        }
        """

            self.copy_source_button = pn.widgets.Button(
                name=f"{secret_token}",
                icon="copy",
                # width=500,
                height=96,
                icon_size="1.5em",
                # button_style="outline",
                button_type="light",
                stylesheets=[stylesheet],
            )

            copy_source_code = "navigator.clipboard.writeText(source);"

            self.copy_source_button.js_on_click(
                args={"source": f"{secret_token}"},
                code=copy_source_code,
            )

            messages = [
                pn.pane.Markdown(
                    "<i class='fa-solid fa-triangle-exclamation fa-2xl' style='color: #98741E;'></i><font size='4' color='#98741E'>  Upload successful **_WITHOUT_** pointing simulation! Your **Upload ID** is the following.</font>"
                ),
                self.copy_source_button,
                pn.pane.Markdown(
                    f"<a href='{os.path.join(outdir, outfile_zip)}'><i class='fa-solid fa-download fa-2xl'></i><font size=4>  Download the results as a zip file</a></font>"
                ),
                pn.pane.Markdown(
                    f"<font size='4' color='#98741E'>Uploaded at {uploaded_time.isoformat(timespec='seconds')}</font>"
                ),
                pn.pane.Markdown(
                    """
                - <font color='#98741E'>Please keep the Upload ID for the observation planning.</font>
                - <font color='#98741E'>You can copy the Upload ID to the clipboard by clicking it.</font>
                - <font color='#98741E'>**It is not recommended to submit a target list without a pointing simulation.**</font>
                """
                ),
            ]

        self.floatpanel.objects = []
        for m in messages:
            self.floatpanel.objects.append(m)
