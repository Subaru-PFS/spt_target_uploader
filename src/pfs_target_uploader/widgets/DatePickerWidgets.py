#!/usr/bin/env python3

from datetime import datetime
from zoneinfo import ZoneInfo

import panel as pn
import param

from ..utils.checker import get_semester_daterange


class DatePickerWidgets(param.Parameterized):
    def __init__(self):
        today = datetime.now(ZoneInfo("US/Hawaii"))

        semester_begin, semester_end = get_semester_daterange(
            today.date(),
            next=True,
        )

        self.date_begin = pn.widgets.DatePicker(
            name="Date Begin (HST)", value=semester_begin.date()
        )
        self.date_end = pn.widgets.DatePicker(
            name="Date End (HST)", value=semester_end.date()
        )

        self.pane = pn.Column("### Observation Period", self.date_begin, self.date_end)
