#!/usr/bin/env python3

import copy
import os
import secrets
import time
from io import BytesIO

import panel as pn
import param
from loguru import logger

from ..utils.checker import validate_input
from ..utils.io import load_input
from ..utils.session import assign_secret_token


class FileInputWidgets(param.Parameterized):
    def __init__(self):
        self.file_input = pn.widgets.FileInput(
            value=None,
            filename=None,
            accept=".csv,.ecsv",
            multiple=False,
            height=40,
        )

        # store previous information for input comparison
        self.previous_filename = None
        self.previous_value = None
        self.previous_mime_type = None

        # hex string to be used as an upload ID
        self.secret_token = None
        self.db_path = None
        self.output_dir = None
        self.use_db = True

        # Last validation result, kept so cb_PPP and cb_submit do not redo the
        # full validation cb_validate already paid for. One entry only: a
        # validated frame for a large list runs to hundreds of megabytes.
        self._cached_key = None
        self._cached_result = None
        self.last_validation_key = None

        self.pane = pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    "<font size=4><i class='fas fa-list-ul'></i>  **Target list**</font> "
                    "<font size=4>(CSV; <a href='doc/examples/example_perseus_cluster_r60arcmin.csv' target='_blank'>example</a>)</font>",
                    # styles={"margin-bottom": "-10px"},
                    # styles={
                    #     "border-left": "10px solid #3A7D7E",
                    #     "border-bottom": "1px solid #3A7D7E",
                    #     "padding-left": "0.5em",
                    # },
                    width=400,
                    # width=370,
                ),
                pn.widgets.TooltipIcon(
                    value="(Optional) Configure the **observation period** in the **Config** tab.",
                    # width=50,
                    margin=(0, 0, 0, -165),
                    # align=("start", "center"),
                ),
            ),
            self.file_input,
            # margin=(10, 0, -10, 0),
        )

    def reset(self):
        self.file_input.filename = None
        self.file_input.mime_type = None
        self.file_input.value = None
        self._cached_key = None
        self._cached_result = None
        self.last_validation_key = None

    def _cache_key(self, date_begin, date_end, single_exptime, min_mag, max_mag):
        """Identity of everything validate() reads.

        The raw bytes go in rather than a digest of them. Comparing the key
        tuples goes through PyObject_RichCompareBool, which short-circuits on
        identity, and Panel hands back the same bytes object until the user
        uploads again -- so the usual comparison is free, against 17 ms to
        SHA-256 a 40 MB list. A distinct object with the same content still
        costs only a memcmp. No extra memory either: previous_value below
        already holds a reference to these bytes, and the surrounding code
        compares them the same way.

        secret_token is part of the key because validate() stamps its first 7
        characters onto every ob_code, and cb_submit assigns a fresh token
        after an upload. Without it, a re-validation after a submit would hand
        back ob_codes carrying the previous upload's ID. It is here to be
        compared, not generated -- the upload ID itself comes from
        assign_secret_token() and is untouched by the cache.
        """
        return (
            self.file_input.filename,
            self.file_input.mime_type,
            self.file_input.value,
            self.secret_token,
            date_begin,
            date_end,
            single_exptime,
            min_mag,
            max_mag,
        )

    def has_cached_validation(
        self,
        date_begin=None,
        date_end=None,
        single_exptime=900.0,
        min_mag=None,
        max_mag=None,
    ):
        """True when validate() with these arguments would skip the real work."""
        if self._cached_result is None:
            return False
        return self._cached_key == self._cache_key(
            date_begin, date_end, single_exptime, min_mag, max_mag
        )

    def validate(
        self,
        date_begin=None,
        date_end=None,
        single_exptime=900.0,
        min_mag=None,
        max_mag=None,
        warn_threshold=100000,
    ):
        t_start = time.time()
        if date_begin >= date_end:
            pn.state.notifications.error(
                "Date Begin must be before Date End.", duration=0
            )
            return None, None, None

        # cb_validate, cb_PPP and cb_submit all validate the same inputs in
        # turn; only the first of them has to do the work. The "very large
        # list" notification below is skipped along with it, which is fine --
        # the user has already been shown it once for this file.
        cache_key = self._cache_key(
            date_begin, date_end, single_exptime, min_mag, max_mag
        )
        if self._cached_result is not None and cache_key == self._cached_key:
            logger.info(
                "Inputs are unchanged since the last validation; reusing the result."
            )
            logger.info(f"Upload ID: {self.secret_token}")
            self.last_validation_key = cache_key
            validation_status, df_input, df_output = self._cached_result
            return (
                copy.deepcopy(validation_status),
                df_input.copy(deep=True),
                df_output.copy(deep=True),
            )

        # update the upload ID when the input file is different from previous validation.
        if (
            (self.file_input.filename != self.previous_filename)
            or (self.file_input.value != self.previous_value)
            or (self.file_input.mime_type != self.previous_mime_type)
        ):

            self.secret_token = assign_secret_token(
                db_path=self.db_path, output_dir=self.output_dir, use_db=self.use_db
            )

            logger.info("New file detected.")
            logger.info(f"    Upload ID updated: {self.secret_token}")
            logger.info(f"    Filename: {self.file_input.filename}")
            logger.info(f"    MIME Type: {self.file_input.mime_type}")

            self.previous_filename = self.file_input.filename
            self.previous_value = self.file_input.value
            self.previous_mime_type = self.file_input.mime_type
        else:
            logger.info("Identical to the previous validation.")
            logger.info(
                "    Upload ID not updated: one or more of the filename, content, "
                "and mime type are identical to the previous validation."
            )

        logger.info(f"Upload ID: {self.secret_token}")

        if self.file_input.filename is not None:
            logger.info(f"{self.file_input.filename} is selected.")
            file_format = os.path.splitext(self.file_input.filename)[-1].replace(
                ".", ""
            )
            df_input, dict_load = load_input(
                BytesIO(self.file_input.value),
                format=file_format,
            )
            # if the input file cannot be read, raise a sticky error notifications
            if not dict_load["status"]:
                pn.state.notifications.error(
                    f"Cannot load the input file. Please check the content. Error: {dict_load['error']}",
                    duration=0,
                )
                return None, None, None

            if df_input.index.size >= warn_threshold:
                pn.state.notifications.info(
                    "The number of objects is very large. It may take a long time to process.",
                    duration=0,
                )
        else:
            logger.info("No file selected.")
            pn.state.notifications.error("Please select a CSV file.")
            return None, None, None

        try:
            validation_status, df_output = validate_input(
                df_input.copy(deep=True),
                date_begin=date_begin,
                date_end=date_end,
                single_exptime=single_exptime,
                min_mag=min_mag,
                max_mag=max_mag,
            )
            t_stop = time.time()
            logger.info(f"Validation finished in {t_stop - t_start:.2f} [s]")

            # convert obj_id to string
            logger.debug(f"{validation_status=}")
            if validation_status["required_keys"]["status"]:
                df_output.insert(1, "obj_id_str", df_output["obj_id"].astype(str))

                # append first 7 characters of secret_token to ob_codes
                # `ob_code` is itself a required column, so this must stay
                # behind the same guard as `obj_id` above -- a list that is
                # missing `ob_code` would otherwise raise here and the
                # "Missing required columns" error panel would never render.
                df_output["ob_code"] = df_output["ob_code"].apply(
                    lambda x: f"{x}_{self.secret_token[:7]}"
                )
        except Exception as e:
            # A bug in validate_input() (or in how its output is consumed
            # here) must not leave the UI stuck with disabled widgets and no
            # feedback -- report it like the other failure paths above.
            logger.exception("Unexpected error during validation")
            pn.state.notifications.error(
                "An unexpected error occurred while validating the input file. "
                f"Please check the content of the file. Error: {e}",
                duration=0,
            )
            return None, None, None

        # Recomputed rather than reusing cache_key from the top: a new file
        # reassigns secret_token above, which is part of the key.
        self._cached_key = self._cache_key(
            date_begin, date_end, single_exptime, min_mag, max_mag
        )
        self._cached_result = (validation_status, df_input, df_output)
        self.last_validation_key = self._cached_key

        # Callers get a private copy in both paths: run_ppp() writes
        # single_exptime into the frame it is handed.
        return (
            copy.deepcopy(validation_status),
            df_input.copy(deep=True),
            df_output.copy(deep=True),
        )
