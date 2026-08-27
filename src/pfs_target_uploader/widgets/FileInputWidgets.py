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
        # validated frame for a large list runs to hundreds of megabytes, and
        # for the same reason the raw input frame is deliberately not kept
        # alongside it -- see validate()'s return.
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
        self.invalidate_cache()

    def invalidate_cache(self):
        """Drop the memoized validation result.

        Callers outside this class need this after anything that makes the
        entry unreachable -- cb_submit assigning a fresh upload ID, for one,
        which changes the key so the old entry could never be hit again while
        still pinning its frames for the life of the session.
        """
        self._cached_key = None
        self._cached_result = None
        self.last_validation_key = None

    def _cache_key(
        self,
        filename,
        mime_type,
        value,
        date_begin,
        date_end,
        single_exptime,
        min_mag,
        max_mag,
    ):
        """Identity of everything validate() reads.

        The file's identity is passed in rather than read from the widget:
        validate() runs in a worker thread while the browser can still deliver
        a new upload, and reading the widget again at the end would file this
        run's result under the *next* file's key -- a wrong result that would
        then stick for the rest of the session.

        The raw bytes go in rather than a digest of them. Comparing the key
        tuples goes through PyObject_RichCompareBool, which short-circuits on
        identity, and Panel hands back the same bytes object until the user
        uploads again -- so the usual comparison is free, against 17 ms to
        SHA-256 a 40 MB list. A distinct object with the same content still
        costs only a memcmp. No extra memory either: previous_value below
        already holds a reference to these bytes, and the surrounding code
        compares them the same way.

        secret_token is read from the instance, not passed in, because it is
        the one component that legitimately changes mid-call: the block below
        reassigns it for a new file. It is part of the key because validate()
        stamps its first 7 characters onto every ob_code, and cb_submit
        assigns a fresh token after an upload -- without it, a re-validation
        after a submit would hand back ob_codes carrying the previous upload's
        ID. It is here to be compared, not generated; the upload ID itself
        comes from assign_secret_token() and is untouched by the cache.
        """
        return (
            filename,
            mime_type,
            value,
            self.secret_token,
            date_begin,
            date_end,
            single_exptime,
            min_mag,
            max_mag,
        )

    def _cached_for(self, key):
        """The stored result for this key, or None if there is not one.

        The single definition of "this key is already validated". It hands
        back the entry rather than a bool so that the caller unpacking it does
        not have to re-establish that it is there -- the check and the use
        stay in one place instead of a reader (or a type checker) having to
        prove the two agree.
        """
        if self._cached_result is None or key != self._cached_key:
            return None
        return self._cached_result

    def has_cached_validation(
        self,
        date_begin=None,
        date_end=None,
        single_exptime=900.0,
        min_mag=None,
        max_mag=None,
    ):
        """True when validate() with these arguments would skip the real work.

        Callers use this to decide whether to blank the panels first. It shares
        _cached_for() with validate() so the two cannot drift; note that it
        does not model validate()'s date-range guard, which sits ahead of the
        cache check. That is harmless only because a matching key implies the
        cached run's dates, which already satisfied begin < end. Any *new*
        early return added ahead of the cache check has to be mirrored here.
        """
        return (
            self._cached_for(
                self._cache_key(
                    self.file_input.filename,
                    self.file_input.mime_type,
                    self.file_input.value,
                    date_begin,
                    date_end,
                    single_exptime,
                    min_mag,
                    max_mag,
                )
            )
            is not None
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
            self.last_validation_key = None
            return None, None

        # Snapshot the file's identity once, up front. Everything below runs
        # in a worker thread (asyncio.to_thread) and a large list takes tens of
        # seconds, during which the browser can still deliver a new upload.
        # Re-reading the widget later would mix two files within one call.
        filename = self.file_input.filename
        mime_type = self.file_input.mime_type
        value = self.file_input.value

        def cache_key():
            return self._cache_key(
                filename,
                mime_type,
                value,
                date_begin,
                date_end,
                single_exptime,
                min_mag,
                max_mag,
            )

        # cb_validate, cb_PPP and cb_submit all validate the same inputs in
        # turn; only the first of them has to do the work.
        cached = self._cached_for(cache_key())
        if cached is not None:
            logger.info(
                "Inputs are unchanged since the last validation; reusing the result."
            )
            logger.info(f"Upload ID: {self.secret_token}")
            self.last_validation_key = cache_key()
            validation_status, df_output = cached
            # Re-raised rather than assumed still on screen: every callback
            # clears the notifications before calling us, so the copy shown
            # after the first validation is gone by the time the user starts
            # the multi-minute simulation this warns about. Counted on the
            # validated frame because the raw one is not kept; validate_input
            # adds and drops columns but never rows.
            if df_output.index.size >= warn_threshold:
                pn.state.notifications.info(
                    "The number of objects is very large. "
                    "It may take a long time to process.",
                    duration=0,
                )
            return copy.deepcopy(validation_status), df_output.copy(deep=True)

        # update the upload ID when the input file is different from previous validation.
        if (
            (filename != self.previous_filename)
            or (value != self.previous_value)
            or (mime_type != self.previous_mime_type)
        ):

            self.secret_token = assign_secret_token(
                db_path=self.db_path, output_dir=self.output_dir, use_db=self.use_db
            )

            logger.info("New file detected.")
            logger.info(f"    Upload ID updated: {self.secret_token}")
            logger.info(f"    Filename: {filename}")
            logger.info(f"    MIME Type: {mime_type}")

            self.previous_filename = filename
            self.previous_value = value
            self.previous_mime_type = mime_type
        else:
            logger.info("Identical to the previous validation.")
            logger.info(
                "    Upload ID not updated: one or more of the filename, content, "
                "and mime type are identical to the previous validation."
            )

        logger.info(f"Upload ID: {self.secret_token}")

        if filename is not None:
            logger.info(f"{filename} is selected.")
            file_format = os.path.splitext(filename)[-1].replace(".", "")
            df_input, dict_load = load_input(
                BytesIO(value),
                format=file_format,
            )
            # if the input file cannot be read, raise a sticky error notifications
            if not dict_load["status"]:
                pn.state.notifications.error(
                    f"Cannot load the input file. Please check the content. Error: {dict_load['error']}",
                    duration=0,
                )
                self.last_validation_key = None
                return None, None

            if df_input.index.size >= warn_threshold:
                pn.state.notifications.info(
                    "The number of objects is very large. It may take a long time to process.",
                    duration=0,
                )
        else:
            logger.info("No file selected.")
            pn.state.notifications.error("Please select a CSV file.")
            self.last_validation_key = None
            return None, None

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
            self.last_validation_key = None
            return None, None

        # Recomputed rather than reusing the key from the top: a new file
        # reassigns secret_token above, which is part of the key. The file's
        # identity still comes from the snapshot taken before the work began.
        self._cached_key = cache_key()
        # df_input is deliberately left out. No caller reads the frame as it
        # came off the CSV, and keeping it would pin a second frame of
        # comparable size -- 49 MB on a 163,000-row list -- for the life of
        # the session, plus a discarded deep copy on every call.
        self._cached_result = (validation_status, df_output)
        # Every failure path above sets this to None, so it always describes
        # the most recent call: a key on success, None on failure. The render
        # gate in pn_app reads it and depends on that being true.
        self.last_validation_key = self._cached_key

        # Callers get a private copy in both paths: run_ppp() writes
        # single_exptime into the frame it is handed.
        return copy.deepcopy(validation_status), df_output.copy(deep=True)
