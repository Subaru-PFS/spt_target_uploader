#!/usr/bin/env python3

import glob
import os
import secrets

from loguru import logger

from .db import upload_id_exists


def assign_secret_token(db_path=None, output_dir=None, use_db=True, nbytes=8):

    st = secrets.token_hex(nbytes)

    logger.info("Checking the uniqueness of the upload ID.")

    while True:
        if use_db:
            if upload_id_exists(st, db_path):
                logger.warning(
                    f"Upload ID {st} already exists. Regenerating a new one."
                )
                st = secrets.token_hex(nbytes)
            else:
                break
        else:
            d = glob.glob(os.path.join(output_dir, f"????/??/????????-??????-{st}"))
            if len(d) > 0:
                logger.warning(
                    f"Upload ID {st} already exists. Regenerating a new one."
                )
                st = secrets.token_hex(nbytes)
            else:
                break
    logger.info(f"Assigning a new secret token as an upload_id: {st}")

    return st
