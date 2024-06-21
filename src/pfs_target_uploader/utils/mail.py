#!/usr/bin/env python3

import datetime
import smtplib
import time
from email.message import EmailMessage

import pytz
from loguru import logger


def send_email(
    config, outdir=None, outfile=None, upload_id=None, upload_time=None, url=None
):
    """
    Send an email notification about a new submission to the PFS Target Uploader.

    Parameters:
    ----------
    config : dict
        A dictionary containing email configuration settings.
    outdir : str, optional
        The upload directory. Default is None.
    outfile : str, optional
        The upload file. Default is None.
    upload_id : str, optional
        The upload ID. Default is None.
    upload_time : datetime, optional
        The upload time. Default is None.
    url : str, optional
        The URL to the PFS Target Uploader. Default is None

    Returns:
    -------
    None

    """

    fmt_t = "%Y-%m-%d %H:%M:%S"

    if upload_time is None:
        t_hst, t_utc = None, None
    else:
        t_hst = upload_time.astimezone(pytz.timezone("US/Hawaii")).strftime(fmt_t)
        t_utc = upload_time.astimezone(pytz.utc).strftime(fmt_t)

    message_text = f"""A new submission has been made on the PFS Target Uploader.

Upload ID: {upload_id}
Upload Time (UTC): {t_utc}
Upload Time (HST): {t_hst}
Upload Directory: {outdir}
Upload File: {outfile}
Upload URL: {url}
"""

    logger.info(f"Seinding an email:\n{message_text}")

    msg = EmailMessage()
    msg["Subject"] = (
        f"[pfs-target-uploader] New submission {upload_id} on the PFS Target Uploader"
    )
    msg["From"] = config["EMAIL_FROM"]
    msg["To"] = config["EMAIL_TO"]
    msg.set_content(message_text)

    with smtplib.SMTP(config["SMTP_SERVER"]) as s:
        s.send_message(msg)
