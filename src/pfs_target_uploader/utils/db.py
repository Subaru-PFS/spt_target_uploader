import sqlite3
from contextlib import closing

import pandas as pd
from loguru import logger


def create_uid_db(db_path):
    """
    Create a SQLite database with a table for upload IDs if it does not exist.
    Parameters
    ----------
    db_path : str
        The file path to the SQLite database.
    Notes
    -----
    This function creates a table named `upload_id` with a single column `upload_id` of type TEXT.
    If the table already exists, it will not be recreated.
    Here, no UNIQUE constraint is added to the `upload_id` column, so duplicate entries can happen.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS upload_id (upload_id TEXT)")
            conn.commit()


def bulk_insert_uid_db(df, db_path):
    logger.info(f"Inserting {df.shape[0]} upload IDs to the database.")
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.executemany(
                "INSERT INTO upload_id VALUES (?)", [(x,) for x in df["upload_id"]]
            )
            conn.commit()


def single_insert_uid_db(upload_id, db_path):
    logger.info(f"Inserting a new upload ID to the database: {upload_id}")
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("INSERT INTO upload_id (upload_id) VALUES (?)", (upload_id,))
            conn.commit()


def upload_id_exists(upload_id, db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql_query("SELECT upload_id FROM upload_id", conn)

    return upload_id in df["upload_id"].values
