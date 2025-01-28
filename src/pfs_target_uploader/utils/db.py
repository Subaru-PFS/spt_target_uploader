import datetime
import shutil
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
        df = pd.read_sql_query(
            f"SELECT upload_id FROM upload_id WHERE upload_id = '{upload_id}'",
            conn,
        )
    return not df.empty  # True if the upload ID exists in the database


def remove_duplicate_uid_db(db_path, backup=True, dry_run=False):
    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql_query("SELECT upload_id FROM upload_id", conn)

    if df["upload_id"].duplicated().any():
        logger.info(f"{df['upload_id'].duplicated().sum()} duplicates found.")
        logger.info(
            f"Duplicate upload IDs: {df.loc[df['upload_id'].duplicated(), 'upload_id'].values}"
        )
        if dry_run:
            logger.info(
                "Dry run mode enabled. No changes will be made to the database."
            )
            return
        if backup:
            logger.info("Creating a backup of the database.")
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            shutil.copyfile(db_path, f"{db_path}.bak-{timestamp}")
            logger.info(f"Bakcup created: {db_path}.bak-{timestamp}")

        with closing(sqlite3.connect(db_path)) as conn:
            with closing(conn.cursor()) as cur:
                # Step 1: Create a temporary table
                cur.execute("CREATE TEMPORARY TABLE unique_upload_id (upload_id TEXT)")

                # Step 2: Insert unique values into the temporary table
                cur.execute(
                    "INSERT INTO unique_upload_id SELECT DISTINCT upload_id FROM upload_id"
                )

                # Step 3: Delete all records from the original table
                cur.execute("DELETE FROM upload_id")

                # Step 4: Insert unique values back into the original table
                cur.execute(
                    "INSERT INTO upload_id SELECT upload_id FROM unique_upload_id"
                )

                # Step 5: Drop the temporary table
                cur.execute("DROP TABLE unique_upload_id")

                conn.commit()

        logger.info("Duplicate upload IDs have been removed from the database.")
    else:
        logger.info("No duplicates found. No changes made to the database.")
