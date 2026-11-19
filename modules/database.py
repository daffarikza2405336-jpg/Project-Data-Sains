from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Generator


DATABASE_PATH = Path(
    os.environ.get(
        "STATLYTICS_DB_PATH",
        str(Path(__file__).resolve().parents[1] / "data" / "statlytics.sqlite3"),
    )
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY COLLATE NOCASE,
    password_salt BLOB NOT NULL,
    password_hash BLOB NOT NULL,
    recovery_hash BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    username TEXT NOT NULL COLLATE NOCASE,
    name TEXT NOT NULL,
    is_favorite INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS projects_owner_updated
    ON projects(username, updated_at DESC);
CREATE TABLE IF NOT EXISTS project_activities (
    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT,
    username TEXT NOT NULL COLLATE NOCASE,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS activities_owner_created
    ON project_activities(username, created_at DESC);
"""


@contextmanager
def database_connection() -> Generator[sqlite3.Connection, None, None]:
    DATABASE_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        DATABASE_PATH.parent.chmod(0o700)
    except OSError:
        pass

    connection = sqlite3.connect(DATABASE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    try:
        connection.executescript(SCHEMA)
        yield connection
        connection.commit()
        try:
            DATABASE_PATH.chmod(0o600)
        except OSError:
            pass
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
