from __future__ import annotations

from datetime import datetime, timezone
import uuid

from .database import database_connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_project(username: str, name: str) -> str:
    project_name = name.strip()
    if not project_name:
        raise ValueError("Nama proyek tidak boleh kosong.")
    if len(project_name) > 80:
        raise ValueError("Nama proyek maksimal 80 karakter.")

    project_id = str(uuid.uuid4())
    timestamp = _now()
    with database_connection() as connection:
        connection.execute(
            """
            INSERT INTO projects (project_id, username, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project_id, username, project_name, timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO project_activities (project_id, username, action, detail, created_at)
            VALUES (?, ?, 'created', ?, ?)
            """,
            (project_id, username, project_name, timestamp),
        )
    return project_id


def list_projects(username: str) -> list[dict[str, object]]:
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT project_id, name, is_favorite, created_at, updated_at
            FROM projects
            WHERE username = ?
            ORDER BY is_favorite DESC, updated_at DESC
            """,
            (username,),
        ).fetchall()
    return [dict(row) for row in rows]


def rename_project(username: str, project_id: str, name: str) -> bool:
    project_name = name.strip()
    if not project_name:
        raise ValueError("Nama proyek tidak boleh kosong.")
    if len(project_name) > 80:
        raise ValueError("Nama proyek maksimal 80 karakter.")

    timestamp = _now()
    with database_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE projects SET name = ?, updated_at = ?
            WHERE project_id = ? AND username = ?
            """,
            (project_name, timestamp, project_id, username),
        )
        if cursor.rowcount != 1:
            return False
        connection.execute(
            """
            INSERT INTO project_activities (project_id, username, action, detail, created_at)
            VALUES (?, ?, 'renamed', ?, ?)
            """,
            (project_id, username, project_name, timestamp),
        )
    return True


def set_project_favorite(username: str, project_id: str, favorite: bool) -> bool:
    timestamp = _now()
    with database_connection() as connection:
        project = connection.execute(
            "SELECT name FROM projects WHERE project_id = ? AND username = ?",
            (project_id, username),
        ).fetchone()
        if project is None:
            return False
        cursor = connection.execute(
            """
            UPDATE projects SET is_favorite = ?, updated_at = ?
            WHERE project_id = ? AND username = ?
            """,
            (int(favorite), timestamp, project_id, username),
        )
        if cursor.rowcount == 1:
            connection.execute(
                """
                INSERT INTO project_activities (project_id, username, action, detail, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    username,
                    "favorited" if favorite else "unfavorited",
                    str(project["name"]),
                    timestamp,
                ),
            )
    return cursor.rowcount == 1


def delete_project(username: str, project_id: str) -> bool:
    with database_connection() as connection:
        project = connection.execute(
            "SELECT name FROM projects WHERE project_id = ? AND username = ?",
            (project_id, username),
        ).fetchone()
        if project is None:
            return False
        connection.execute(
            "DELETE FROM projects WHERE project_id = ? AND username = ?",
            (project_id, username),
        )
        connection.execute(
            """
            INSERT INTO project_activities (project_id, username, action, detail, created_at)
            VALUES (?, ?, 'deleted', ?, ?)
            """,
            (project_id, username, str(project["name"]), _now()),
        )
    return True


def list_activities(username: str, limit: int = 20) -> list[dict[str, object]]:
    if limit < 1 or limit > 100:
        raise ValueError("Batas aktivitas harus antara 1 dan 100.")
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT action, detail, created_at
            FROM project_activities
            WHERE username = ?
            ORDER BY activity_id DESC
            LIMIT ?
            """,
            (username, limit),
        ).fetchall()
    return [dict(row) for row in rows]
