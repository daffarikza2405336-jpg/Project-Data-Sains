from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import re
import secrets
import sqlite3

from .database import database_connection


PASSWORD_ITERATIONS = 310_000
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_DUMMY_SALT = b"StatlyticsLabDummySalt"
_DUMMY_HASH = hashlib.pbkdf2_hmac(
    "sha256", b"invalid-password", _DUMMY_SALT, PASSWORD_ITERATIONS
)


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
        dklen=32,
    )


def _validate_password(password: str) -> None:
    if len(password) < 12:
        raise ValueError("Kata sandi harus memiliki minimal 12 karakter.")
    if len(password) > 1024:
        raise ValueError("Kata sandi terlalu panjang.")


def register_user(username: str, password: str) -> str:
    normalized_username = username.strip()
    if not USERNAME_PATTERN.fullmatch(normalized_username):
        raise ValueError(
            "Username harus 3–32 karakter dan hanya boleh berisi huruf, angka, titik, garis bawah, atau tanda hubung."
        )
    _validate_password(password)

    salt = secrets.token_bytes(16)
    password_digest = _password_hash(password, salt)
    recovery_code = secrets.token_urlsafe(24)
    recovery_digest = hashlib.sha256(recovery_code.encode("utf-8")).digest()
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with database_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (username, password_salt, password_hash, recovery_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    normalized_username,
                    salt,
                    password_digest,
                    recovery_digest,
                    created_at,
                ),
            )
    except sqlite3.IntegrityError as error:
        if "users.username" in str(error):
            raise ValueError("Username tersebut sudah digunakan.") from error
        raise
    return recovery_code


def authenticate_user(username: str, password: str) -> str | None:
    if len(password) > 1024:
        return None
    with database_connection() as connection:
        user = connection.execute(
            "SELECT username, password_salt, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

    if user is None:
        hmac.compare_digest(
            _password_hash(password, _DUMMY_SALT),
            _DUMMY_HASH,
        )
        return None

    candidate_hash = _password_hash(password, user["password_salt"])
    if not hmac.compare_digest(candidate_hash, user["password_hash"]):
        return None
    return str(user["username"])


def change_password(username: str, current_password: str, new_password: str) -> str:
    _validate_password(new_password)
    with database_connection() as connection:
        user = connection.execute(
            "SELECT password_salt, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if user is None:
            raise ValueError("Akun tidak ditemukan.")
        if not hmac.compare_digest(
            _password_hash(current_password, user["password_salt"]),
            user["password_hash"],
        ):
            raise ValueError("Kata sandi saat ini tidak cocok.")
        new_salt = secrets.token_bytes(16)
        recovery_code = secrets.token_urlsafe(24)
        recovery_digest = hashlib.sha256(recovery_code.encode("utf-8")).digest()
        connection.execute(
            """
            UPDATE users
            SET password_salt = ?, password_hash = ?, recovery_hash = ?
            WHERE username = ?
            """,
            (
                new_salt,
                _password_hash(new_password, new_salt),
                recovery_digest,
                username,
            ),
        )
    return recovery_code


def reset_password(username: str, recovery_code: str, new_password: str) -> str:
    _validate_password(new_password)
    recovery_digest = hashlib.sha256(recovery_code.strip().encode("utf-8")).digest()
    with database_connection() as connection:
        user = connection.execute(
            "SELECT recovery_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        if user is None or not hmac.compare_digest(
            recovery_digest,
            user["recovery_hash"],
        ):
            raise ValueError("Username atau recovery code tidak cocok.")
        new_salt = secrets.token_bytes(16)
        new_recovery_code = secrets.token_urlsafe(24)
        new_recovery_hash = hashlib.sha256(
            new_recovery_code.encode("utf-8")
        ).digest()
        connection.execute(
            """
            UPDATE users
            SET password_salt = ?, password_hash = ?, recovery_hash = ?
            WHERE username = ?
            """,
            (
                new_salt,
                _password_hash(new_password, new_salt),
                new_recovery_hash,
                username.strip(),
            ),
        )
    return new_recovery_code
