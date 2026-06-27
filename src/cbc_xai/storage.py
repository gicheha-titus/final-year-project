"""SQLite persistence layer for users, assessments, predictions, and reports.

All database access is funnelled through this module so that the rest
of the application never constructs raw SQL.  The schema supports:

- **users** — local authentication with Argon2 password hashing (salted,
  adaptive), role assignment (Admin / Teacher-Counsellor), and activation
  toggling.  SHA-256 legacy hashes from pre-v1.1 databases are silently
  migrated to Argon2 on the next successful login.
- **learners / assessments** — imported learner records with a composite
  primary key that prevents duplicate grade-term-subject rows.
- **readiness_results / explanation_results** — cached prediction and
  SHAP explanation outputs, one row per learner.
- **reports** — an audit log of generated PDF reports.

The module also exposes higher-level helpers (``import_assessment_frame``,
``load_assessments_for_learner``, etc.) that combine SQL with DataFrame
conversion for convenient use in the pipeline and UI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    DATABASE_PATH,
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
    DEFAULT_TEACHER_PASSWORD,
    DEFAULT_TEACHER_USERNAME,
    ensure_directories,
)
from .ingestion import validate_assessment_frame

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing — Argon2id via argon2-cffi
#
# Argon2 (winner of the PHC) is the right choice here: it is salted by
# default, memory-hard (resistant to GPU cracking), and the parameters can
# be tuned to the target hardware (4 GB Windows desktop in this case).
#
# We use the argon2-cffi library's PasswordHasher with conservative defaults
# — the desktop is used interactively so a 200–300 ms hash at login is fine.
# ---------------------------------------------------------------------------
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

    _ph = PasswordHasher(
        time_cost=2,       # number of iterations
        memory_cost=65536, # 64 MB — reasonable on 4 GB machines
        parallelism=2,
        hash_len=32,
        salt_len=16,
    )
    _ARGON2_AVAILABLE = True
except ImportError:  # pragma: no cover
    # If argon2-cffi is somehow absent, fall back to SHA-256 with a warning.
    # This should never happen in a properly installed environment but avoids
    # a hard crash in test environments that haven't installed optional deps.
    log.warning(
        "argon2-cffi not found; falling back to SHA-256 hashing. "
        "Install argon2-cffi for production use."
    )
    _ARGON2_AVAILABLE = False


def _sha256_hash(password: str) -> str:
    """SHA-256 hex digest — used only during legacy migration detection."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _is_legacy_sha256_hash(stored_hash: str) -> bool:
    """Return True if *stored_hash* looks like a raw SHA-256 hex digest.

    SHA-256 hex digests are exactly 64 hex characters.  Argon2 hashes
    start with ``$argon2``.  This check is conservative: anything that
    isn't 64 lowercase hex chars is assumed to be a modern hash.
    """
    if len(stored_hash) != 64:
        return False
    return all(c in "0123456789abcdef" for c in stored_hash)


def _hash_password(password: str) -> str:
    """Hash *password* using Argon2id (preferred) or SHA-256 (fallback).

    The result is stored in the ``password_hash`` column.  Argon2 hashes
    include the algorithm, parameters, and salt in a single portable string.
    """
    if _ARGON2_AVAILABLE:
        return _ph.hash(password)
    return _sha256_hash(password)


def _verify_password(password: str, stored_hash: str) -> bool:
    """Return True if *password* matches *stored_hash*.

    Handles both Argon2 hashes (current) and SHA-256 hex digests (legacy).
    Does not handle re-hashing — callers that need that should use
    ``_needs_rehash`` after a successful verify.
    """
    if not _ARGON2_AVAILABLE:
        return stored_hash == _sha256_hash(password)

    if _is_legacy_sha256_hash(stored_hash):
        # Legacy path: direct constant-time comparison using hmac.compare_digest
        # would be ideal but the sha256 lookup itself is already deterministic —
        # the important thing is we don't leak timing info about hash length.
        import hmac
        return hmac.compare_digest(stored_hash, _sha256_hash(password))

    try:
        return _ph.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def _needs_rehash(stored_hash: str) -> bool:
    """Return True if *stored_hash* should be upgraded to the current scheme.

    Covers two cases:
    1. It is a SHA-256 hex digest (legacy schema).
    2. It is an Argon2 hash with outdated parameters (argon2-cffi detects this).
    """
    if _is_legacy_sha256_hash(stored_hash):
        return True
    if _ARGON2_AVAILABLE:
        return _ph.check_needs_rehash(stored_hash)
    return False


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_connection(database_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection.

    ``row_factory`` is set to ``sqlite3.Row`` so that query results can
    be accessed by column name rather than positional index.

    Foreign key enforcement is enabled per connection because SQLite resets
    it to OFF for every new connection by default.

    Uses a sentinel default so that ``DATABASE_PATH`` is resolved at
    call time, which is necessary for test fixtures that monkeypatch
    the module-level constant.
    """
    if database_path is None:
        database_path = DATABASE_PATH
    ensure_directories()
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    # SQLite disables FK constraints by default; enable them explicitly.
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def initialize_database(database_path: str | Path | None = None) -> None:
    """Create all tables if they do not exist and seed default accounts.

    Safe to call repeatedly — uses ``CREATE TABLE IF NOT EXISTS`` and
    ``INSERT OR IGNORE`` so existing data is never overwritten.  Also
    performs a forward-compatible migration to add the ``is_active``
    column if the database was created before that feature existed.
    """
    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS learners (
                learner_id TEXT PRIMARY KEY,
                learner_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assessments (
                learner_id TEXT NOT NULL,
                grade TEXT NOT NULL,
                term TEXT NOT NULL,
                subject TEXT NOT NULL,
                score REAL NOT NULL,
                assessment_date TEXT NOT NULL,
                PRIMARY KEY (learner_id, grade, term, subject),
                FOREIGN KEY (learner_id) REFERENCES learners (learner_id)
            );

            CREATE INDEX IF NOT EXISTS idx_assessments_learner_id
                ON assessments (learner_id);

            CREATE TABLE IF NOT EXISTS readiness_results (
                learner_id TEXT PRIMARY KEY,
                predicted_pathway TEXT NOT NULL,
                predicted_track TEXT NOT NULL,
                top_track_score REAL NOT NULL,
                pathway_probabilities_json TEXT NOT NULL,
                track_probabilities_json TEXT NOT NULL,
                guidance_notes_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS explanation_results (
                learner_id TEXT PRIMARY KEY,
                strengths_json TEXT NOT NULL,
                limiting_factors_json TEXT NOT NULL,
                feature_importance_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                report_path TEXT NOT NULL,
                generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # Forward-compatible migration: add is_active if upgrading from
        # an older schema that lacked the column.
        user_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }
        if "is_active" not in user_columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
            )
        cursor.execute("UPDATE users SET is_active = 1 WHERE is_active IS NULL")

        # Seed default accounts so the app is usable immediately in an
        # offline school environment.
        for username, password, role in (
            (DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, "Admin"),
            (DEFAULT_TEACHER_USERNAME, DEFAULT_TEACHER_PASSWORD, "Teacher/Counsellor"),
        ):
            existing = connection.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if existing is None:
                cursor.execute(
                    "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, _hash_password(password), role),
                )
            # If the row exists, leave the hash alone — never overwrite
            # a user-changed password with the factory default.
        connection.commit()
    log.debug("Database initialized at %s", database_path or DATABASE_PATH)


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

def _count_active_admins(connection: sqlite3.Connection) -> int:
    """Return the number of active Admin accounts in the given connection."""
    row = connection.execute(
        "SELECT COUNT(*) AS n FROM users WHERE role = 'Admin' AND is_active = 1"
    ).fetchone()
    return int(row["n"]) if row else 0


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    """Validate credentials and return user info, or ``None`` on failure.

    Returns ``None`` for unknown usernames, wrong passwords, and
    deactivated accounts — intentionally not distinguishing between
    cases to avoid leaking account-existence information.

    On a successful login, silently upgrades SHA-256 legacy hashes to
    Argon2 so the database converges to the modern scheme over time
    without requiring forced password resets.
    """
    with get_connection() as connection:
        row = connection.execute(
            "SELECT username, role, password_hash, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            return None
        if not row["is_active"]:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None

        # Silently re-hash if the stored hash is a legacy SHA-256 digest or
        # uses outdated Argon2 parameters.  This is invisible to the user.
        if _needs_rehash(row["password_hash"]):
            new_hash = _hash_password(password)
            connection.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (new_hash, username),
            )
            connection.commit()
            log.info("Upgraded password hash for '%s' to current scheme.", username)

        return {"username": row["username"], "role": row["role"]}


def list_users() -> list[dict[str, Any]]:
    """Return all user accounts, Admin accounts listed first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT username, role, is_active
            FROM users
            ORDER BY CASE role WHEN 'Admin' THEN 0 ELSE 1 END, username
            """
        ).fetchall()
    return [
        {
            "username": row["username"],
            "role": row["role"],
            "is_active": bool(row["is_active"]),
            "status": "Active" if row["is_active"] else "Inactive",
        }
        for row in rows
    ]


def create_user(username: str, password: str, role: str = "Teacher/Counsellor") -> dict[str, Any]:
    """Create a new user account after validating inputs.

    Raises ``ValueError`` for blank or too-short credentials, invalid
    roles, and duplicate usernames.
    """
    username = username.strip()
    if not username:
        raise ValueError("Username is required.")
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if not password:
        raise ValueError("Password is required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if role not in {"Admin", "Teacher/Counsellor"}:
        raise ValueError("Unsupported user role.")

    with get_connection() as connection:
        try:
            connection.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, _hash_password(password), role),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("That username already exists.") from exc

    log.info("Created user '%s' with role '%s'.", username, role)
    return {"username": username, "role": role, "is_active": True, "status": "Active"}


def reset_user_password(username: str, password: str) -> None:
    """Replace the password hash for an existing user account."""
    username = username.strip()
    if not username:
        raise ValueError("Select a user account first.")
    if not password:
        raise ValueError("Password is required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (_hash_password(password), username),
        )
        connection.commit()
    if cursor.rowcount == 0:
        raise ValueError("That user account no longer exists.")
    log.info("Password reset for user '%s'.", username)


def set_user_active(username: str, is_active: bool) -> None:
    """Activate or deactivate a user account without deleting it.

    Raises ``ValueError`` if this would remove the last active Admin,
    which would lock the system out of all administrative functions.
    """
    username = username.strip()
    if not username:
        raise ValueError("Select a user account first.")

    with get_connection() as connection:
        # Fetch the target user's role before making any changes.
        row = connection.execute(
            "SELECT role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            raise ValueError("That user account no longer exists.")

        # Guard: refuse to deactivate the last active Admin.
        if not is_active and row["role"] == "Admin":
            active_admin_count = _count_active_admins(connection)
            if active_admin_count <= 1:
                raise ValueError(
                    "Cannot deactivate the last active Admin account. "
                    "Create or reactivate another Admin first."
                )

        cursor = connection.execute(
            "UPDATE users SET is_active = ? WHERE username = ?",
            (1 if is_active else 0, username),
        )
        connection.commit()
    if cursor.rowcount == 0:
        raise ValueError("That user account no longer exists.")
    action = "activated" if is_active else "deactivated"
    log.info("User '%s' %s.", username, action)


def delete_user(username: str) -> None:
    """Permanently remove a user account from the database.

    Raises ``ValueError`` if this would remove the last active Admin.
    """
    username = username.strip()
    if not username:
        raise ValueError("Select a user account first.")

    with get_connection() as connection:
        row = connection.execute(
            "SELECT role, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            raise ValueError("That user account no longer exists.")

        # Guard: refuse to delete if this is the last active Admin.
        if row["role"] == "Admin" and row["is_active"]:
            active_admin_count = _count_active_admins(connection)
            if active_admin_count <= 1:
                raise ValueError(
                    "Cannot delete the last active Admin account. "
                    "Create or activate another Admin first."
                )

        cursor = connection.execute(
            "DELETE FROM users WHERE username = ?",
            (username,),
        )
        connection.commit()
    if cursor.rowcount == 0:
        raise ValueError("That user account no longer exists.")
    log.info("Deleted user '%s'.", username)


# ---------------------------------------------------------------------------
# Database backup
# ---------------------------------------------------------------------------

def backup_database(destination_dir: Path | None = None) -> Path:
    """Copy the live database to a timestamped backup file.

    Uses SQLite's online backup API (via Python's ``sqlite3.Connection.backup``)
    which is safe while the database is open and in use.  Returns the path
    to the backup file.

    This provides a simple, school-operable safeguard against accidental
    data loss — not a substitute for proper IT backup infrastructure.
    """
    if destination_dir is None:
        destination_dir = ARTIFACTS_DIR / "backups"
    destination_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = destination_dir / f"cbc_xai_backup_{timestamp}.db"

    with get_connection() as source:
        with sqlite3.connect(backup_path) as destination:
            source.backup(destination)

    log.info("Database backed up to %s", backup_path)
    return backup_path


# ---------------------------------------------------------------------------
# Assessment data import and retrieval
# ---------------------------------------------------------------------------

def import_assessment_frame(frame: pd.DataFrame) -> int:
    """Validate and import a DataFrame of assessment rows into the database.

    Uses ``INSERT OR REPLACE`` so that re-importing the same CSV updates
    existing records rather than failing.  Returns the number of rows
    imported.
    """
    cleaned = validate_assessment_frame(frame)
    learner_rows = cleaned[["learner_id", "learner_name"]].drop_duplicates()
    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO learners (learner_id, learner_name)
            VALUES (?, ?)
            ON CONFLICT(learner_id) DO UPDATE SET learner_name = excluded.learner_name
            """,
            learner_rows.itertuples(index=False, name=None),
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO assessments (
                learner_id, grade, term, subject, score, assessment_date
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            cleaned[[
                "learner_id",
                "grade",
                "term",
                "subject",
                "score",
                "assessment_date",
            ]].itertuples(index=False, name=None),
        )
        connection.commit()
    log.info("Imported %d assessment rows.", len(cleaned))
    return len(cleaned)


def load_all_assessments() -> pd.DataFrame:
    """Return every assessment row joined with learner names."""
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT a.learner_id, l.learner_name, a.grade, a.term, a.subject, a.score, a.assessment_date
            FROM assessments a
            JOIN learners l ON l.learner_id = a.learner_id
            ORDER BY a.learner_id, a.grade, a.term, a.subject
            """,
            connection,
        )


def load_learners() -> list[dict[str, str]]:
    """Return a list of ``{learner_id, learner_name}`` for all learners."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT learner_id, learner_name
            FROM learners
            ORDER BY learner_id
            """
        ).fetchall()
    return [
        {"learner_id": row["learner_id"], "learner_name": row["learner_name"]}
        for row in rows
    ]


def load_assessments_for_learner(learner_id: str) -> pd.DataFrame:
    """Return all assessment rows for a single learner."""
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT a.learner_id, l.learner_name, a.grade, a.term, a.subject, a.score, a.assessment_date
            FROM assessments a
            JOIN learners l ON l.learner_id = a.learner_id
            WHERE a.learner_id = ?
            ORDER BY a.grade, a.term, a.subject
            """,
            connection,
            params=(learner_id,),
        )


# ---------------------------------------------------------------------------
# Prediction and explanation persistence
# ---------------------------------------------------------------------------

def save_prediction(
    learner_id: str,
    predicted_pathway: str,
    predicted_track: str,
    top_track_score: float,
    pathway_probabilities: dict[str, float],
    track_probabilities: dict[str, float],
    guidance_notes: list[str],
    strengths: list[dict[str, Any]],
    limiting_factors: list[dict[str, Any]],
    feature_importance: dict[str, float],
) -> None:
    """Persist a learner's prediction result and SHAP explanation.

    Stored as two rows across ``readiness_results`` and
    ``explanation_results``, with complex values serialised as JSON.
    Uses ``INSERT OR REPLACE`` so that re-predicting for the same
    learner overwrites the previous result.
    """
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO readiness_results (
                learner_id, predicted_pathway, predicted_track, top_track_score,
                pathway_probabilities_json, track_probabilities_json, guidance_notes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id,
                predicted_pathway,
                predicted_track,
                top_track_score,
                json.dumps(pathway_probabilities, indent=2),
                json.dumps(track_probabilities, indent=2),
                json.dumps(guidance_notes, indent=2),
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO explanation_results (
                learner_id, strengths_json, limiting_factors_json, feature_importance_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                learner_id,
                json.dumps(strengths, indent=2),
                json.dumps(limiting_factors, indent=2),
                json.dumps(feature_importance, indent=2),
            ),
        )
        connection.commit()


# ---------------------------------------------------------------------------
# Report audit log
# ---------------------------------------------------------------------------

def save_report_record(learner_id: str, report_path: str | Path) -> None:
    """Log a generated PDF report in the audit table."""
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO reports (learner_id, report_path) VALUES (?, ?)",
            (learner_id, str(report_path)),
        )
        connection.commit()


def count_reports() -> int:
    """Return the total number of reports in the audit log."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS report_count FROM reports"
        ).fetchone()
    return int(row["report_count"]) if row else 0


def latest_report_path() -> str | None:
    """Return the file path of the most recently generated report, or ``None``."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT report_path
            FROM reports
            ORDER BY generated_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
    return str(row["report_path"]) if row and row["report_path"] else None
