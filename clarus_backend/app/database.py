"""
Database module — SQLite, one table, matching the encounter-record
fields described in Chapter 3's Data Models section.

SQLite (not a full Postgres/MySQL server) is the right call for a
capstone-scale project: zero setup, one file on disk, plenty fast enough
for a screening tool that isn't handling concurrent hospital-wide traffic.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.user_model import USER_ROLES, User, UserRole

DB_PATH = os.getenv(
    "CLARUS_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "clarus.db"),
)

# The Philippines is UTC+8 with no daylight saving, so a fixed offset is
# exact and needs no timezone database.
PH_TZ = timezone(timedelta(hours=8))


def _connect() -> sqlite3.Connection:
    # timeout: wait up to 10 s on a locked database instead of failing at
    # once (the reminder job and API requests can write at the same time).
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db():
    conn = _connect()
    # WAL lets reads and writes proceed concurrently (persists in the file).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encounters (
            encounter_id TEXT PRIMARY KEY,
            worker_name TEXT,
            quality_pass INTEGER,
            quality_reason TEXT,
            dr_class TEXT,
            triage TEXT,
            confidence REAL,
            triage_confidence REAL,
            severity TEXT,
            severity_confidence REAL,
            blur_score REAL,
            illumination_score REAL,
            gradcam_url TEXT,
            patient_id TEXT,
            model_variant TEXT DEFAULT 'integrated',
            timestamp TEXT,
            created_by_user_id TEXT,
            FOREIGN KEY (created_by_user_id) REFERENCES users(user_id)
        )
    """)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(encounters)")
    }
    if "created_by_user_id" not in columns:
        conn.execute(
            "ALTER TABLE encounters ADD COLUMN created_by_user_id TEXT"
        )
    if "patient_id" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN patient_id TEXT")
    if "triage_confidence" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN triage_confidence REAL")
    if "severity" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN severity TEXT")
    if "severity_confidence" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN severity_confidence REAL")
    if "blur_score" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN blur_score REAL")
    if "illumination_score" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN illumination_score REAL")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_encounters_timestamp
        ON encounters(timestamp DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_encounters_creator
        ON encounters(created_by_user_id)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS follow_up_plans (
            follow_up_id TEXT PRIMARY KEY,
            encounter_id TEXT,
            patient_id TEXT NOT NULL,
            contact_method TEXT NOT NULL,
            contact_value TEXT NOT NULL,
            consent_given INTEGER NOT NULL,
            created_by_user_id TEXT,
            return_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'scheduled',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id),
            FOREIGN KEY (created_by_user_id) REFERENCES users(user_id)
        )
    """)
    follow_up_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(follow_up_plans)")
    }
    if "reminder_count" not in follow_up_columns:
        conn.execute(
            "ALTER TABLE follow_up_plans "
            "ADD COLUMN reminder_count INTEGER DEFAULT 0"
        )
    if "last_reminder_sent_at" not in follow_up_columns:
        conn.execute(
            "ALTER TABLE follow_up_plans "
            "ADD COLUMN last_reminder_sent_at TEXT"
        )
    if "created_by_user_id" not in follow_up_columns:
        conn.execute(
            "ALTER TABLE follow_up_plans ADD COLUMN created_by_user_id TEXT"
        )
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_follow_up_return_date
        ON follow_up_plans(return_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_follow_up_status
        ON follow_up_plans(status)
    """)
    _migrate_users_schema(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (
                role IN (
                    'health_worker',
                    'specialist',
                    'ophthalmologist_optometrist'
                )
            ),
            is_admin INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
            is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_role_active
        ON users(role, is_active)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email
        ON users(email COLLATE NOCASE)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS referrals (
            referral_id TEXT PRIMARY KEY,
            encounter_id TEXT NOT NULL UNIQUE,
            patient_id TEXT NOT NULL,
            created_by_user_id TEXT NOT NULL,
            assigned_to_user_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'assigned' CHECK (
                status IN (
                    'assigned',
                    'in_review',
                    'reviewed',
                    'completed',
                    'cancelled'
                )
            ),
            consent_given INTEGER NOT NULL CHECK (consent_given IN (0, 1)),
            consent_recorded_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id),
            FOREIGN KEY (created_by_user_id) REFERENCES users(user_id),
            FOREIGN KEY (assigned_to_user_id) REFERENCES users(user_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_referrals_assignee_status
        ON referrals(assigned_to_user_id, status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_referrals_creator
        ON referrals(created_by_user_id)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS referral_notes (
            note_id TEXT PRIMARY KEY,
            referral_id TEXT NOT NULL,
            author_user_id TEXT NOT NULL,
            note TEXT NOT NULL,
            recommendation TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (referral_id) REFERENCES referrals(referral_id),
            FOREIGN KEY (author_user_id) REFERENCES users(user_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_referral_notes_referral
        ON referral_notes(referral_id, created_at)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            audit_id TEXT PRIMARY KEY,
            user_id TEXT,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_logs_entity
        ON audit_logs(entity_type, entity_id, created_at)
    """)
    admin_ids = [
        row[0]
        for row in conn.execute(
            "SELECT user_id FROM users WHERE is_admin = 1 "
            "ORDER BY created_at, user_id"
        ).fetchall()
    ]
    if len(admin_ids) > 1:
        conn.executemany(
            "UPDATE users SET is_admin = 0 WHERE user_id = ?",
            [(user_id,) for user_id in admin_ids[1:]],
        )
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_one_admin
        ON users(is_admin)
        WHERE is_admin = 1
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            notification_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            referral_id TEXT,
            notification_type TEXT NOT NULL,
            message TEXT NOT NULL,
            read_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (referral_id) REFERENCES referrals(referral_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_notifications_user
        ON notifications(user_id, read_at, created_at)
    """)
    conn.commit()
    conn.close()


def _migrate_users_schema(conn: sqlite3.Connection) -> None:
    table = conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name = 'users'"
    ).fetchone()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if (
        table is None
        or (
            "'admin'" not in (table[0] or "").lower()
            and "is_admin" in columns
        )
    ):
        return

    conn.execute("""
        CREATE TABLE users_migrated (
            user_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (
                role IN (
                    'health_worker',
                    'specialist',
                    'ophthalmologist_optometrist'
                )
            ),
            is_admin INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
            is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        INSERT INTO users_migrated
        (user_id, full_name, email, password_hash, role, is_admin, is_active,
         created_at, updated_at)
        SELECT user_id, full_name, email, password_hash,
               CASE WHEN role = 'admin' THEN 'specialist' ELSE role END,
               CASE WHEN role = 'admin' THEN 1 ELSE 0 END,
               is_active, created_at, updated_at
        FROM users
    """)
    conn.execute("DROP TABLE users")
    conn.execute("ALTER TABLE users_migrated RENAME TO users")


def create_user(
    full_name: str,
    email: str,
    password_hash: str,
    role: UserRole,
    is_admin: bool = False,
) -> dict:
    full_name = full_name.strip()
    email = email.strip().lower()
    _validate_user_fields(full_name, email, password_hash, role)

    now = _utc_now()
    user_id = str(uuid.uuid4())
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO users
               (user_id, full_name, email, password_hash, role, is_admin,
                is_active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (user_id, full_name, email, password_hash, role, int(is_admin), now, now),
        )
        conn.commit()
    except sqlite3.IntegrityError as error:
        conn.rollback()
        if "users.email" in str(error).lower():
            raise ValueError("A user with this email already exists.") from error
        raise
    finally:
        conn.close()

    return get_user(user_id)


def get_user(user_id: str) -> dict | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT user_id, full_name, email, role, is_admin, is_active,
                  created_at, updated_at
           FROM users
           WHERE user_id = ?""",
        (user_id,),
    ).fetchone()
    conn.close()
    return _public_user(row)


def get_user_for_auth(email: str) -> dict | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT user_id, full_name, email, password_hash, role, is_admin, is_active,
                  created_at, updated_at
           FROM users
           WHERE email = ? COLLATE NOCASE""",
        (email.strip().lower(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def count_users() -> int:
    conn = _connect()
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return int(count)


def list_users(role: UserRole | None = None) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    if role is None:
        rows = conn.execute(
            """SELECT user_id, full_name, email, role, is_active,
                      created_at, updated_at
               FROM users
               ORDER BY full_name COLLATE NOCASE"""
        ).fetchall()
    else:
        _validate_role(role)
        rows = conn.execute(
            """SELECT user_id, full_name, email, role, is_active,
                      created_at, updated_at
               FROM users
               WHERE role = ?
               ORDER BY full_name COLLATE NOCASE""",
            (role,),
        ).fetchall()
    conn.close()
    return [_public_user(row) for row in rows]


def set_user_active(user_id: str, is_active: bool) -> dict | None:
    conn = _connect()
    cursor = conn.execute(
        "UPDATE users SET is_active = ?, updated_at = ? WHERE user_id = ?",
        (int(is_active), _utc_now(), user_id),
    )
    conn.commit()
    conn.close()
    return get_user(user_id) if cursor.rowcount else None


def _validate_user_fields(
    full_name: str,
    email: str,
    password_hash: str,
    role: UserRole,
) -> None:
    if not full_name:
        raise ValueError("A user's full name is required.")
    if not email:
        raise ValueError("A user's email is required.")
    if not password_hash:
        raise ValueError("A password hash is required.")
    _validate_role(role)


def _validate_role(role: str) -> None:
    if role not in USER_ROLES:
        valid_roles = ", ".join(USER_ROLES)
        raise ValueError(f"Invalid user role. Expected one of: {valid_roles}.")


def _public_user(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return User.from_row(dict(row)).to_public_dict()


def get_encounter(encounter_id: str) -> dict | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT * FROM encounters
           WHERE encounter_id = ? AND model_variant = 'integrated'""",
        (encounter_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_encounter_for_gradcam(filename: str) -> dict | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT * FROM encounters
           WHERE model_variant = 'integrated'
             AND gradcam_url LIKE ?""",
        (f"%/gradcam/{filename}",),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_history_for_user(user: dict, limit: int = 50) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    if user["is_admin"]:
        query = """
            SELECT * FROM encounters
            WHERE model_variant = 'integrated'
            ORDER BY timestamp DESC LIMIT ?
        """
        params = (limit,)
    elif user["role"] == "health_worker":
        query = """
            SELECT * FROM encounters
            WHERE model_variant = 'integrated'
              AND created_by_user_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """
        params = (user["user_id"], limit)
    else:
        query = """
            SELECT e.* FROM encounters e
            JOIN referrals r ON r.encounter_id = e.encounter_id
            WHERE e.model_variant = 'integrated'
              AND r.assigned_to_user_id = ?
            ORDER BY e.timestamp DESC LIMIT ?
        """
        params = (user["user_id"], limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_referral(
    encounter_id: str,
    patient_id: str,
    created_by_user_id: str,
    assigned_to_user_id: str,
) -> dict:
    timestamp = _utc_now()
    referral_id = str(uuid.uuid4())
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO referrals
               (referral_id, encounter_id, patient_id, created_by_user_id,
                assigned_to_user_id, status, consent_given,
                consent_recorded_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'assigned', 1, ?, ?, ?)""",
            (
                referral_id,
                encounter_id,
                patient_id,
                created_by_user_id,
                assigned_to_user_id,
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as error:
        conn.rollback()
        if "referrals.encounter_id" in str(error).lower():
            raise ValueError(
                "A referral already exists for this encounter."
            ) from error
        raise
    finally:
        conn.close()
    return get_referral(referral_id)


def get_referrals_for_user(user: dict) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    if user["is_admin"]:
        rows = conn.execute(
            "SELECT * FROM referrals ORDER BY updated_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM referrals
               WHERE assigned_to_user_id = ? OR created_by_user_id = ?
               ORDER BY updated_at DESC""",
            (user["user_id"], user["user_id"]),
        ).fetchall()
    conn.close()
    return [_referral_dict(row) for row in rows]


def get_referral(referral_id: str) -> dict | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM referrals WHERE referral_id = ?",
        (referral_id,),
    ).fetchone()
    conn.close()
    return _referral_dict(row) if row else None


def update_referral_status(referral_id: str, status: str) -> dict | None:
    conn = _connect()
    cursor = conn.execute(
        """UPDATE referrals
           SET status = ?, updated_at = ?
           WHERE referral_id = ?""",
        (status, _utc_now(), referral_id),
    )
    conn.commit()
    conn.close()
    return get_referral(referral_id) if cursor.rowcount else None


def create_referral_note(
    referral_id: str,
    author_user_id: str,
    note: str,
    recommendation: str | None,
) -> dict:
    note_id = str(uuid.uuid4())
    created_at = _utc_now()
    conn = _connect()
    conn.execute(
        """INSERT INTO referral_notes
           (note_id, referral_id, author_user_id, note, recommendation, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (note_id, referral_id, author_user_id, note, recommendation, created_at),
    )
    conn.commit()
    conn.close()
    return get_referral_note(note_id)


def get_referral_notes(referral_id: str) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM referral_notes
           WHERE referral_id = ?
           ORDER BY created_at ASC""",
        (referral_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_referral_note(note_id: str) -> dict | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM referral_notes WHERE note_id = ?",
        (note_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def write_audit_log(
    user_id: str | None,
    action: str,
    entity_type: str,
    entity_id: str | None,
) -> None:
    conn = _connect()
    conn.execute(
        """INSERT INTO audit_logs
           (audit_id, user_id, action, entity_type, entity_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (str(uuid.uuid4()), user_id, action, entity_type, entity_id, _utc_now()),
    )
    conn.commit()
    conn.close()


def create_notification(
    user_id: str,
    notification_type: str,
    message: str,
    referral_id: str | None = None,
) -> dict:
    notification_id = str(uuid.uuid4())
    created_at = _utc_now()
    conn = _connect()
    conn.row_factory = sqlite3.Row
    conn.execute(
        """INSERT INTO notifications
           (notification_id, user_id, referral_id, notification_type,
            message, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            notification_id,
            user_id,
            referral_id,
            notification_type,
            message,
            created_at,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM notifications WHERE notification_id = ?",
        (notification_id,),
    ).fetchone()
    conn.close()
    return dict(row)


def get_notifications(user_id: str) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM notifications
           WHERE user_id = ?
           ORDER BY created_at DESC
           LIMIT 100""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_notification_read(notification_id: str, user_id: str) -> bool:
    conn = _connect()
    cursor = conn.execute(
        """UPDATE notifications SET read_at = ?
           WHERE notification_id = ? AND user_id = ?""",
        (_utc_now(), notification_id, user_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def _referral_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    result["consent_given"] = bool(result["consent_given"])
    result["encounter"] = get_encounter(result["encounter_id"])
    result["notes"] = get_referral_notes(result["referral_id"])
    return result


def save_encounter(worker_name: str, quality_result: dict, dr_class: str,
                    triage: str, confidence: float, gradcam_url: str,
                    triage_confidence: float | None = None,
                    severity: str | None = None,
                    severity_confidence: float | None = None,
                    model_variant: str = "integrated",
                    patient_id: str | None = None,
                    blur_score: float | None = None,
                    illumination_score: float | None = None,
                    created_by_user_id: str | None = None) -> str:
    """
    model_variant distinguishes which pipeline produced this prediction —
    "integrated" (the full Clarus workflow) or "baseline" (a plain CNN
    with none of the four added components), matching the Comparative
    Evaluation Module described in Chapter 3. Both stubbed identically
    for now until Person A's real models exist for each.
    """
    timestamp = _utc_now()
    conn = _connect()

    for attempt in range(5):
        encounter_id = str(uuid.uuid4())
        try:
            conn.execute(
                """INSERT INTO encounters
                   (encounter_id, worker_name, quality_pass, quality_reason,
                                                     dr_class, triage, confidence, triage_confidence,
                                                     severity, severity_confidence, blur_score, illumination_score,
                                                     gradcam_url, patient_id, model_variant, timestamp,
                                                     created_by_user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (encounter_id, worker_name, int(quality_result["quality_pass"]),
                      quality_result.get("reason"), dr_class, triage, confidence,
                      triage_confidence, severity, severity_confidence,
                      quality_result.get("metrics", {}).get("laplacian_variance"),
                      quality_result.get("metrics", {}).get("mean_brightness"),
                      gradcam_url,
                      patient_id, model_variant, timestamp, created_by_user_id),
            )
            conn.commit()
            conn.close()
            return encounter_id
        except sqlite3.IntegrityError:
            continue

    conn.close()
    raise RuntimeError("Failed to generate a unique encounter_id after 5 attempts")


def get_history(limit: int = 50) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM encounters ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def encounter_is_monitor(encounter_id: str) -> bool:
    conn = _connect()
    row = conn.execute(
        """SELECT triage FROM encounters
           WHERE encounter_id = ? AND model_variant = 'integrated'""",
        (encounter_id,),
    ).fetchone()
    conn.close()
    return row is not None and row[0] == "Monitor"


def save_follow_up_plan(
    patient_id: str,
    contact_method: str,
    contact_value: str,
    consent_given: bool,
    return_date: str,
    encounter_id: str | None = None,
    created_by_user_id: str | None = None,
) -> dict:
    follow_up_id = str(uuid.uuid4())
    timestamp = _utc_now()
    conn = _connect()
    conn.execute(
        """INSERT INTO follow_up_plans
           (follow_up_id, encounter_id, patient_id, contact_method,
            contact_value, consent_given, created_by_user_id, return_date, status,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?)""",
        (follow_up_id, encounter_id, patient_id, contact_method,
         contact_value, int(consent_given), created_by_user_id, return_date,
         timestamp, timestamp),
    )
    conn.commit()
    conn.close()
    return get_follow_up_plan(follow_up_id)


def get_follow_up_plans(user: dict) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    query = """SELECT * FROM follow_up_plans
           ORDER BY CASE
                        WHEN status = 'overdue' THEN 0
                        WHEN status = 'scheduled' AND return_date < date('now') THEN 0
                        ELSE 1
                    END,
                    return_date ASC"""
    if not user["is_admin"]:
        query = query.replace(
            "ORDER BY",
            "WHERE created_by_user_id = ? ORDER BY",
        )
        rows = conn.execute(query, (user["user_id"],)).fetchall()
    else:
        rows = conn.execute(query).fetchall()
    conn.close()
    return [_follow_up_dict(row) for row in rows]


def get_follow_up_plan(follow_up_id: str) -> dict | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM follow_up_plans WHERE follow_up_id = ?",
        (follow_up_id,),
    ).fetchone()
    conn.close()
    return _follow_up_dict(row) if row else None


def get_follow_up_plan_for_user(follow_up_id: str, user: dict) -> dict | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    if user["is_admin"]:
        row = conn.execute(
            "SELECT * FROM follow_up_plans WHERE follow_up_id = ?",
            (follow_up_id,),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT * FROM follow_up_plans
               WHERE follow_up_id = ? AND created_by_user_id = ?""",
            (follow_up_id, user["user_id"]),
        ).fetchone()
    conn.close()
    return _follow_up_dict(row) if row else None


def get_due_email_reminders() -> list[dict]:
    """
    Return scheduled, consented email plans due for one of two reminders.

    The first reminder is due seven days before the return date. The second
    reminder is due on the return date if the patient has not returned.
    "Today" is the Philippine calendar date (UTC+8), not the UTC date.
    """
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM follow_up_plans
           WHERE contact_method = 'Email'
             AND consent_given = 1
             AND status = 'scheduled'
             AND (
                 (
                     reminder_count = 0
                     AND date(return_date, '-7 days')
                         <= date('now', '+8 hours')
                 )
                 OR (
                     reminder_count = 1
                     AND return_date <= date('now', '+8 hours')
                 )
             )"""
    ).fetchall()
    conn.close()
    return [_follow_up_dict(row) for row in rows]


def record_reminder_sent(follow_up_id: str) -> None:
    conn = _connect()
    conn.execute(
        """UPDATE follow_up_plans
           SET reminder_count = reminder_count + 1,
               last_reminder_sent_at = ?
           WHERE follow_up_id = ?""",
        (_utc_now(), follow_up_id),
    )
    conn.commit()
    conn.close()


def _follow_up_dict(row: sqlite3.Row) -> dict:
    result = dict(row)
    result["consent_given"] = bool(result["consent_given"])
    if (
        result["status"] == "scheduled"
        and result["return_date"] < datetime.now(PH_TZ).date().isoformat()
    ):
        result["status"] = "overdue"
    return result


def update_follow_up_plan(
    follow_up_id: str, updates: dict, user: dict
) -> dict | None:
    if not updates:
        return get_follow_up_plan_for_user(follow_up_id, user)

    if "return_date" in updates:
        # A new date restarts the reminder cycle (7 days before, then on the day).
        updates["reminder_count"] = 0
        updates["last_reminder_sent_at"] = None

    updates["updated_at"] = _utc_now()
    assignments = ", ".join(f"{key} = ?" for key in updates)
    values = [*updates.values(), follow_up_id]
    conn = _connect()
    where = "follow_up_id = ?"
    if not user["is_admin"]:
        where += " AND created_by_user_id = ?"
        values.append(user["user_id"])
    cursor = conn.execute(
        f"UPDATE follow_up_plans SET {assignments} WHERE {where}",
        values,
    )
    conn.commit()
    conn.close()
    return get_follow_up_plan_for_user(follow_up_id, user) if cursor.rowcount else None