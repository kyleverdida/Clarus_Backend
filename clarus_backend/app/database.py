"""
Database module — SQLite, one table, matching the encounter-record
fields described in Chapter 3's Data Models section.

SQLite (not a full Postgres/MySQL server) is the right call for a
capstone-scale project: zero setup, one file on disk, plenty fast enough
for a screening tool that isn't handling concurrent hospital-wide traffic.
"""

import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = "clarus.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
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
            gradcam_url TEXT,
            patient_id TEXT,
            model_variant TEXT DEFAULT 'integrated',
            timestamp TEXT
        )
    """)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(encounters)")
    }
    if "patient_id" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN patient_id TEXT")
    if "triage_confidence" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN triage_confidence REAL")
    if "severity" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN severity TEXT")
    if "severity_confidence" not in columns:
        conn.execute("ALTER TABLE encounters ADD COLUMN severity_confidence REAL")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_encounters_timestamp
        ON encounters(timestamp DESC)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS follow_up_plans (
            follow_up_id TEXT PRIMARY KEY,
            encounter_id TEXT,
            patient_id TEXT NOT NULL,
            contact_method TEXT NOT NULL,
            contact_value TEXT NOT NULL,
            consent_given INTEGER NOT NULL,
            return_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'scheduled',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_follow_up_return_date
        ON follow_up_plans(return_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_follow_up_status
        ON follow_up_plans(status)
    """)
    conn.commit()
    conn.close()


def save_encounter(worker_name: str, quality_result: dict, dr_class: str,
                    triage: str, confidence: float, gradcam_url: str,
                    triage_confidence: float | None = None,
                    severity: str | None = None,
                    severity_confidence: float | None = None,
                    model_variant: str = "integrated",
                    patient_id: str | None = None) -> str:
    """
    model_variant distinguishes which pipeline produced this prediction —
    "integrated" (the full Clarus workflow) or "baseline" (a plain CNN
    with none of the four added components), matching the Comparative
    Evaluation Module described in Chapter 3. Both stubbed identically
    for now until Person A's real models exist for each.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)

    for attempt in range(5):
        encounter_id = str(uuid.uuid4())[:8]
        try:
            conn.execute(
                """INSERT INTO encounters
                   (encounter_id, worker_name, quality_pass, quality_reason,
                                                     dr_class, triage, confidence, triage_confidence,
                                                     severity, severity_confidence, gradcam_url, patient_id,
                                        model_variant, timestamp)
                                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (encounter_id, worker_name, int(quality_result["quality_pass"]),
                      quality_result.get("reason"), dr_class, triage, confidence,
                      triage_confidence, severity, severity_confidence, gradcam_url,
                      patient_id, model_variant, timestamp),
            )
            conn.commit()
            conn.close()
            return encounter_id
        except sqlite3.IntegrityError:
            continue

    conn.close()
    raise RuntimeError("Failed to generate a unique encounter_id after 5 attempts")


def get_history(limit: int = 50) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM encounters ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def encounter_is_monitor(encounter_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
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
) -> dict:
    follow_up_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO follow_up_plans
           (follow_up_id, encounter_id, patient_id, contact_method,
            contact_value, consent_given, return_date, status,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?)""",
        (follow_up_id, encounter_id, patient_id, contact_method,
         contact_value, int(consent_given), return_date, timestamp, timestamp),
    )
    conn.commit()
    conn.close()
    return get_follow_up_plan(follow_up_id)


def get_follow_up_plans() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM follow_up_plans
           ORDER BY CASE
                        WHEN status = 'overdue' THEN 0
                        WHEN status = 'scheduled' AND return_date < date('now') THEN 0
                        ELSE 1
                    END,
                    return_date ASC"""
    ).fetchall()
    conn.close()
    return [_follow_up_dict(row) for row in rows]


def get_follow_up_plan(follow_up_id: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM follow_up_plans WHERE follow_up_id = ?",
        (follow_up_id,),
    ).fetchone()
    conn.close()
    return _follow_up_dict(row) if row else None


def _follow_up_dict(row: sqlite3.Row) -> dict:
    result = dict(row)
    result["consent_given"] = bool(result["consent_given"])
    if (
        result["status"] == "scheduled"
        and result["return_date"] < datetime.now(timezone.utc).date().isoformat()
    ):
        result["status"] = "overdue"
    return result


def update_follow_up_plan(follow_up_id: str, updates: dict) -> dict | None:
    if not updates:
        return get_follow_up_plan(follow_up_id)

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    assignments = ", ".join(f"{key} = ?" for key in updates)
    values = [*updates.values(), follow_up_id]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        f"UPDATE follow_up_plans SET {assignments} WHERE follow_up_id = ?",
        values,
    )
    conn.commit()
    conn.close()
    return get_follow_up_plan(follow_up_id) if cursor.rowcount else None