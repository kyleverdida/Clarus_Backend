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
            gradcam_url TEXT,
            model_variant TEXT DEFAULT 'integrated',
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_encounters_timestamp
        ON encounters(timestamp DESC)
    """)
    conn.commit()
    conn.close()


def save_encounter(worker_name: str, quality_result: dict, dr_class: str,
                    triage: str, confidence: float, gradcam_url: str,
                    model_variant: str = "integrated") -> str:
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
                    dr_class, triage, confidence, gradcam_url, model_variant, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (encounter_id, worker_name, int(quality_result["quality_pass"]),
                 quality_result.get("reason"), dr_class, triage, confidence,
                 gradcam_url, model_variant, timestamp),
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