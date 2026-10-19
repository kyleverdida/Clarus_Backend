"""
Clarus backend — FastAPI application.

Matches the API contract already built into the Flutter app's
`screening_result.dart`:

  POST /predict
    in:  image file (multipart/form-data)
    out: { quality_pass, triage, confidence, gradcam_url, encounter_id }

  GET /history
    out: list of past encounters

Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import base64
import uuid
from datetime import date
from pathlib import Path
from typing import Literal

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, EmailStr, Field

from app.email_service import GmailAuthError, send_followup_reminder
from app.quality import check_image_quality
from app.classifier import classify_image, load_model, model_info
from app.triage import map_to_triage
from app.database import (
    encounter_is_monitor, get_due_email_reminders, get_follow_up_plans, get_history,
    init_db, record_reminder_sent, save_encounter, save_follow_up_plan, update_follow_up_plan,
)

app = FastAPI(title="Clarus API")
scheduler = BackgroundScheduler(timezone="Asia/Manila")

GRADCAM_DIR = Path(__file__).resolve().parent.parent / "static" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/gradcam", StaticFiles(directory=GRADCAM_DIR), name="gradcam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


ContactMethod = Literal["Email"]
FollowUpStatus = Literal["scheduled", "completed", "rescheduled", "missed"]


class FollowUpCreate(BaseModel):
    patient_id: str = Field(min_length=1, max_length=120)
    encounter_id: str | None = Field(default=None, max_length=120)
    contact_method: ContactMethod
    contact_value: EmailStr  # reminders are email-only; reject anything else
    consent_given: bool
    return_date: date


class FollowUpUpdate(BaseModel):
    status: FollowUpStatus | None = None
    return_date: date | None = None


@app.on_event("startup")
def on_startup():
    init_db()
    load_model()
    scheduler.add_job(
        send_due_reminders,
        "cron",
        hour=9,
        minute=0,
        id="daily_due_reminders",
        replace_existing=True,
        misfire_grace_time=3600,  # still run if the server was down at 9:00
    )
    scheduler.start()


@app.on_event("shutdown")
def on_shutdown():
    if scheduler.running:
        scheduler.shutdown(wait=False)


@app.get("/")
def root():
    return {"status": "Clarus API running"}


@app.get("/model-info")
def get_model_info():
    return model_info()


@app.post("/predict")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    worker_name: str = Form(default="Unknown"),
    patient_id: str = Form(default="Unknown"),
):
    image_bytes = await image.read()

    quality_result = check_image_quality(image_bytes)

    if not quality_result["quality_pass"]:
        return {
            "quality_pass": False,
            "triage": None,
            "confidence": 0.0,
            "triage_confidence": None,
            "severity": None,
            "severity_confidence": None,
            "gradcam_url": None,
            "encounter_id": None,
            "quality_reason": quality_result["reason"],
        }

    integrated_classification = await run_in_threadpool(classify_image, image_bytes, "integrated")
    if integrated_classification["status"] not in ("ok", "ok_low_quality"):
        return {
            "quality_pass": False,
            "triage": None,
            "confidence": 0.0,
            "triage_confidence": None,
            "severity": None,
            "severity_confidence": None,
            "gradcam_url": None,
            "encounter_id": None,
            "quality_reason": integrated_classification["reason"],
        }
    baseline_classification = await run_in_threadpool(classify_image, image_bytes, "baseline")

    triage = integrated_classification.get("triage") or map_to_triage(integrated_classification["dr_class"])
    baseline_triage = baseline_classification.get("triage") or map_to_triage(baseline_classification["dr_class"])

    gradcam_name = f"{uuid.uuid4().hex}.png"
    (GRADCAM_DIR / gradcam_name).write_bytes(base64.b64decode(integrated_classification["gradcam_png_base64"]))
    gradcam_url = f"{str(request.base_url).rstrip('/')}/gradcam/{gradcam_name}"

    encounter_id = save_encounter(
        worker_name=worker_name,
        quality_result=quality_result,
        dr_class=integrated_classification["dr_class"],
        triage=triage,
        confidence=integrated_classification["confidence"],
        triage_confidence=integrated_classification["confidence"],
        severity=integrated_classification["dr_class"],
        severity_confidence=integrated_classification["confidence"],
        gradcam_url=gradcam_url,
        model_variant="integrated",
        patient_id=patient_id.strip() or "Unknown",
    )
    save_encounter(
        worker_name=worker_name,
        quality_result=quality_result,
        dr_class=baseline_classification["dr_class"],
        triage=baseline_triage,
        confidence=baseline_classification["confidence"],
        triage_confidence=baseline_classification["confidence"],
        severity=baseline_classification["dr_class"],
        severity_confidence=baseline_classification["confidence"],
        gradcam_url=None,  # baseline has no explainability layer by design
        model_variant="baseline",
        patient_id=patient_id.strip() or "Unknown",
    )

    return {
        "quality_pass": True,
        "triage": triage,
        "confidence": integrated_classification["confidence"],
        "triage_confidence": integrated_classification["confidence"],
        "severity": integrated_classification["dr_class"],
        "severity_confidence": integrated_classification["confidence"],
        "gradcam_url": gradcam_url,
        "encounter_id": encounter_id,
        "needs_clinician_review": integrated_classification["needs_clinician_review"],
    }


@app.get("/history")
def history():
    # Only shows the user-facing "integrated" results — baseline records
    # exist purely for research analysis, not for a health worker to see.
    return [row for row in get_history() if row.get("model_variant") != "baseline"]


@app.get("/comparison")
def comparison():
    """
    Research/evaluation endpoint — not called by the Flutter app.
    Returns all records (both variants) for offline analysis, e.g. in a
    notebook, when producing the Chapter 4 comparative results.
    """
    return get_history(limit=1000)


@app.post("/follow-ups", status_code=201)
def create_follow_up(plan: FollowUpCreate):
    if not plan.consent_given:
        raise HTTPException(
            status_code=422,
            detail="Patient consent is required before reminders can be scheduled.",
        )
    if plan.return_date < date.today():
        raise HTTPException(
            status_code=422,
            detail="The return date must be today or in the future.",
        )
    if plan.encounter_id and not encounter_is_monitor(plan.encounter_id):
        raise HTTPException(
            status_code=422,
            detail="Follow-up plans can only be attached to Monitor encounters.",
        )

    return save_follow_up_plan(
        patient_id=plan.patient_id.strip(),
        encounter_id=plan.encounter_id,
        contact_method=plan.contact_method,
        contact_value=plan.contact_value.strip(),
        consent_given=plan.consent_given,
        return_date=plan.return_date.isoformat(),
    )


@app.get("/follow-ups")
def follow_ups():
    return get_follow_up_plans()


def send_due_reminders() -> dict:
    due = get_due_email_reminders()
    sent, failed = 0, 0
    for plan in due:
        try:
            success = send_followup_reminder(
                plan["contact_value"], plan["return_date"]
            )
        except GmailAuthError as exc:
            # Gmail itself is unusable; every remaining send would fail too.
            print(f"[reminders] {exc}")
            return {
                "checked": len(due),
                "sent": sent,
                "failed": len(due) - sent,
                "error": str(exc),
            }
        if success:
            record_reminder_sent(plan["follow_up_id"])
            sent += 1
        else:
            failed += 1
    return {"checked": len(due), "sent": sent, "failed": failed}


@app.post("/reminders/send")
def trigger_reminders():
    """Manually trigger the reminder check for testing and demonstrations."""
    return send_due_reminders()


@app.patch("/follow-ups/{follow_up_id}")
def update_follow_up(follow_up_id: str, plan: FollowUpUpdate):
    updates = plan.model_dump(exclude_none=True)
    if "return_date" in updates:
        if updates["return_date"] < date.today():
            raise HTTPException(
                status_code=422,
                detail="The return date must be today or in the future.",
            )
        updates["return_date"] = updates["return_date"].isoformat()

    updated = update_follow_up_plan(follow_up_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="Follow-up plan not found.")
    return updated


def _color_for(triage: str) -> str:
    return {"Normal": "22c55e", "Monitor": "eab308", "Refer": "ef4444"}.get(triage, "999999")