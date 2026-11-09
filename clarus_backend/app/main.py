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
import hmac
import os
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Literal

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, Header, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, EmailStr, Field

from app.email_service import GmailAuthError, send_followup_reminder
from app.quality import check_image_quality
from app.classifier import classify_image, load_model, model_info
from app.triage import map_to_triage
from app.database import (
    count_users, create_referral, create_referral_note, create_user,
    encounter_is_monitor, get_due_email_reminders, get_encounter,
    get_encounter_for_gradcam, get_follow_up_plan_for_user,
    gradcam_is_authorized,
    create_notification, get_follow_up_plans, get_history, get_history_for_user,
    get_notifications, get_referral, get_referrals_for_user, get_referral_notes,
    get_user, init_db, list_users, mark_notification_read, record_reminder_sent,
    save_encounter, save_follow_up_plan, set_user_active, update_follow_up_plan,
    update_referral_status, write_audit_log,
)
from app.auth import (
    authenticate_user, create_access_token, get_current_user, hash_password,
    require_admin, require_roles,
)
from app.user_model import UserRole

app = FastAPI(title="Clarus API")
scheduler = BackgroundScheduler(timezone="Asia/Manila")
_failed_logins: dict[str, list[float]] = {}
_LOGIN_WINDOW_SECONDS = 300
_LOGIN_ATTEMPT_LIMIT = 5

GRADCAM_DIR = Path(__file__).resolve().parent.parent / "static" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CLARUS_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


ContactMethod = Literal["Email"]
FollowUpStatus = Literal["scheduled", "completed", "rescheduled", "missed"]
ReferralStatus = Literal[
    "assigned", "in_review", "reviewed", "completed", "cancelled"
]


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


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class BootstrapAdminRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8)


class CreateUserRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8)
    role: UserRole


class ReferralCreate(BaseModel):
    encounter_id: str = Field(min_length=1, max_length=120)
    assigned_to_user_id: str = Field(min_length=1, max_length=120)
    consent_given: bool


class ReferralStatusUpdate(BaseModel):
    status: ReferralStatus


class ReferralNoteCreate(BaseModel):
    note: str = Field(min_length=1, max_length=5000)
    recommendation: str | None = Field(default=None, max_length=5000)


class ReportAuditRequest(BaseModel):
    encounter_id: str = Field(min_length=1, max_length=120)
    recipient: str | None = Field(default=None, max_length=320)


class UserActiveUpdate(BaseModel):
    is_active: bool


@app.post("/auth/bootstrap-admin", status_code=201)
def bootstrap_admin(
    request: BootstrapAdminRequest,
    bootstrap_secret: str | None = Header(default=None, alias="X-Bootstrap-Secret"),
):
    configured_secret = os.getenv("CLARUS_BOOTSTRAP_SECRET")
    if not configured_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLARUS_BOOTSTRAP_SECRET is not configured.",
        )
    if bootstrap_secret is None or not hmac.compare_digest(
        bootstrap_secret, configured_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bootstrap secret.",
        )
    if count_users() != 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The initial administrator has already been created.",
        )

    try:
        user = create_user(
            full_name=request.full_name,
            email=str(request.email),
            password_hash=hash_password(request.password),
            role="specialist",
            is_admin=True,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        token = create_access_token(user)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.post("/auth/login")
def login(request: LoginRequest):
    key = str(request.email).lower()
    now = time.monotonic()
    attempts = [
        timestamp
        for timestamp in _failed_logins.get(key, [])
        if now - timestamp < _LOGIN_WINDOW_SECONDS
    ]
    if len(attempts) >= _LOGIN_ATTEMPT_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Try again later.",
        )
    user = authenticate_user(str(request.email), request.password)
    if user is None:
        attempts.append(now)
        _failed_logins[key] = attempts
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _failed_logins.pop(key, None)
    try:
        token = create_access_token(user)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.get("/auth/me")
def current_user(current_user: dict = Depends(get_current_user)):
    return current_user


@app.get("/auth/users")
def users(current_user: dict = Depends(require_admin)):
    return list_users()


@app.get("/auth/clinical-users")
def clinical_users(current_user: dict = Depends(get_current_user)):
    return [
        user
        for user in list_users()
        if user["is_active"]
        and user["role"] in ("specialist", "ophthalmologist_optometrist")
    ]


@app.post("/auth/users", status_code=201)
def create_managed_user(
    request: CreateUserRequest,
    current_user: dict = Depends(require_admin),
):
    try:
        user = create_user(
            full_name=request.full_name,
            email=str(request.email),
            password_hash=hash_password(request.password),
            role=request.role,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    write_audit_log(
        current_user["user_id"], "user_created", "user", user["user_id"]
    )
    return user


@app.patch("/auth/users/{user_id}/active")
def update_managed_user_status(
    user_id: str,
    request: UserActiveUpdate,
    current_user: dict = Depends(require_admin),
):
    if user_id == current_user["user_id"] and not request.is_active:
        raise HTTPException(status_code=422, detail="The active administrator cannot be deactivated.")
    user = set_user_active(user_id, request.is_active)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    write_audit_log(current_user["user_id"], "user_status_changed", "user", user_id)
    return user


@app.post("/referrals", status_code=201)
def create_referral_record(
    request: ReferralCreate,
    current_user: dict = Depends(
        require_roles(
            "health_worker",
            "specialist",
            "ophthalmologist_optometrist",
        )
    ),
):
    if not request.consent_given:
        raise HTTPException(
            status_code=422,
            detail="Patient consent is required before creating a referral.",
        )

    encounter = get_encounter(request.encounter_id)
    if encounter is None:
        raise HTTPException(status_code=404, detail="Encounter not found.")
    if encounter["triage"] != "Refer":
        raise HTTPException(
            status_code=422,
            detail="Only Refer encounters can be sent to a clinician.",
        )

    assignee = get_user(request.assigned_to_user_id)
    if assignee is None or not assignee["is_active"]:
        raise HTTPException(status_code=404, detail="Assigned user not found.")
    if assignee["role"] not in (
        "specialist",
        "ophthalmologist_optometrist",
    ):
        raise HTTPException(
            status_code=422,
            detail="Referrals can only be assigned to clinical users.",
        )

    try:
        referral = create_referral(
            encounter_id=encounter["encounter_id"],
            patient_id=encounter["patient_id"] or "Unknown",
            created_by_user_id=current_user["user_id"],
            assigned_to_user_id=assignee["user_id"],
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    write_audit_log(
        current_user["user_id"], "referral_created", "referral", referral["referral_id"]
    )
    create_notification(
        assignee["user_id"],
        "referral_assigned",
        f"New referral assigned for patient {referral['patient_id']}.",
        referral["referral_id"],
    )
    return referral


@app.get("/referrals")
def referrals(current_user: dict = Depends(get_current_user)):
    result = get_referrals_for_user(current_user)
    write_audit_log(current_user["user_id"], "referrals_listed", "referral", None)
    return result


@app.get("/referrals/{referral_id}")
def referral_detail(
    referral_id: str,
    current_user: dict = Depends(get_current_user),
):
    referral = _get_authorized_referral(referral_id, current_user)
    write_audit_log(current_user["user_id"], "referral_viewed", "referral", referral_id)
    return referral


@app.patch("/referrals/{referral_id}")
def update_referral(
    referral_id: str,
    request: ReferralStatusUpdate,
    current_user: dict = Depends(get_current_user),
):
    referral = _get_authorized_referral(referral_id, current_user)
    if (
        current_user["role"]
        not in ("specialist", "ophthalmologist_optometrist")
        and not current_user["is_admin"]
    ):
        raise HTTPException(
            status_code=403,
            detail="Only clinical users can update referral status.",
        )
    updated = update_referral_status(referral_id, request.status)
    write_audit_log(
        current_user["user_id"],
        "referral_status_changed",
        "referral",
        referral_id,
    )
    if updated is not None:
        create_notification(
            _get_notification_recipient(referral, current_user["user_id"]),
            "referral_status_changed",
            f"Referral {referral_id} status changed to {request.status}.",
            referral_id,
        )
    return updated


@app.get("/referrals/{referral_id}/notes")
def referral_notes(
    referral_id: str,
    current_user: dict = Depends(get_current_user),
):
    _get_authorized_referral(referral_id, current_user)
    write_audit_log(
        current_user["user_id"], "referral_notes_viewed", "referral", referral_id
    )
    return get_referral_notes(referral_id)


@app.post("/referrals/{referral_id}/notes", status_code=201)
def add_referral_note(
    referral_id: str,
    request: ReferralNoteCreate,
    current_user: dict = Depends(get_current_user),
):
    _get_authorized_referral(referral_id, current_user)
    if current_user["role"] not in (
        "specialist",
        "ophthalmologist_optometrist",
    ):
        raise HTTPException(
            status_code=403,
            detail="Only clinical users can add referral notes.",
        )
    note = create_referral_note(
        referral_id=referral_id,
        author_user_id=current_user["user_id"],
        note=request.note.strip(),
        recommendation=request.recommendation.strip()
        if request.recommendation
        else None,
    )
    write_audit_log(
        current_user["user_id"], "referral_note_added", "referral", referral_id
    )
    return note


def _get_authorized_referral(referral_id: str, current_user: dict) -> dict:
    referral = get_referral(referral_id)
    if referral is None:
        raise HTTPException(status_code=404, detail="Referral not found.")
    if (
        not current_user["is_admin"]
        and referral["assigned_to_user_id"] != current_user["user_id"]
        and referral["created_by_user_id"] != current_user["user_id"]
    ):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this referral.",
        )
    return referral


def _get_notification_recipient(referral: dict, actor_user_id: str) -> str:
    if referral["created_by_user_id"] != actor_user_id:
        return referral["created_by_user_id"]
    return referral["assigned_to_user_id"]


@app.get("/notifications")
def notifications(current_user: dict = Depends(get_current_user)):
    return get_notifications(current_user["user_id"])


@app.post("/notifications/{notification_id}/read")
def read_notification(
    notification_id: str,
    current_user: dict = Depends(get_current_user),
):
    if not mark_notification_read(notification_id, current_user["user_id"]):
        raise HTTPException(status_code=404, detail="Notification not found.")
    return {"ok": True}


@app.post("/audit/report-share")
def audit_report_share(
    request: ReportAuditRequest,
    current_user: dict = Depends(get_current_user),
):
    write_audit_log(
        current_user["user_id"],
        "report_shared",
        "encounter",
        request.encounter_id,
    )
    return {"ok": True}


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
def get_model_info(current_user: dict = Depends(get_current_user)):
    return model_info()


@app.get("/gradcam/{filename}")
def get_gradcam(
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    path = (GRADCAM_DIR / filename).resolve()
    if path.parent != GRADCAM_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Visualization not found.")
    encounter = get_encounter_for_gradcam(filename)
    if encounter is None:
        raise HTTPException(status_code=404, detail="Visualization not found.")
    if not gradcam_is_authorized(filename, current_user):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this visualization.",
        )
    write_audit_log(current_user["user_id"], "gradcam_viewed", "gradcam", filename)
    return FileResponse(path, media_type="image/png")


@app.post("/predict")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    worker_name: str = Form(default="Unknown"),
    patient_id: str = Form(default="Unknown"),
    current_user: dict = Depends(require_roles("health_worker", "specialist")),
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
        blur_score=quality_result.get("metrics", {}).get("laplacian_variance"),
        illumination_score=quality_result.get("metrics", {}).get("mean_brightness"),
        created_by_user_id=current_user["user_id"],
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
        blur_score=quality_result.get("metrics", {}).get("laplacian_variance"),
        illumination_score=quality_result.get("metrics", {}).get("mean_brightness"),
        created_by_user_id=current_user["user_id"],
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
def history(current_user: dict = Depends(get_current_user)):
    # Only shows the user-facing "integrated" results — baseline records
    # exist purely for research analysis, not for a health worker to see.
    return get_history_for_user(current_user)


@app.get("/analytics/summary")
def analytics_summary(current_user: dict = Depends(require_admin)):
    rows = get_history_for_user(current_user, limit=10000)
    total = len(rows)
    counts = {}
    for row in rows:
        triage = row.get("triage") or "Unknown"
        counts[triage] = counts.get(triage, 0) + 1
    return {"total_screenings": total, "triage_counts": counts}


@app.get("/comparison")
def comparison(current_user: dict = Depends(require_admin)):
    """
    Research/evaluation endpoint — not called by the Flutter app.
    Returns all records (both variants) for offline analysis, e.g. in a
    notebook, when producing the Chapter 4 comparative results.
    """
    return get_history(limit=1000)


@app.post("/follow-ups", status_code=201)
def create_follow_up(
    plan: FollowUpCreate,
    current_user: dict = Depends(get_current_user),
):
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
        created_by_user_id=current_user["user_id"],
    )


@app.get("/follow-ups")
def follow_ups(current_user: dict = Depends(get_current_user)):
    return get_follow_up_plans(current_user)


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
def trigger_reminders(current_user: dict = Depends(require_admin)):
    """Manually trigger the reminder check for testing and demonstrations."""
    return send_due_reminders()


@app.patch("/follow-ups/{follow_up_id}")
def update_follow_up(
    follow_up_id: str,
    plan: FollowUpUpdate,
    current_user: dict = Depends(get_current_user),
):
    if get_follow_up_plan_for_user(follow_up_id, current_user) is None:
        raise HTTPException(status_code=404, detail="Follow-up plan not found.")
    updates = plan.model_dump(exclude_none=True)
    if "return_date" in updates:
        if updates["return_date"] < date.today():
            raise HTTPException(
                status_code=422,
                detail="The return date must be today or in the future.",
            )
        updates["return_date"] = updates["return_date"].isoformat()

    updated = update_follow_up_plan(follow_up_id, updates, current_user)
    if updated is None:
        raise HTTPException(status_code=404, detail="Follow-up plan not found.")
    return updated


def _color_for(triage: str) -> str:
    return {"Normal": "22c55e", "Monitor": "eab308", "Refer": "ef4444"}.get(triage, "999999")