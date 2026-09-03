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

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.quality import check_image_quality
from app.classifier import classify_image
from app.triage import map_to_triage
from app.database import init_db, save_encounter, get_history

app = FastAPI(title="Clarus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"status": "Clarus API running"}


@app.post("/predict")
async def predict(image: UploadFile = File(...), worker_name: str = Form(default="Unknown")):
    image_bytes = await image.read()

    quality_result = check_image_quality(image_bytes)

    if not quality_result["quality_pass"]:
        return {
            "quality_pass": False,
            "triage": None,
            "confidence": 0.0,
            "gradcam_url": None,
            "encounter_id": None,
            "quality_reason": quality_result["reason"],
        }

    # Runs BOTH variants for the Comparative Evaluation Module (Objective 4):
    # the "integrated" result is what's shown to the user; "baseline" is
    # computed and logged purely for later research comparison, never
    # shown in the app. This keeps the Flutter contract completely
    # unchanged — no risk of another frontend/backend sync issue.
    integrated_classification = classify_image(image_bytes, variant="integrated")
    baseline_classification = classify_image(image_bytes, variant="baseline")

    triage = map_to_triage(integrated_classification["dr_class"])
    baseline_triage = map_to_triage(baseline_classification["dr_class"])

    gradcam_url = f"https://placehold.co/400x400/{_color_for(triage)}/white/png?text={triage}"

    encounter_id = save_encounter(
        worker_name=worker_name,
        quality_result=quality_result,
        dr_class=integrated_classification["dr_class"],
        triage=triage,
        confidence=integrated_classification["confidence"],
        gradcam_url=gradcam_url,
        model_variant="integrated",
    )
    save_encounter(
        worker_name=worker_name,
        quality_result=quality_result,
        dr_class=baseline_classification["dr_class"],
        triage=baseline_triage,
        confidence=baseline_classification["confidence"],
        gradcam_url=None,  # baseline has no explainability layer by design
        model_variant="baseline",
    )

    return {
        "quality_pass": True,
        "triage": triage,
        "confidence": integrated_classification["confidence"],
        "gradcam_url": gradcam_url,
        "encounter_id": encounter_id,
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


def _color_for(triage: str) -> str:
    return {"Normal": "22c55e", "Monitor": "eab308", "Refer": "ef4444"}.get(triage, "999999")