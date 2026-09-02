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

# Allows the Flutter app (running on emulator/device) to call this API
# during development. Tighten this before any real deployment.
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

    # Step 1: quality gate — matches Functional Requirements Item 1
    quality_result = check_image_quality(image_bytes)

    if not quality_result["quality_pass"]:
        # Still return a valid response shape — the Flutter app checks
        # `quality_pass` and shows the recapture dialog itself.
        return {
            "quality_pass": False,
            "triage": None,
            "confidence": 0.0,
            "gradcam_url": None,
            "encounter_id": None,
            "quality_reason": quality_result["reason"],
        }

    # Step 2: classification — STUBBED until Person A's model is ready
    classification = classify_image(image_bytes)

    # Step 3: referral-triage mapping — pending clinical review, see triage.py
    triage = map_to_triage(classification["dr_class"])

    # Step 4: Grad-CAM — STUBBED with a placeholder image until a real
    # trained model exists to generate real heatmaps against.
        # .png forces raster output — placehold.co defaults to SVG, which
    # Flutter's Image.network cannot decode (no built-in SVG support).
        # Correct placehold.co syntax confirmed against their own docs:
    # format goes AFTER the colors as its own segment, not attached to size.
    gradcam_url = f"https://placehold.co/400x400/{_color_for(triage)}/white/png?text={triage}"
    # Step 5: persist the encounter record
    encounter_id = save_encounter(
        worker_name=worker_name,
        quality_result=quality_result,
        dr_class=classification["dr_class"],
        triage=triage,
        confidence=classification["confidence"],
        gradcam_url=gradcam_url,
    )

    return {
        "quality_pass": True,
        "triage": triage,
        "confidence": classification["confidence"],
        "gradcam_url": gradcam_url,
        "encounter_id": encounter_id,
    }


@app.get("/history")
def history():
    return get_history()


def _color_for(triage: str) -> str:
    return {"Normal": "22c55e", "Monitor": "eab308", "Refer": "ef4444"}.get(triage, "999999")
