# Clarus Backend

## What this is

FastAPI backend for `Clarus_App`. It implements the screening pipeline:
image-quality gating → classification → referral-triage mapping → Grad-CAM
result URL → database logging.

The backend is ready to connect to the Flutter frontend, but the classifier
and Grad-CAM are still placeholders until the trained model is integrated.

## Setup

From this directory (`clarus_backend`):

```
pip install -r requirements.txt
```

## Run it

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

FastAPI provides interactive API documentation at
`http://localhost:8000/docs`.

## API contract

### `POST /predict`

Accepts `multipart/form-data` with:

- `image`: the retinal image file
- `worker_name`: optional health-worker name; defaults to `Unknown`

Successful response:

```json
{
	"quality_pass": true,
	"triage": "Normal",
	"confidence": 0.91,
	"gradcam_url": "https://placehold.co/400x400/...",
	"encounter_id": "a1b2c3d4"
}
```

If the image fails quality checks, the response is still `200 OK` and has
`quality_pass: false`, `triage: null`, `confidence: 0.0`,
`gradcam_url: null`, `encounter_id: null`, and a `quality_reason` field. The
rejected image is not saved as an encounter.

### `GET /history`

Returns the user-facing integrated screening history. Baseline comparison
records are excluded from this response.

### `GET /comparison`

Returns integrated and baseline records for research and evaluation. This is
not used by `Clarus_App`.

### `GET /`

Returns a simple health response: `{ "status": "Clarus API running" }`.

## What's real vs. stubbed right now

| Component | Status |
|---|---|
| Image-quality check (`app/quality.py`) | **Real** — actual OpenCV blur/brightness detection, tested against a deliberately blurred image and correctly rejected it |
| Classifier (`app/classifier.py`) | **Stubbed** — returns simulated predictions. Replace `classify_image()` with real inference once the trained model exists. Keep the function signature and return shape identical. |
| Referral-triage mapping (`app/triage.py`) | **Real logic**, but the mapping itself is a proposed design pending clinical review — see the docstring and your own Chapter 1 Limitations |
| Grad-CAM | **Stubbed** — returns a placeholder image URL; real heatmaps require a trained model |
| Database (`app/database.py`) | **Real** — SQLite stores integrated and baseline records |

## Connecting `Clarus_App`

1. Start this backend with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
2. In `Clarus_App/lib/services/api_service.dart`, set `useMockApi = false`.
3. Use `http://10.0.2.2:8000` as the base URL on an Android emulator. Use
	`http://127.0.0.1:8000` for a Flutter desktop or web client running on the
	same computer.
4. Submit the image as the `image` multipart field. The response fields match
	the screening result model used by `Clarus_App`.

For a physical device, replace the host with the computer's local network IP
and make sure the device and computer are on the same network.

## Swapping in the real model (hand-off from Person A)

Open `app/classifier.py`. Replace the body of `classify_image()` with real
model loading + inference. The function must still:
- Accept raw image bytes
- Return a dict shaped exactly like `{"dr_class": "...", "confidence": 0.0-1.0}`
- Use a class name from `DR_CLASSES` that matches your Chapter 3 design

Everything downstream (triage mapping, database, API response) will keep
working unchanged, because it only depends on that return shape — not on
how the prediction was made.

## Current verification

```
Good image   -> 200 OK, quality_pass: true, triage assigned
Blurry image -> 200 OK, quality_pass: false, reason: "blurry"
History      -> returns integrated user-facing encounters only
```

The `/comparison` endpoint includes both model variants. Each accepted image
creates one integrated record for the app and one baseline record for research.
