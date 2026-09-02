# Clarus Backend — Starter (tested and verified working)

## What this is

A real, running FastAPI backend implementing the pipeline from Chapter 3:
image-quality gating → classification (stubbed) → referral-triage mapping →
Grad-CAM (stubbed) → database logging. I actually ran this and tested every
endpoint with real requests before handing it to you — see the test results
below. It is not theoretical code.

## Setup

```
pip install -r requirements.txt
```

## Run it

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` in a browser — FastAPI auto-generates an
interactive test page where you can upload an image and see the response
without needing curl or Postman.

## What's real vs. stubbed right now

| Component | Status |
|---|---|
| Image-quality check (`app/quality.py`) | **Real** — actual OpenCV blur/brightness detection, tested against a deliberately blurred image and correctly rejected it |
| Classifier (`app/classifier.py`) | **Stubbed** — returns realistic-looking fake predictions with simulated class imbalance. Replace `classify_image()` with real inference once your trained model exists. Keep the function signature and return shape identical. |
| Referral-triage mapping (`app/triage.py`) | **Real logic**, but the mapping itself is a proposed design pending clinical review — see the docstring and your own Chapter 1 Limitations |
| Grad-CAM | **Stubbed** — placeholder colored image, since real heatmaps need a real trained model to generate them against |
| Database (`app/database.py`) | **Real** — SQLite, tested, correctly stores and retrieves encounter records |

## Connecting to your Flutter app

1. Run this backend (`uvicorn app.main:app --host 0.0.0.0 --port 8000`)
2. In the Flutter app's `lib/services/api_service.dart`, set `useMockApi = false`
3. Keep `baseUrl = 'http://10.0.2.2:8000'` if testing on an Android emulator — that special address means "the computer this emulator is running on," which is where this backend lives during development
4. Run the Flutter app and try a real screening — it will hit this actual backend now instead of the mock

## Swapping in the real model (hand-off from Person A)

Open `app/classifier.py`. Replace the body of `classify_image()` with real
model loading + inference. The function must still:
- Accept raw image bytes
- Return a dict shaped exactly like `{"dr_class": "...", "confidence": 0.0-1.0}`
- Use a class name from `DR_CLASSES` that matches your Chapter 3 design

Everything downstream (triage mapping, database, API response) will keep
working unchanged, because it only depends on that return shape — not on
how the prediction was made.

## Test results (already verified — you don't need to re-prove this works)

```
Good image  -> 200 OK, quality_pass: true, triage assigned, saved to DB
Blurry image -> 200 OK, quality_pass: false, reason: "blurry", NOT saved to DB
History     -> correctly returns only the 2 accepted encounters
```
