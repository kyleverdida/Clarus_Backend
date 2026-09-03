"""
CNN classification module.

STUB VERSION — returns plausible fake predictions so the rest of the
pipeline (triage mapping, database, API, Flutter app) can be built and
tested end-to-end before Person A's trained model exists.

*** HAND-OFF POINT WITH PERSON A ***
When the real model is ready, replace the body of `classify_image()` with
actual inference — but keep the function signature and return shape
IDENTICAL, so nothing else in the backend needs to change.

Supports two variants for the Comparative Evaluation Module (Chapter 3):
  - "integrated": the full Clarus model, trained WITH class-imbalance
    handling (focal loss / targeted augmentation per Objective 3)
  - "baseline": a conventional CNN, trained WITHOUT imbalance handling —
    this is the comparison point for Objective 4

Right now both variants call the same stub logic and will look identical
in output. Once Person A has two actual trained model files (one per
variant), each branch below should load and run its own model instead of
sharing this stub.
"""

import random

DR_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]


def classify_image(image_bytes: bytes, variant: str = "integrated") -> dict:
    """
    Takes raw image bytes (already passed quality gating) plus which
    model variant to use, returns the underlying DR severity
    classification and confidence for that variant.

    variant: "integrated" or "baseline" — see module docstring.
    """
    if variant == "baseline":
        weights = [0.7, 0.15, 0.08, 0.05, 0.02]
    else:
        weights = [0.5, 0.25, 0.15, 0.07, 0.03]

    predicted_class = random.choices(DR_CLASSES, weights=weights, k=1)[0]
    confidence = round(random.uniform(0.72, 0.97), 4)

    return {
        "dr_class": predicted_class,
        "confidence": confidence,
        "variant": variant,
    }