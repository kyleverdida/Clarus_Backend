"""
CNN classification module.

STUB VERSION — returns plausible fake predictions so the rest of the
pipeline (triage mapping, database, API, Flutter app) can be built and
tested end-to-end before Person A's trained model exists.

*** HAND-OFF POINT WITH PERSON A ***
When the real model is ready, replace the body of `classify_image()` with
actual inference — but keep the function signature and return shape
IDENTICAL, so nothing else in the backend needs to change. That's the
whole point of isolating this into one function.

Expected real implementation will likely:
  1. Load the trained model once at startup (not per-request — too slow)
  2. Preprocess the image (resize, normalize) to match training preprocessing
  3. Run inference, get class probabilities
  4. Return the predicted class + confidence in this same dict shape
"""

import random

# These must match whatever classes Person A's model is actually trained on.
# Standard 5-class ICDR DR severity scale — confirm this matches their
# training labels exactly before wiring in the real model.
DR_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]


def classify_image(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes (already passed quality gating), returns the
    underlying DR severity classification and confidence.

    STUB: picks a weighted-random class to simulate realistic class
    imbalance (mostly No_DR / Mild, occasionally Severe) rather than
    uniform randomness, so the rest of the app can be tested against
    a realistic-feeling distribution.
    """
    weights = [0.5, 0.25, 0.15, 0.07, 0.03]  # rough imbalance simulation
    predicted_class = random.choices(DR_CLASSES, weights=weights, k=1)[0]
    confidence = round(random.uniform(0.72, 0.97), 4)

    return {
        "dr_class": predicted_class,
        "confidence": confidence,
    }
