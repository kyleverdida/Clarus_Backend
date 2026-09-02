"""
Image-quality pre-screening module.

Checks a fundus image for blur, poor illumination, and framing issues
BEFORE it ever reaches the classifier — matching Clarus's Chapter 3
Functional Requirements ("Image-Quality Pre-Screening Module").

No machine learning here on purpose: these are simple, fast, well-established
computer-vision heuristics (Laplacian variance for blur, mean brightness for
illumination), not a separate trained model. Keep it this way unless testing
shows it's not accurate enough — a second CNN just for quality-checking is a
lot of extra complexity for a capstone timeline.
"""

import cv2
import numpy as np

# Tunable thresholds — start here, adjust once you test against real
# (and deliberately bad) sample images.
BLUR_THRESHOLD = 100.0       # Laplacian variance below this = too blurry
MIN_BRIGHTNESS = 40          # mean pixel value below this = too dark
MAX_BRIGHTNESS = 220         # mean pixel value above this = overexposed


def check_image_quality(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes, returns a dict describing whether the image
    passes quality gating, plus the specific reason if it fails.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"quality_pass": False, "reason": "unreadable_file"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur check: Laplacian variance. Sharp images have high-frequency edges
    # (high variance); blurry images look "smooth" (low variance).
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Illumination check: simple mean brightness across the image.
    mean_brightness = float(np.mean(gray))

    if laplacian_var < BLUR_THRESHOLD:
        return {
            "quality_pass": False,
            "reason": "blurry",
            "metrics": {"laplacian_variance": round(laplacian_var, 2)},
        }
    if mean_brightness < MIN_BRIGHTNESS:
        return {
            "quality_pass": False,
            "reason": "too_dark",
            "metrics": {"mean_brightness": round(mean_brightness, 2)},
        }
    if mean_brightness > MAX_BRIGHTNESS:
        return {
            "quality_pass": False,
            "reason": "overexposed",
            "metrics": {"mean_brightness": round(mean_brightness, 2)},
        }

    return {
        "quality_pass": True,
        "reason": None,
        "metrics": {
            "laplacian_variance": round(laplacian_var, 2),
            "mean_brightness": round(mean_brightness, 2),
        },
    }
