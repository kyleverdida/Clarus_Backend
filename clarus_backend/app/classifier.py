"""
CNN classification module - the REAL Clarus model (replaces the earlier random stub).

The trained package lives in <project>/model (the "For_App_Ready_Code_Of_CNN_Model" folder from the training notebook: clarus_core.py,
clarus_final.json, the model folders). clarus_core.ClarusPredictor does everything for one image in a single pass - background removal,
enhancement, BOTH networks, calibration, Grad-CAM - so the first call per image does the work and the second variant call of the same
request is answered from a small cache.

classify_image(image_bytes, variant) keeps the old name and the old keys (dr_class, confidence, variant) and adds the new ones.
variant "integrated" = the model decision shown to the user (the notebook's FINAL choice); "baseline" = the Baseline CNN, logged for the study.
"""
import hashlib
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path

DR_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
MODEL_DIR = Path(os.getenv("CLARUS_MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))

_predictor = None
_load_lock = threading.Lock()
_cache, _cache_lock = OrderedDict(), threading.Lock()


def load_model():
    """Load the networks ONCE (called at server start-up so the first request is not slow)."""
    global _predictor
    with _load_lock:
        if _predictor is None:
            if not (MODEL_DIR / "clarus_final.json").exists():
                raise RuntimeError(f"Clarus model package not found in {MODEL_DIR}. Copy the 'For_App_Ready_Code_Of_CNN_Model' folder there "
                                   "or set the CLARUS_MODEL_DIR environment variable.")
            sys.path.insert(0, str(MODEL_DIR))
            from clarus_core import ClarusPredictor
            _predictor = ClarusPredictor.from_final(MODEL_DIR)
    return _predictor


def model_info() -> dict:
    return load_model().info()


def analyse(image_bytes: bytes) -> dict:
    """Full clarus_core result for one image (cached by content hash for the integrated + baseline calls of one request)."""
    key = hashlib.sha256(image_bytes).hexdigest()
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    result = load_model().predict(image_bytes)
    with _cache_lock:
        _cache[key] = result
        while len(_cache) > 4:
            _cache.popitem(last=False)
    return result


def classify_image(image_bytes: bytes, variant: str = "integrated") -> dict:
    """
    -> usable image: {"status", "dr_class", "confidence", "variant", "triage", "referable_probability", "needs_clinician_review",
                      "gradcam_png_base64" (integrated only), "core_version"}
       unusable image (no retina found / not an image): {"status": "ungradable_retake" | "invalid_image", "reason": "..."}  (no prediction)
    """
    r = analyse(image_bytes)
    if r["status"] not in ("ok", "ok_low_quality"):
        issues = r.get("quality", {}).get("issues") or []
        return {"status": r["status"], "reason": issues[0] if issues else ("unreadable_file" if r["status"] == "invalid_image" else "retina_not_detected"),
                "message": r.get("message"), "variant": variant}
    block = r["final"] if variant == "integrated" else r.get("baseline", r["final"])
    out = {"status": r["status"], "dr_class": block["predicted_label"], "confidence": block["confidence"], "variant": variant,
           "triage": block["triage"], "referable_probability": block["referable_probability"],
           "needs_clinician_review": bool(r["needs_clinician_review"]) if variant == "integrated" else None, "core_version": r["core_version"]}
    if variant == "integrated":
        out["gradcam_png_base64"] = block["explanation"]["overlay_png_base64"]
    return out