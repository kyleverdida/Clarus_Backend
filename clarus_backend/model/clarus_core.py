"""
clarus_core.py (v4.2) - the Clarus CNN core: background removal, image enhancement, EfficientNet-B0, Grad-CAM and the app predictor.

ONE module is imported by BOTH the training notebook and the app backend, so training and inference preprocess every
image in exactly the same way (a mismatch silently lowers accuracy).

----------------------------------------------------------------------------------------------------------------------
HOW AN UPLOADED FUNDUS IMAGE FLOWS THROUGH THIS MODULE
----------------------------------------------------------------------------------------------------------------------
  read_image        decode PNG/JPEG/bytes/array -> RGB, EXIF orientation applied, long side limited to MAX_SIDE
  find_retina       locate the circular retina, build a 0/1 mask (background removal)
  square_crop       crop to the retina and pad to a square, background = black
  quality_*         blur / brightness / framing checks (study Sec. 3) -> list of issues, never changes the image
  enhance           CNN INPUT: contrast enhancement inside the retina only (see "Enhancement pipelines" below)
  enhance_view      DISPLAY ONLY: natural-colour CLAHE view that the Grad-CAM heat is drawn on (the CNN never sees it)
  _Net.probs        EfficientNet-B0 -> 5 class probabilities (flip test-time augmentation + temperature calibration)
  triage            class -> Normal / Monitor / Refer (study mapping) or a probability threshold on P(Severe)+P(PDR)
  _Net.cam          Grad-CAM of the predicted class (where in the retina the evidence came from)
  gradcam_overlay   soft, semi-transparent heat + one outline, drawn on the enhanced retina
  ClarusPredictor.from_final   the APP ENTRY POINT: loads the "Final" folder written by the training notebook and returns
                    result["final"] = the decision to show (Integrated, Baseline or their Ensemble, chosen on VALIDATION data)

----------------------------------------------------------------------------------------------------------------------
THE MODEL (build_model)
----------------------------------------------------------------------------------------------------------------------
  EfficientNet-B0 pretrained on ImageNet (transfer learning) -> GlobalAveragePooling -> Dropout -> Dense(256, swish)
  -> Dropout -> Dense(5, softmax).  Supervised multi-class classification of the 5 APTOS grades:
  No_DR, Mild, Moderate, Severe, Proliferative_DR.  Input is 256x256 RGB in the range 0-255 (EfficientNet rescales inside).
  Baseline  = cross-entropy loss.      Integrated = focal loss (gamma 2) + targeted augmentation of Severe/PDR (study).

----------------------------------------------------------------------------------------------------------------------
ENHANCEMENT PIPELINES (the CNN-input preprocessing; the name is stored in preprocess_config.json next to every model)
----------------------------------------------------------------------------------------------------------------------
  bgremoval+benGraham-v1                    legacy: masked Ben Graham, fixed gain 4.  Models trained with it keep working:
                                            the predictor reads the pipeline name from the model folder and uses it.
  bgremoval+denoise+adaptiveBenGraham-v2    optional, NOT validated (compare with v1 first): edge-preserving denoise -> masked Ben Graham
                                            with a per-image adaptive gain -> smooth (tanh) limiting instead of hard clipping.
  A model must be used with the pipeline it was trained with; this module refuses/auto-selects accordingly.

App usage (load ONCE at server start-up, then call predict() for every request - it is thread-safe):
    from clarus_core import ClarusPredictor
    clarus = ClarusPredictor.from_final("For_App_Ready_Code_Of_CNN_Model")   # once, at server start-up (this folder holds clarus_core.py, clarus_final.json and the models)
    result = clarus.predict(image_bytes_or_path_or_rgb_array)       # JSON-serialisable dict; show result["final"]; never raises on a bad upload
    result["status"] -> "ok" | "ok_low_quality" | "ungradable_retake" (no retina / blocked) | "invalid_image" (not an image / too small)
    clarus.info()    -> what is loaded (for a /health or /version endpoint and for logging)
    # (the old form ClarusPredictor(integrated_path, baseline_path) still works; result then has no "final" block)

CLI:  python clarus_core.py fundus.png --final For_App_Ready_Code_Of_CNN_Model --save-dir out/
      python clarus_core.py fundus.png --model Integrated.keras [--baseline Baseline.keras] --save-dir out/

Decision support only - not a diagnosis.
"""
import argparse, base64, json, threading, time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# ======================================================================================================================
# SETTINGS  (training notebook + app must agree; the ones that change the CNN input are saved to preprocess_config.json)
# ======================================================================================================================
CORE_VERSION = "4.3"
FINAL_FILE = "clarus_final.json"            # written by the training notebook into <Model_Last>/For_App_Ready_Code_Of_CNN_Model
IMG_SIZE = (256, 256)                       # CNN input size (the notebook may change it; models record it in preprocess_config.json)
LEGACY_PIPELINE = "bgremoval+benGraham-v1"
V2_PIPELINE = "bgremoval+denoise+adaptiveBenGraham-v2"
PIPELINES = (LEGACY_PIPELINE, V2_PIPELINE)
PIPELINE = LEGACY_PIPELINE                  # pipeline for NEW training (the notebook sets it); _apply_preprocess_config() switches it to match a loaded model
REMOVE_BACKGROUND = True                    # False = keep the full frame (only for the background-removal ablation)
MASK_MODE = "circle"                        # shape of the kept region: "circle" = the retinal disc (fitted circle, dark/shaded rim is kept) | "hull" = the old convex hull of the bright blob.
                                            # Saved in preprocess_config.json; models trained before v4.3 have no entry and keep "hull".
CIRCLE_GROW = 0.12                          # the circle may add at most this fraction of the retina size beyond the detected blob (repairs dark rim, never reaches far into the backdrop)
NUM_CLASSES = 5
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
REFER_CLASSES = [3, 4]                      # Severe + Proliferative_DR -> Refer (the study's proposed, not-yet-clinically-validated mapping)
TRIAGE = np.array(["Normal", "Monitor", "Monitor", "Refer", "Refer"])
CAM_LAYER = "top_activation"                # Grad-CAM layer inside EfficientNet-B0 (8x8 at 256 px). "block6a_expand_activation" = 16x16 (finer, noisier)
CAM_TARGET = "logit"                       # Grad-CAM score: "logit" = class logit (standard Grad-CAM, no saturation) | "prob" = softmax probability (maps flatten when the model is very sure)
MAX_SIDE, DISPLAY = 1024, 512               # working size of a decoded upload / size of the display images (Grad-CAM, enhanced view)
MIN_SIDE = 128                              # smaller uploads are rejected as invalid_image (no usable retinal detail)
MIN_CONFIDENCE = 0.50                       # below this (calibrated) confidence a result is flagged for clinician review

# v2 enhancement constants (heuristics - compare v1 vs v2 on your validation set before relying on them)
V2_DENOISE = dict(d=5, sigmaColor=20, sigmaSpace=3)   # bilateral filter: smooths grain but keeps vessel / lesion edges
V2_TARGET_SPREAD, V2_GAIN_RANGE = 24.0, (2.5, 6.0)    # high-pass spread (grey levels) the adaptive gain aims for / its allowed range

# How the Grad-CAM heat is drawn (display only - nothing here touches the CNN). Change with ci.OVERLAY.update(...)
OVERLAY = dict(style="soft",       # "soft" = gentle palette + per-pixel transparency | "jet" = the old rainbow, uniform blend
               alpha=0.70,         # strongest opacity of the heat (0-1). Lower = retina more visible
               floor=0.30,         # map values below this fraction of the peak are NOT coloured at all (retina untouched)
               contour=0.70,       # draw one thin outline around the strongest area (map >= this fraction of peak); None/False = no outline
               base="enhanced")    # picture under the heat: "enhanced" (CLAHE natural colour) | "original"

# Heuristic quality limits (study: blur, poor illumination, framing). Starting values - calibrate on smartphone-lens images.
QUALITY = dict(min_sharpness=15.0, brightness=(25, 220), min_area=0.25, max_center_offset=0.20, min_red_ratio=1.10)
DISCLAIMER = "Clinical decision-support output only. Not a diagnosis; confirm with a qualified clinician."


# ======================================================================================================================
# 1. IMAGE LOADING
# ======================================================================================================================
def read_image(image):
    """Path / raw bytes / numpy array -> RGB uint8, long side limited to MAX_SIDE (INTER_AREA = clean down-scaling).
    cv2.imdecode applies the EXIF orientation, so phone photos are upright. Raises ValueError for a non-image."""
    if isinstance(image, (str, Path)):
        image = np.fromfile(str(image), np.uint8)             # np.fromfile (not cv2.imread) also works with non-ASCII Drive paths
    if isinstance(image, (bytes, bytearray)):
        image = np.frombuffer(bytes(image), np.uint8)
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a path, bytes, or numpy array")
    if image.ndim == 1:                                       # encoded file bytes
        if image.size == 0:
            raise ValueError("Input is empty.")
        bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Input is not a valid PNG/JPEG image.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)            # OpenCV is BGR, everything else here is RGB
    else:                                                     # already a pixel array (grey or colour)
        rgb = (image if image.ndim == 3 else np.stack([image] * 3, -1))[..., :3].astype(np.uint8)
    s = MAX_SIDE / max(rgb.shape[:2])
    return cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb


# ======================================================================================================================
# 2. BACKGROUND REMOVAL  (classical computer vision - no learning involved)
# ======================================================================================================================
def find_retina(rgb):
    """Locate the retinal disc. Returns (mask uint8 {0,1} same size as rgb, info dict).

    Logic: blur -> threshold -> clean up -> keep the largest blob -> fill its convex hull -> trim the rim.
      * The threshold depends on the backdrop, which is auto-detected from the image border:
          dark border            -> luminance threshold (APTOS / EyePACS style black frame)
          neutral grey/white     -> colourfulness (max(R,G,B) - min(R,G,B)) with Otsu: a fundus is strongly coloured, the backdrop is not
          coloured backdrop      -> Otsu on grey; the side that contains the image centre is the retina
      * Opening/closing removes specks and small holes. The CONVEX HULL means dark lesions or haemorrhages inside the retina can
        never be cut out. A final erosion trims the ragged rim / vignette.
      * If nothing plausible is found the whole frame is kept (info['fallback'] = True) and the quality check flags it."""
    h, w = rgb.shape[:2]
    full = np.ones((h, w), np.uint8)
    if not REMOVE_BACKGROUND:
        return full, dict(background_removed=False, fallback=False, retina_area_fraction=1.0, center_offset=0.0)
    blur = cv2.GaussianBlur(rgb, (0, 0), max(h, w) / 200)
    g = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    chroma = (blur.max(-1) - blur.min(-1)).astype(np.uint8)
    ring = lambda a: np.concatenate([a[:6].ravel(), a[-6:].ravel(), a[:, :6].ravel(), a[:, -6:].ravel()])   # the 6-px frame of the image
    otsu = lambda a: cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    if np.median(ring(g)) < 40:                               # dark backdrop. circle mode: threshold just above the backdrop's own noise level (shaded retina stays in); hull mode: the old rule
        m = g > (np.clip(np.percentile(ring(g), 75) + 6, 8, 25) if MASK_MODE == "circle" else np.clip(0.5 * otsu(g), 8, 25))
    elif np.median(ring(chroma)) < 30:                        # neutral light/grey backdrop
        m = chroma > max(otsu(chroma), 20)
    else:                                                     # coloured backdrop
        t = otsu(g)
        m = g > t if np.median(g[h // 4: 3 * h // 4, w // 4: 3 * w // 4]) > t else g <= t
    m = m.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((max(w, h) // 60) | 1,) * 2)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fail = dict(background_removed=False, fallback=True, retina_area_fraction=1.0, center_offset=0.0)
    if not cnts:
        return full, fail
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if MASK_MODE == "circle":
        mask = _disc_from_blob(mask)
    area = float(mask.mean())
    if area < 0.03:                                           # noise only; a small-but-real retina is flagged by the quality check instead
        return full, fail
    bx, by, bw, bh = cv2.boundingRect(mask)
    e = max(1, int(0.012 * max(bw, bh)))
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1,) * 2))
    ys, xs = np.nonzero(mask)
    offset = float(np.hypot(xs.mean() - w / 2, ys.mean() - h / 2) / max(h, w))      # how far the retina is from the frame centre
    return mask, dict(background_removed=area < 0.98, fallback=False, retina_area_fraction=round(area, 4),
                      center_offset=round(offset, 4))


def _disc_from_blob(blob):
    """Turn the detected blob into the retinal DISC. A fundus photo is a circle, but dark / shaded parts of the rim fall under the
    threshold and bite notches out of the blob. Fit a circle to the blob outline (least squares, ignoring the points that lie on the image
    border: a retina cut off by the frame is straight there, not round) and add the fitted disc to the blob - but only within CIRCLE_GROW of the
    retina size, so a flat-sided or partly hidden retina is never inflated into the backdrop. The result is always the blob plus repaired rim."""
    h, w = blob.shape
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    p = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    p = p[(p[:, 0] > 3) & (p[:, 0] < w - 4) & (p[:, 1] > 3) & (p[:, 1] < h - 4)]       # drop the points where the frame cuts the retina
    if len(p) < 30:
        return blob
    q = p
    for _ in range(8):                                                                  # robust fit: notches lie INSIDE the true circle, so refit on the outer points only
        A = np.c_[2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))]                            # Kasa fit: x^2 + y^2 = 2ax + 2by + c
        (a, b, c), *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
        r = float(np.sqrt(max(c + a * a + b * b, 0)))
        keep = np.hypot(p[:, 0] - a, p[:, 1] - b) >= r * 0.97                           # points on / outside the circle (at most 3 % inside)
        if keep.sum() < 30 or keep.sum() == len(q): break
        q = p[keep]
    x, y, bw, bh = cv2.boundingRect(blob)
    if not (0.4 * max(bw, bh) <= 2 * r <= 1.4 * max(bw, bh)):                           # implausible fit (not a round retina): keep the blob
        return blob
    disc = np.zeros_like(blob); cv2.circle(disc, (int(round(a)), int(round(b))), int(round(r)), 1, -1, cv2.LINE_8)
    g = 2 * int(CIRCLE_GROW * max(bw, bh)) + 1
    reach = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    return (blob | (disc & reach)).astype(np.uint8)


def square_crop(rgb, mask):
    """Crop to the retina's bounding box, zero everything outside the mask, centre it in a square. -> (retina_rgb, mask), both side x side.
    A square keeps the retina circular when it is resized to the CNN input (no aspect-ratio distortion)."""
    x, y, bw, bh = cv2.boundingRect(mask)
    side = max(bw, bh)
    r, m = np.zeros((side, side, 3), np.uint8), np.zeros((side, side), np.uint8)
    oy, ox = (side - bh) // 2, (side - bw) // 2
    r[oy:oy + bh, ox:ox + bw] = (rgb * mask[..., None])[y:y + bh, x:x + bw]
    m[oy:oy + bh, ox:ox + bw] = mask[y:y + bh, x:x + bw]
    return r, m


# ======================================================================================================================
# 3. ENHANCEMENT
# ======================================================================================================================
def _resize_pair(retina, mask):
    """Resize retina + mask to the CNN input size. The mask is eroded by one pixel so rim pixels that were blended with the
    black background are dropped."""
    img = cv2.resize(retina, IMG_SIZE[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.erode(cv2.resize(mask, IMG_SIZE[::-1], interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8)).astype(np.float32)
    return img, m


def _masked_blur(img, m, sigma):
    """Gaussian blur that only averages pixels INSIDE the mask (normalised convolution). A plain blur would mix in the black
    background and create a bright ring at the disc edge after the subtraction below."""
    return cv2.GaussianBlur(img * m[..., None], (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma)[..., None] + 1e-6)


def _enhance_v1(retina, mask):
    """LEGACY pipeline: Ben Graham contrast enhancement out = 4*(img - local_mean) + 128 inside the retina (masked blur)."""
    img, m = _resize_pair(retina, mask)
    out = np.clip(4 * (img - _masked_blur(img, m, IMG_SIZE[0] / 30)) + 128, 0, 255)
    out[m == 0] = 128
    return out.astype(np.uint8)


def _enhance_v2(retina, mask):
    """NEW pipeline - Ben Graham with three refinements. Why each step:
      1. Bilateral denoise (edge-preserving). The Ben Graham gain multiplies EVERYTHING that differs from the local average, grain and
         JPEG noise included. Smoothing flat areas first (but not edges) stops that noise being boosted while vessels and lesions stay sharp.
      2. High-pass: img - local_mean (masked Gaussian, sigma = size/30) removes uneven illumination and the fundus' colour cast and keeps
         the fine structure (vessels, microaneurysms, haemorrhages, exudates).
      3. Adaptive gain: the legacy fixed x4 over-exposes contrasty images and under-exposes flat ones. The gain is set per image so the
         robust spread (MAD) of the high-pass hits V2_TARGET_SPREAD, limited to V2_GAIN_RANGE. Dim phone photos get more, harsh ones less.
      4. Soft limiting: 128 + 127*tanh(v/127) behaves like a straight line for normal values but rolls off smoothly instead of
         clipping bright exudates / dark haemorrhages flat at 0/255 (hard clipping destroys their internal detail).
    Background pixels are neutral grey 128, exactly like v1."""
    img, m = _resize_pair(retina, mask)
    den = cv2.bilateralFilter(img.astype(np.uint8), **V2_DENOISE).astype(np.float32)
    hp = den - _masked_blur(den, m, IMG_SIZE[0] / 30)
    vals = hp[m > 0]
    spread = 1.4826 * np.median(np.abs(vals - np.median(vals))) if vals.size else 0.0         # robust std (MAD)
    gain = float(np.clip(V2_TARGET_SPREAD / max(spread, 1e-3), *V2_GAIN_RANGE)) if vals.size else 4.0
    out = 128 + 127 * np.tanh(gain * hp / 127)
    out[m == 0] = 128
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance(retina, mask):
    """CNN input for the active PIPELINE -> IMG_SIZE uint8 RGB (background = 128). The ONLY place the pipelines differ."""
    return (_enhance_v1 if PIPELINE == LEGACY_PIPELINE else _enhance_v2)(retina, mask)


ben_graham = enhance                                          # name used by earlier versions of the notebook


def enhance_view(retina_rgb, mask):
    """DISPLAY-ONLY natural-colour enhancement for the picture the Grad-CAM is drawn on (the CNN never sees it).
    The CNN input is grey and noisy to the human eye, so heat drawn on it is hard to read. Here: CLAHE (contrast-limited adaptive
    histogram equalisation, clip 2, 8x8 tiles) on the L channel of LAB, blended 70/30 with the original lightness so it stays natural,
    then a mild unsharp mask. The background is filled with the retina's mean lightness while CLAHE runs, so it does not bias the tiles
    at the disc edge, and is zeroed again afterwards. Colours (a/b channels) are untouched."""
    inside = mask > 0
    if not inside.any():
        return retina_rgb
    lab = cv2.cvtColor(retina_rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].copy()
    L[~inside] = int(L[inside].mean())
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    lab[..., 0] = np.clip(0.7 * eq + 0.3 * lab[..., 0], 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = cv2.addWeighted(rgb, 1.35, cv2.GaussianBlur(rgb, (0, 0), max(retina_rgb.shape[:2]) / 80), -0.35, 0)
    return rgb * mask[..., None].astype(np.uint8)


def prepare(image, full=False):
    """Decode + background removal + enhancement.
    full=False (training, fast): {'x': IMG_SIZE uint8 CNN input, 'info': background-removal info}
    full=True  (inference): adds 'orig', 'retina' (native square), 'retina_disp'/'mask_disp' (DISPLAY square),
                            'view_disp' (enhanced natural-colour retina), 'rgba' (transparent background), 'rgba_view'."""
    rgb = read_image(image)
    mask, info = find_retina(rgb)
    retina, msq = square_crop(rgb, mask)
    out = {"x": enhance(retina, msq), "info": info}
    if full:
        d = cv2.resize(retina, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA)
        md = cv2.resize(msq, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
        view = enhance_view(d, md)
        alpha = lambda mm: np.uint8(255 * cv2.GaussianBlur(mm.astype(np.float32), (0, 0), 1.5))      # soft edge for the transparent PNGs
        out.update(orig=rgb, retina=retina, retina_disp=d, mask_disp=md, view_disp=view,
                   rgba=np.dstack([retina, alpha(msq)]), rgba_view=np.dstack([view, alpha(md)]))
    return out


def dhash64(retina):
    """64-bit difference hash of the background-removed retina (hex). Near-identical images (re-encoded / re-scaled copies) differ in few
    bits. Used by the notebook to find train/test leakage."""
    g = cv2.resize(cv2.cvtColor(retina, cv2.COLOR_RGB2GRAY), (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return format(int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).ravel()), 2), "016x")


# ======================================================================================================================
# 4. QUALITY PRE-SCREEN  (study: blur, illumination, framing).  Rule-based, thresholds in QUALITY
# ======================================================================================================================
def quality_issues(m):
    """m: dict with fallback, sharpness, brightness, red_green_ratio, retina_area_fraction, center_offset -> list of issue names."""
    q, issues = QUALITY, []
    if m["fallback"]: issues.append("retina_not_detected")
    if m["sharpness"] < q["min_sharpness"]: issues.append("blurry")
    if m["brightness"] < q["brightness"][0]: issues.append("too_dark")
    if m["brightness"] > q["brightness"][1]: issues.append("overexposed")
    if not m["fallback"] and m["retina_area_fraction"] < q["min_area"]: issues.append("retina_too_small_in_frame")
    if m["center_offset"] > q["max_center_offset"]: issues.append("retina_off_center")
    if m["red_green_ratio"] < q["min_red_ratio"]: issues.append("unusual_colour_for_fundus")
    return issues


def quality_metrics(P):
    """Numbers behind the quality check, measured inside the retina only (sharpness = variance of the Laplacian)."""
    d, m, info = P["retina_disp"], P["mask_disp"], P["info"]
    inner = cv2.erode(m, np.ones((21, 21), np.uint8)) > 0     # ignore the disc rim (a strong edge, not sharpness)
    gray = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
    use = inner if inner.any() else m > 0
    return {"fallback": bool(info["fallback"]), "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F)[use].var()), 1),
            "brightness": round(float(gray[use].mean()), 1), "red_green_ratio": round(float(d[..., 0][use].mean() / (d[..., 1][use].mean() + 1)), 2),
            "retina_area_fraction": info["retina_area_fraction"], "center_offset": info["center_offset"]}


def assess_quality(P):
    """-> {'passed': bool, 'issues': [...], 'metrics': {...}}"""
    m = quality_metrics(P); issues = quality_issues(m)
    return {"passed": not issues, "issues": issues, "metrics": {k: v for k, v in m.items() if k != "fallback"}}


def save_quality(folder):
    """Store the (calibrated) quality limits next to a model so the app applies the same ones."""
    json.dump({**QUALITY, "brightness": list(QUALITY["brightness"])}, open(Path(folder) / "quality.json", "w"), indent=2)


def apply_quality(q):
    """Update QUALITY from a dict (quality.json / clarus_final.json)."""
    QUALITY.update({k: (tuple(v) if k == "brightness" else v) for k, v in q.items()})


def prepare_for_training(image):
    """Training/evaluation cache entry: CNN input + background-removal info + quality metrics + a (legacy) hash column kept so existing caches
    stay valid; duplicate detection now compares the CNN inputs themselves (see the notebook, cell 5c)."""
    P = prepare(image, full=True)
    return {"x": P["x"], "info": {**P["info"], **quality_metrics(P), "dhash": dhash64(P["retina"])}}


# ======================================================================================================================
# 5. MODEL
# ======================================================================================================================
@keras.utils.register_keras_serializable(package="Clarus")
class FocalLoss(keras.losses.Loss):
    """Multiclass focal loss (the study's primary class-imbalance strategy):  FL = -(1 - p_t)^gamma * log(p_t)
    p_t is the probability the model gave the TRUE class. Easy, confident examples (p_t near 1) are down-weighted by (1-p_t)^gamma, so
    training concentrates on hard and rare examples (Severe / Proliferative) instead of the many easy No_DR images. gamma = 0 is plain
    cross-entropy. alpha=None -> study-exact; alpha=[5 weights] -> additionally class-weighted (optional extension)."""
    def __init__(self, gamma=2.0, alpha=None, name="clarus_focal_loss", **kw):
        super().__init__(name=name, **kw); self.gamma = float(gamma); self.alpha = None if alpha is None else [float(a) for a in alpha]
    def call(self, y_true, y_pred):
        y = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), NUM_CLASSES)
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        pt = tf.reduce_sum(y * p, -1)
        w = 1.0 if self.alpha is None else tf.reduce_sum(y * tf.constant(self.alpha, tf.float32), -1)
        return tf.reduce_mean(-w * tf.pow(1 - pt, self.gamma) * tf.math.log(pt))
    def get_config(self): return {**super().get_config(), "gamma": self.gamma, "alpha": self.alpha}


def build_model(name, pretrained=True, dropout=0.4):
    """EfficientNet-B0 (ImageNet) + GAP + Dropout + Dense(256) + Dropout + softmax(5). The backbone starts frozen (transfer learning stage 1);
    the notebook unfreezes its upper layers for stage 2. The output layer is float32 even under mixed precision (stable softmax)."""
    bb = EfficientNetB0(include_top=False, weights="imagenet" if pretrained else None, input_shape=(*IMG_SIZE, 3))
    bb.trainable = False
    inp = keras.Input((*IMG_SIZE, 3), name="fundus_image")
    x = bb(inp, training=False)                                # training=False keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop1")(x)
    x = layers.Dense(256, activation="swish", name="fc")(x)
    x = layers.Dropout(dropout / 2, name="drop2")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="dr_class")(x)
    return keras.Model(inp, out, name=name)


def save_preprocess_config(folder):
    """Write the preprocessing identity next to a model so inference can reproduce training exactly."""
    json.dump({"img_size": list(IMG_SIZE), "remove_background": REMOVE_BACKGROUND, "pipeline": PIPELINE, "mask": MASK_MODE},
              open(Path(folder) / "preprocess_config.json", "w"), indent=2)


def read_preprocess_config(model_path):
    """The model folder's preprocess_config.json as a dict, or None. A config without a 'pipeline' key predates v2 -> legacy."""
    f = Path(model_path).with_name("preprocess_config.json")
    if not f.exists():
        return None
    cfg = json.load(open(f)); cfg.setdefault("pipeline", LEGACY_PIPELINE); cfg.setdefault("mask", "hull")
    return cfg


def _apply_preprocess_config(model_path):
    """Make inference use exactly what the model was trained with: image size must match; background removal and the enhancement
    pipeline are switched to the model's. (Old models keep working with their old pipeline.)"""
    global REMOVE_BACKGROUND, PIPELINE, MASK_MODE
    cfg = read_preprocess_config(model_path)
    if cfg is None:
        return
    if tuple(cfg["img_size"]) != tuple(IMG_SIZE):
        raise ValueError(f"Model trained at {cfg['img_size']}, module IMG_SIZE is {IMG_SIZE}.")
    if cfg["pipeline"] not in PIPELINES:
        raise ValueError(f"Model trained with unknown pipeline {cfg['pipeline']!r}; known: {PIPELINES}. Use the clarus_core.py saved with that model.")
    PIPELINE, REMOVE_BACKGROUND, MASK_MODE = cfg["pipeline"], bool(cfg["remove_background"]), cfg["mask"]


# ======================================================================================================================
# 6. PREDICTION HELPERS: calibration, Grad-CAM, overlay
# ======================================================================================================================
def triage_of(p, thr=None):
    """Triage tier for one probability vector. thr=None -> the study mapping of the predicted class (Normal / Monitor / Refer).
    thr=number -> Refer when P(Severe)+P(PDR) >= thr, otherwise Normal or Monitor by the arg-max of the three milder classes."""
    if thr is None:
        return str(TRIAGE[int(np.argmax(p))])
    return "Refer" if float(np.sum(np.asarray(p)[REFER_CLASSES])) >= thr else ("Normal" if int(np.argmax(np.asarray(p)[:3])) == 0 else "Monitor")


class Ensemble:
    """Mean of several Keras models' probabilities behind the same predict() call, so the notebook's evaluation code treats the ensemble
    like a single model. (Calibration is applied AFTER averaging: temper(mean(raw), T_ensemble).)"""
    def __init__(self, models): self.models = list(models)
    def predict(self, x, verbose=0, **kw): return np.mean([m.predict(x, verbose=verbose, **kw) for m in self.models], 0)


def temper(p, T):
    """Temperature scaling of probabilities (== softmax(log p / T)). T > 1 softens over-confident outputs, T < 1 sharpens; T = 1 changes
    nothing. It never changes the arg-max class, only how confident the numbers are. T is fitted on the validation set (calibration.json)."""
    q = np.asarray(p, np.float64) ** (1.0 / T)
    return q / q.sum(-1, keepdims=True)


def cam_up(cam, size=None):
    """Upsample a small Grad-CAM grid (e.g. 8x8) to size x size (default CNN input) with bicubic interpolation, clipped to 0-1."""
    return np.clip(cv2.resize(np.asarray(cam, np.float32), (size or IMG_SIZE[0],) * 2, interpolation=cv2.INTER_CUBIC), 0, 1)


def cam_agreement(a, b, top=0.2):
    """How similar are two Grad-CAM maps (same shape)? Pearson correlation and IoU of their top-`top` (20 %) most salient pixels."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0
    ta, tb = a >= np.quantile(a, 1 - top), b >= np.quantile(b, 1 - top)
    return {"cam_pearson": round(r, 3), "cam_top20_iou": round(float((ta & tb).sum() / max((ta | tb).sum(), 1)), 3)}


class _Net:
    """float32 inference copy of a saved Clarus model + Grad-CAM (float32 keeps gradients stable even if training used mixed precision).
    Reads calibration.json (temperature, tuned referral threshold) from the model folder if the notebook wrote one."""
    def __init__(self, path, tta):
        self.tta, self.file = tta, Path(path).name
        f = Path(path).with_name("calibration.json")
        self.cal = json.load(open(f)) if f.exists() else {}
        self.T, self._cam = float(self.cal.get("temperature", 1.0)), {}
        old = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("float32")
        try:                                                   # rebuild under float32 and copy the weights across
            src = keras.models.load_model(str(path), compile=False)
            self.model = build_model(Path(path).stem, pretrained=False)
            self.model.set_weights(src.get_weights())
        finally:
            keras.mixed_precision.set_global_policy(old)
        self.bb = self.model.get_layer("efficientnetb0")
        self.head = self.model.layers[self.model.layers.index(self.bb) + 1:]

    def raw(self, batch):
        """[B,H,W,3] float 0-255 -> [B,5] UNcalibrated probabilities. Flip test-time augmentation: average the prediction of the image and
        of its horizontal mirror (a cheap ensemble that steadies the result)."""
        p = self.model(batch, training=False).numpy()
        if self.tta: p = (p + self.model(batch[:, :, ::-1], training=False).numpy()) / 2
        return p

    def probs(self, batch):
        """[B,H,W,3] float 0-255 -> [B,5] calibrated probabilities (temperature from calibration.json)."""
        return temper(self.raw(batch), self.T)

    def cam(self, batch, cls, layer=None):
        """Grad-CAM at `layer` (default CAM_LAYER) of the class logit (CAM_TARGET="logit", default) or the softmax probability ("prob").
        Logic: (1) take the feature maps A_k of a late convolutional layer; (2) differentiate the class score with respect to them and
        average the gradients over space -> one importance weight w_k per feature map; (3) CAM = ReLU(sum_k w_k * A_k): the places whose
        features push the score UP; (4) scale to 0-1. The logit (pre-softmax) is the default because softmax saturates and its gradients vanish
        for confident predictions. batch [B,H,W,3]; cls int or [B] ints -> [B,h,w] maps scaled to 0-1 per image."""
        layer, batch = layer or CAM_LAYER, tf.cast(batch, tf.float32)
        if layer not in self._cam:
            self._cam[layer] = keras.Model(self.bb.input, [self.bb.get_layer(layer).output, self.bb.output])
        cls = tf.constant(np.broadcast_to(np.asarray(cls, np.int32), (batch.shape[0],)).copy())
        with tf.GradientTape() as tape:
            tape.watch(batch)
            f, x = self._cam[layer](batch, training=False)
            for l in self.head[:-1]: x = l(x)
            z = tf.matmul(x, self.head[-1].kernel) + self.head[-1].bias            # pre-softmax logits
            score = tf.reduce_sum((tf.nn.softmax(z) if CAM_TARGET == "prob" else z) * tf.one_hot(cls, NUM_CLASSES), -1)
        w = tf.reduce_mean(tape.gradient(score, f), axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(f * w, -1))
        return (cam / tf.maximum(tf.reduce_max(cam, axis=(1, 2), keepdims=True), 1e-12)).numpy()


def _soft_palette():
    """256-colour palette for the soft overlay: pale yellow -> amber -> magenta. The fundus itself is orange-red, so the strongest evidence
    uses magenta (it stands out from the retina without the glare of pure red), and pale yellow marks weak evidence. The old JET rainbow
    painted the whole retina blue/green/red."""
    t, stops = np.linspace(0, 1, 256), np.array([0.0, 0.5, 1.0])
    cols = np.array([(255, 250, 150), (255, 196, 50), (200, 30, 110)], np.float32)
    return np.stack([np.interp(t, stops, cols[:, k]) for k in range(3)], 1).astype(np.uint8)


SOFT_PALETTE = _soft_palette()


def gradcam_overlay(P, cam, style=None, alpha=None, floor=None, contour=None, base=None):
    """Draw a Grad-CAM map on the background-removed retina. Defaults come from OVERLAY (override per call or via OVERLAY.update).
    style='soft' (default):
      * the 8x8 map is upsampled and lightly smoothed (removes the blocky look);
      * values below `floor` are not coloured at all, so most of the retina stays completely untouched;
      * above the floor, colour AND opacity grow with the evidence (opacity <= `alpha`): weak areas are a faint yellow wash, the strongest
        magenta, and you can still see vessels and lesions through it;
      * one thin dark+white outline marks the strongest area (map >= `contour`) so the region of interest stays visible even on a bright retina.
    style='jet': the old rainbow blend over the whole retina, strength = alpha (floor and contour are ignored), kept for comparison.
    Returns RGBA at DISPLAY x DISPLAY with a transparent background."""
    o = {**OVERLAY, **{k: v for k, v in dict(style=style, alpha=alpha, floor=floor, contour=contour, base=base).items() if v is not None}}
    m = P["mask_disp"]
    under = (P["view_disp"] if o["base"] == "enhanced" and "view_disp" in P else P["retina_disp"]).astype(np.float32)
    heat = cv2.resize(np.asarray(cam, np.float32), (DISPLAY, DISPLAY), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(cv2.GaussianBlur(heat, (0, 0), DISPLAY / 80), 0, None) * m
    heat = np.clip(heat / max(float(heat.max()), 1e-6), 0, 1)
    if o["style"] == "jet":                                    # legacy look
        col = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (under * (1 - o["alpha"]) + col * o["alpha"]) * m[..., None]         # uniform blend of strength alpha (floor / contour are not used by the rainbow)
    else:
        t = np.clip((heat - o["floor"]) / (1 - o["floor"]), 0, 1)                      # 0 below the floor, 1 at the peak
        a = (o["alpha"] * t ** 1.2)[..., None]                                          # opacity follows the evidence
        col = SOFT_PALETTE[np.uint8(255 * t)].astype(np.float32)
        rgb = (under * (1 - a) + col * a) * m[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if o["contour"] and o["style"] != "jet":
        cs, _ = cv2.findContours((heat >= o["contour"]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cs, -1, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.drawContours(rgb, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return np.dstack([rgb, m * 255]).astype(np.uint8)


def encode_png(a):
    """RGB / RGBA uint8 array -> base64 PNG string (JSON-friendly; the app decodes it to show the picture)."""
    a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA if a.shape[-1] == 4 else cv2.COLOR_RGB2BGR)
    return base64.b64encode(cv2.imencode(".png", a)[1].tobytes()).decode()


def decode_png(b64):
    """base64 PNG string -> RGB / RGBA array (inverse of encode_png)."""
    a = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA if a.shape[-1] == 4 else cv2.COLOR_BGR2RGB)


# ======================================================================================================================
# 7. THE APP PREDICTOR
# ======================================================================================================================
class ClarusPredictor:
    def __init__(self, integrated_path, baseline_path=None, tta=True, refer_threshold=None, quality_gate="warn", final=None):
        """Load the Integrated model and optionally the Baseline (both are always reported, the study logs both).
        final: the dict from clarus_final.json (use ClarusPredictor.from_final) -> adds result["final"], the decision the app shows.
        refer_threshold=None -> study mapping (triage of the predicted class). A number (e.g. 0.35) -> Refer when
        P(Severe)+P(PDR) >= that value (more sensitive to referable cases). "tuned" -> each model's own validation-tuned
        threshold from its calibration.json. quality_gate: 'warn' = predict and report issues | 'block' = refuse low-quality images."""
        assert quality_gate in ("warn", "block")
        _apply_preprocess_config(integrated_path)
        q = Path(integrated_path).with_name("quality.json")
        if q.exists(): apply_quality(json.load(open(q)))
        self.final = final
        if final:
            apply_quality(final.get("quality", {}))
            if (final.get("pipeline"), tuple(final.get("img_size", IMG_SIZE))) != (PIPELINE, tuple(IMG_SIZE)):
                raise ValueError(f"clarus_final.json expects pipeline {final.get('pipeline')!r} at {final.get('img_size')}, the models loaded use {PIPELINE!r} at {list(IMG_SIZE)}.")
            if final["choice"] != "Integrated" and not baseline_path:
                raise ValueError(f"Final choice is {final['choice']!r} but no Baseline model was found.")
        if baseline_path:
            b = read_preprocess_config(baseline_path)
            if b and b["pipeline"] != PIPELINE:
                raise ValueError(f"Baseline was trained with pipeline {b['pipeline']!r} but Integrated with {PIPELINE!r}. Both models share one "
                                 "preprocessed input, so they must use the same pipeline - retrain one of them.")
        self.main = _Net(integrated_path, tta)
        self.base = _Net(baseline_path, tta) if baseline_path else None
        self.refer_threshold, self.quality_gate, self.tta = refer_threshold, quality_gate, tta
        if refer_threshold == "tuned" and not all("refer_threshold" in n.cal for n in (self.main, self.base) if n):
            raise ValueError("refer_threshold='tuned' needs calibration.json (with refer_threshold) next to every model - run the calibration cell.")
        self._lock = threading.RLock()        # Keras models and the cached Grad-CAM sub-models are not safe to call from several threads at once
        blank = tf.zeros((1, *IMG_SIZE, 3))   # warm-up: the first TensorFlow call traces the graph (slow) - do it now, not on the first user's request
        for n in (self.main, self.base):
            if n: n.probs(blank); n.cam(blank, 0)

    def info(self):
        """JSON-serialisable description of what is loaded (for a /health or /version endpoint and for logging)."""
        net = lambda n: None if n is None else {"file": n.file, "temperature": n.T, "tuned_refer_threshold": n.cal.get("refer_threshold")}
        return {"core_version": CORE_VERSION, "pipeline": PIPELINE, "img_size": list(IMG_SIZE), "remove_background": REMOVE_BACKGROUND, "tta": self.tta,
                "quality_gate": self.quality_gate, "quality_limits": QUALITY, "min_confidence": MIN_CONFIDENCE, "cam_layer": CAM_LAYER,
                "final": None if not self.final else {k: self.final.get(k) for k in ("choice", "operating_point", "temperature", "refer_threshold", "recipe", "created")},
                "integrated": net(self.main), "baseline": net(self.base)}

    @classmethod
    def from_final(cls, final_dir, tta=True, quality_gate="warn"):
        """THE APP ENTRY POINT. final_dir = <Model_Last>/For_App_Ready_Code_Of_CNN_Model (holds clarus_final.json, clarus_core.py and the model folders). Member model
        paths inside clarus_final.json are relative to that same folder, so the folder alone can be copied to a server unchanged. Raises a clear error if a file is missing."""
        final_dir = Path(final_dir)
        cfg = json.load(open(final_dir / FINAL_FILE)); root = final_dir
        paths = {n: root / p for n, p in cfg["members"].items()}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing: raise FileNotFoundError(f"Model file(s) listed in {FINAL_FILE} not found: {missing}")
        return cls(paths["Integrated"], paths.get("Baseline"), tta=tta, quality_gate=quality_gate, final=cfg)

    def _final(self, P, batch, cams, quality, explain):
        """The decision the app shows: the validation-selected model (Integrated, Baseline or their Ensemble) at the configured operating point
        (study mapping, or the validation-tuned referral threshold). Ensemble = temper(mean of both models' raw probabilities, T_ensemble);
        its Grad-CAM is the mean of both models' maps for the ensemble's class."""
        f, ch = self.final, self.final["choice"]
        thr = f.get("refer_threshold") if f.get("operating_point") == "tuned" else None
        if ch == "Ensemble": p = temper(np.mean([n.raw(batch)[0] for n in (self.main, self.base)], 0), f["temperature"])
        else: p = (self.main if ch == "Integrated" else self.base).probs(batch)[0]
        cls = int(p.argmax())
        r = {"model": ch, "operating_point": f.get("operating_point", "study"), "predicted_class": cls, "predicted_label": CLASS_NAMES[cls],
             "confidence": round(float(p[cls]), 4), "probabilities": {n: round(float(v), 4) for n, v in zip(CLASS_NAMES, p)},
             "triage": triage_of(p, thr), "referable_probability": round(float(p[REFER_CLASSES].sum()), 4),
             "needs_clinician_review": bool(float(p[cls]) < MIN_CONFIDENCE or not quality["passed"])}
        if explain:
            if ch == "Ensemble":
                cam = np.mean([n.cam(batch, cls)[0] for n in (self.main, self.base)], 0); cam = cam / max(float(cam.max()), 1e-12)
            else: cam = cams[ch]
            r["explanation"] = {"method": "Grad-CAM" + (" (mean of both models)" if ch == "Ensemble" else ""), "layer": CAM_LAYER,
                                "target_class": CLASS_NAMES[cls], "overlay_png_base64": encode_png(gradcam_overlay(P, cam))}
        return r

    def _run(self, net, P, batch, explain):
        """One model's result: class, calibrated probabilities, triage tier and (optionally) its Grad-CAM overlay."""
        p = net.probs(batch)[0]; cls = int(p.argmax()); p_ref = float(p[REFER_CLASSES].sum())
        thr = net.cal["refer_threshold"] if self.refer_threshold == "tuned" else self.refer_threshold
        triage = triage_of(p, thr)
        r = {"model": net.file, "predicted_class": cls, "predicted_label": CLASS_NAMES[cls], "confidence": round(float(p[cls]), 4),
             "probabilities": {n: round(float(v), 4) for n, v in zip(CLASS_NAMES, p)},
             "triage": str(triage), "referable_probability": round(p_ref, 4)}
        cam = None
        if explain:
            cam = net.cam(batch, cls)[0]
            r["explanation"] = {"method": "Grad-CAM", "layer": CAM_LAYER, "target_class": CLASS_NAMES[cls],
                                "overlay_png_base64": encode_png(gradcam_overlay(P, cam))}
        return r, cam

    def predict(self, image, explain=True):
        """image: path | bytes | RGB array -> JSON-serialisable dict. Never raises for a bad upload - look at result["status"]:
          ok                 gradable image, prediction made
          ok_low_quality     prediction made, but the quality check found issues (needs_clinician_review is True)
          ungradable_retake  no retina found (or quality_gate='block' and the image failed the check): NO prediction, ask the user to retake
          invalid_image      not a decodable PNG/JPEG, or smaller than MIN_SIDE px: NO prediction"""
        t0, meta = time.time(), {"core_version": CORE_VERSION, "disclaimer": DISCLAIMER}
        try:
            rgb = read_image(image)
            if min(rgb.shape[:2]) < MIN_SIDE:
                raise ValueError(f"Image too small ({rgb.shape[1]}x{rgb.shape[0]} px); at least {MIN_SIDE} px per side is needed.")
        except (ValueError, TypeError, cv2.error) as e:       # whatever is wrong with the upload, the API answers instead of crashing
            return {"status": "invalid_image", "message": str(e), **meta}
        P = prepare(rgb, full=True)
        quality = assess_quality(P)
        if "retina_not_detected" in quality["issues"] or (self.quality_gate == "block" and not quality["passed"]):   # no retina = nothing to grade, whatever the gate
            return {"status": "ungradable_retake", "message": "No gradable retina found - please retake the photo.", "quality": quality, **meta}
        batch = tf.cast(P["x"], tf.float32)[None]
        with self._lock:
            r, cam = self._run(self.main, P, batch, explain)
            cam_b, rb = None, None
            if self.base is not None:
                rb, cam_b = self._run(self.base, P, batch, explain)    # logged for the comparative study
            fin = self._final(P, batch, {"Integrated": cam, "Baseline": cam_b}, quality, explain) if self.final else None
        out = {"status": "ok" if quality["passed"] else "ok_low_quality", **r}
        out["needs_clinician_review"] = bool(out["confidence"] < MIN_CONFIDENCE or not quality["passed"])
        out["quality"] = quality
        out["background_removal"] = {**P["info"], "pipeline": PIPELINE}
        if explain:
            out["background_removal"]["png_base64"] = encode_png(P["rgba"])                  # background-removed original colours
            out["background_removal"]["enhanced_png_base64"] = encode_png(P["rgba_view"])    # background-removed + CLAHE (what the heat sits on)
        if rb is not None:
            out["baseline"] = rb
            out["comparison"] = {"same_class": out["predicted_class"] == rb["predicted_class"], "same_triage": out["triage"] == rb["triage"],
                                 **(cam_agreement(cam, cam_b) if explain else {})}
        if fin is not None:                                                     # the decision to show to the user
            out["final"] = fin
            out["needs_clinician_review"] = fin["needs_clinician_review"]
        out.update(meta, latency_ms=int((time.time() - t0) * 1000))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clarus single-image DR prediction")
    ap.add_argument("image"); ap.add_argument("--final", help="For_App_Ready_Code_Of_CNN_Model folder (clarus_final.json)")
    ap.add_argument("--model"); ap.add_argument("--baseline")
    ap.add_argument("--no-tta", action="store_true"); ap.add_argument("--refer-threshold", help="number, or 'tuned' (only with --model)")
    ap.add_argument("--save-dir", help="write background-removed / enhanced PNGs + Grad-CAM PNGs here")
    a = ap.parse_args()
    if a.final: clarus = ClarusPredictor.from_final(a.final, tta=not a.no_tta)
    elif a.model: clarus = ClarusPredictor(a.model, a.baseline, tta=not a.no_tta, refer_threshold=(a.refer_threshold if a.refer_threshold in (None, "tuned") else float(a.refer_threshold)))
    else: ap.error("give --final or --model")
    res = clarus.predict(a.image)
    if a.save_dir and "png_base64" in res.get("background_removal", {}):
        d = Path(a.save_dir); d.mkdir(parents=True, exist_ok=True)
        for name, b64 in [("background_removed.png", res["background_removal"]["png_base64"]),
                          ("enhanced_view.png", res["background_removal"]["enhanced_png_base64"]),
                          ("gradcam_integrated.png", res["explanation"]["overlay_png_base64"]),
                          *([("gradcam_baseline.png", res["baseline"]["explanation"]["overlay_png_base64"])] if "baseline" in res else []),
                          *([("gradcam_final.png", res["final"]["explanation"]["overlay_png_base64"])] if "final" in res else [])]:
            (d / name).write_bytes(base64.b64decode(b64))
    for blk in (res, res.get("baseline", {}), res.get("final", {}), res.get("background_removal", {})):
        for k in ("png_base64", "enhanced_png_base64"): blk.pop(k, None)
        blk.get("explanation", {}).pop("overlay_png_base64", None)
    print(json.dumps(res, indent=2))
