"""predict_gfe: photo -> annotated image, top expression, 9 scores, zone importance.
Shared by the Gradio page and the Streamlit backup. Owner: Member A."""
import os
import sys
import json
import numpy as np
import joblib

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.append(_here)
_BASE = os.path.abspath(os.path.join(_here, ".."))

import cv_pipeline                                        # B's photo -> points -> vector code

# ---- load the five frozen model files once, at import ----
_scaler = joblib.load(os.path.join(_BASE, "models", "final_scaler.pkl"))
_idx = joblib.load(os.path.join(_BASE, "models", "selected_indices.pkl"))
_model = joblib.load(os.path.join(_BASE, "models", "final_model.pkl"))
with open(os.path.join(_BASE, "models", "zone_weights.json")) as f:
    _zw = json.load(f)
with open(os.path.join(_BASE, "models", "label_names.json")) as f:
    _names = json.load(f)

assert isinstance(_idx, list) and len(_idx) == 80 and len(set(_idx)) == 80, "selected_indices.pkl must be ONE list of 80"
assert list(_model.classes_) == _names, "model class order differs from label_names.json"
ZONE_WEIGHTS = _zw["overall"]                             # model-level bars, the same for every photo

def _softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()

def scores_for_normalized(v200):
    """One already-normalized row (200,) -> raw decision scores (9,).
    The parity check and predict_gfe both go through this exact code."""
    v = np.asarray(v200, dtype=np.float64).reshape(1, 200)
    return _model.decision_function(_scaler.transform(v)[:, _idx])[0]

def predict_gfe(img, use_clahe=False):
    """img: RGB uint8 array. use_clahe: optional CLAHE contrast fix in the CV step only
    (passed to cv_pipeline.get_points; the model is unaffected). Returns (annotated image, message text,
    relative-score dict for the 9 expressions, zone-importance dict)."""
    pts = cv_pipeline.get_points(img, use_clahe=use_clahe)                     # (100, 2) pixel coords, or None
    if pts is None:
        return img, "No face detected. Try a clearer, front-facing photo.", {}, ZONE_WEIGHTS

    v = cv_pipeline.get_vector(pts)                       # (1, 200), ALREADY normalized by normalize_row
    if v is None or not np.isfinite(v).all():
        return img, "Face found, but the landmarks could not be processed.", {}, ZONE_WEIGHTS

    # sanity guard: after normalization the mouth (points 48-67) must sit BELOW the
    # nose tip (y > 0 in image coords). If not, the landmark map is wrong (e.g. left/right swap).
    if v.reshape(100, 2)[48:68, 1].mean() <= 0:
        return img, "Face found, but the landmark layout looks wrong (possible left/right swap in the map).", {}, ZONE_WEIGHTS

    raw = scores_for_normalized(v[0])                     # 9 raw decision scores
    rel = _softmax(raw)                                   # DISPLAY ONLY, labelled 'relative'
    top = _names[int(np.argmax(raw))]
    scores = {n: float(p) for n, p in zip(_names, rel)}

    annotated = cv_pipeline.draw_overlay(img, pts, ZONE_WEIGHTS)
    if annotated is None:
        annotated = img
    msg = f"**Top expression: {top}**  (raw score {raw.max():.2f})\n\nBars show *relative* scores (softmax of decision scores), not probabilities. Demo only."
    return annotated, msg, scores, ZONE_WEIGHTS
