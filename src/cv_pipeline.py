"""CV pipeline: photo -> MediaPipe landmarks -> 100 Kinect-style points -> (1, 200) vector -> overlay."""
import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.append(_here)
from landmark_map import KINECT_TO_MEDIAPIPE_MAP
from normalize import normalize_row
from zone_utils import point_to_zone

MODEL_PATH = os.path.join(_here, "..", "models", "face_landmarker.task")

_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
)

def enhance(img_rgb):
    """CLAHE on the L channel in LAB colour. Returns an RGB uint8 image."""
    if img_rgb is None:
        return None
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def extract_478(img_rgb):
    """img_rgb: uint8 (H, W, 3) RGB. Returns (478, 2) pixel coords
    (468 face points + 10 iris points), or None if no face."""
    if img_rgb is None:
        return None
    h, w = img_rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=np.ascontiguousarray(img_rgb))
    res = _landmarker.detect(image)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

def extract_468(img_rgb):
    """Returns (468, 2) pixel coords, or None if no face."""
    pts = extract_478(img_rgb)
    return None if pts is None else pts[:468]

def get_points(img_rgb, use_clahe=False):
    """Photo -> (100, 2) pixel coords in Kinect order, or None if no face.
    use_clahe=True runs the optional CLAHE contrast fix first. It is off by
    default because it did not improve detection in our tests. The image size
    does not change, so the points still fit the original photo."""
    if use_clahe:
        img_rgb = enhance(img_rgb)
    pts = extract_478(img_rgb)
    if pts is None:
        return None
    return pts[KINECT_TO_MEDIAPIPE_MAP]

def get_vector(pts100):
    """(100, 2) pixel coords -> (1, 200) normalized vector (same code as training)."""
    if pts100 is None:
        return None
    return normalize_row(pts100.reshape(200))[None, :]

_RED = (230, 50, 50)
_AMBER = (245, 170, 30)
_GREEN = (60, 180, 90)
_GRAY = (150, 150, 150)

def _draw_dots(img_rgb, pts100, zone_weights):
    """Draw the 100 points on a copy of the photo, coloured by RANK of each
    zone's weight: top third red, middle third amber, bottom third green.
    (Rank, not size, because the zone weights are close together.)
    A zone name missing from zone_weights is drawn gray.
    Returns None if there is no image or no face."""
    if img_rgb is None or pts100 is None:
        return None
    p2z = point_to_zone()
    zones = sorted(zone_weights, key=zone_weights.get, reverse=True)
    n = len(zones)
    colour = {}
    for k, z in enumerate(zones):
        frac = (k + 0.5) / n
        colour[z] = _RED if frac < 1 / 3 else (_AMBER if frac < 2 / 3 else _GREEN)
    out = img_rgb.copy()
    r = max(2, int(round(min(out.shape[:2]) / 150)))
    for i, (x, y) in enumerate(pts100):
        cv2.circle(out, (int(round(x)), int(round(y))), r,
                   colour.get(p2z[i], _GRAY), -1)
    return out


def draw_overlay(img_rgb, pts100, zone_weights, style="zones"):
    """style="zones" (default): shaded zones from overlay_view.py. style="dots": the 100 loose dots."""
    if style == "dots":
        return _draw_dots(img_rgb, pts100, zone_weights)
    from overlay_view import draw_zone_overlay
    return draw_zone_overlay(img_rgb, pts100, zone_weights)
