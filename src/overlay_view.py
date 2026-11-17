"""Presentation layer: draw the 100 points as coloured facial ZONES instead of loose dots.
Does not touch the model or the map. Colours show each zone's RANK of importance:
top third red, middle third amber, bottom third green."""
import numpy as np
import cv2
from zones import ZONES

# Kinect point numbers per region (left/right as seen in the picture)
LEFT_EYE   = list(range(0, 8)) + [87]
RIGHT_EYE  = list(range(8, 16)) + [88]
LEFT_BROW  = list(range(16, 26)) + list(range(90, 95))     # brow outline + the line above it
RIGHT_BROW = list(range(26, 36)) + list(range(95, 100))
NOSE       = list(range(36, 48)) + [89]
MOUTH      = list(range(48, 68))
CONTOUR    = list(range(68, 87))                           # jaw line, in order

# fail loudly if A ever changes the zones
assert set(LEFT_EYE + RIGHT_EYE) == set(ZONES['eyes'])
assert set(LEFT_BROW + RIGHT_BROW) == set(ZONES['eyebrows'])
assert set(NOSE) == set(ZONES['nose'])
assert set(MOUTH) == set(ZONES['mouth'])
assert set(CONTOUR) == set(ZONES['face_contour'])

_RED, _AMBER, _GREEN, _GRAY = (230, 50, 50), (245, 170, 30), (60, 180, 90), (150, 150, 150)

def _rank_colours(zone_weights):
    zones = sorted(zone_weights, key=zone_weights.get, reverse=True)
    n = len(zones)
    out = {}
    for k, z in enumerate(zones):
        frac = (k + 0.5) / n
        out[z] = _RED if frac < 1 / 3 else (_AMBER if frac < 2 / 3 else _GREEN)
    return out

def _smooth(pts, rounds=2):
    """Chaikin corner cutting: turns a wobbly open polyline into a smooth curve."""
    p = np.asarray(pts, dtype=np.float64)
    for _ in range(rounds):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.stack([q, r], axis=1).reshape(-1, 2)
        p = np.vstack([p[:1], mid, p[-1:]])
    return p

def draw_zone_overlay(img_rgb, pts100, zone_weights, show_dots=False, alpha=0.35, convex_jaw=True):
    """Returns a copy of the photo with each zone shaded by importance rank,
    or None if there is no image or no face."""
    if img_rgb is None or pts100 is None:
        return None
    colours = _rank_colours(zone_weights)
    base = img_rgb.copy()
    fill = img_rgb.copy()
    t = max(1, int(round(min(img_rgb.shape[:2]) / 280)))     # line thickness scales with the photo
    pts = np.asarray(pts100, dtype=np.float64)

    regions = [(LEFT_EYE, 'eyes'), (RIGHT_EYE, 'eyes'), (LEFT_BROW, 'eyebrows'),
               (RIGHT_BROW, 'eyebrows'), (NOSE, 'nose'), (MOUTH, 'mouth')]
    hulls = []
    for idx, z in regions:
        h = cv2.convexHull(np.round(pts[idx]).astype(np.int32))
        cv2.fillConvexPoly(fill, h, colours.get(z, _GRAY))
        hulls.append((h, colours.get(z, _GRAY)))
    out = cv2.addWeighted(fill, alpha, base, 1 - alpha, 0)
    for h, c in hulls:
        cv2.polylines(out, [h], True, c, t, cv2.LINE_AA)
    line = pts[CONTOUR]
    if convex_jaw:                       # a jaw never dents inward: skip points that fall inside the outline
        hull_i = cv2.convexHull(np.round(line).astype(np.int32), returnPoints=False).flatten().tolist()
        line = line[sorted(set(hull_i) | {0, len(CONTOUR) - 1})]
    curve = np.round(_smooth(line)).astype(np.int32)
    cv2.polylines(out, [curve], False, colours.get('face_contour', _GRAY), t * 2, cv2.LINE_AA)
    if show_dots:
        for x, y in pts:
            cv2.circle(out, (int(round(x)), int(round(y))), max(1, t), (255, 255, 255), -1, cv2.LINE_AA)
    return out
