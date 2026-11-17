import numpy as np

NOSE = 89

def normalize_rows(p):
    """p: (N, 100, 2) or (N, 200) raw (x, y) landmarks -> (N, 200) eye-aligned features."""
    pts = np.asarray(p, dtype=np.float64).reshape(-1, 100, 2).copy()
    left = pts[:, 0:8].mean(axis=1)
    right = pts[:, 8:16].mean(axis=1)
    v = right - left
    eye_dist = np.linalg.norm(v, axis=1)
    eye_dist = np.where(eye_dist > 1e-8, eye_dist, 1.0)
    ang = np.arctan2(v[:, 1], v[:, 0])

    pts = pts - pts[:, NOSE : NOSE + 1]
    c, s = np.cos(-ang), np.sin(-ang)
    p_x = pts[:, :, 0] * c[:, None] - pts[:, :, 1] * s[:, None]
    p_y = pts[:, :, 0] * s[:, None] + pts[:, :, 1] * c[:, None]
    out = np.stack([p_x, p_y], axis=-1) / eye_dist[:, None, None]
    return out.reshape(len(pts), -1)

def normalize_row(v):
    """Single-frame (200,) wrapper kept identical for Member B's cv_pipeline.py."""
    return normalize_rows(np.asarray(v, dtype=np.float64).reshape(1, 100, 2))[0]
