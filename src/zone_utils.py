"""Helpers built on A's zones.ZONES (zone names are A's: eyes, eyebrows, nose, mouth, face_contour)."""
from zones import ZONES

def point_to_zone():
    """List of 100 zone names, one per Kinect point."""
    m = [None] * 100
    for zone, pts in ZONES.items():
        for p in pts:
            m[p] = zone
    return m

def feature_to_zone():
    """List of 200 zone names (feature j belongs to point j // 2)."""
    p2z = point_to_zone()
    return [p2z[j // 2] for j in range(200)]
