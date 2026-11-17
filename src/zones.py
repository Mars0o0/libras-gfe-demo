# Zones from UCI data_descriptions.txt (0-indexed, 100 points)
ZONES = {
    "eyes": list(range(0, 16)) + [87, 88],                    # both eyes + irises (18)
    "eyebrows": list(range(16, 36)) + list(range(90, 100)),   # eyebrows + lines above (30)
    "nose": list(range(36, 48)) + [89],                       # nose + nose tip (13)
    "mouth": list(range(48, 68)),                             # (20)
    "face_contour": list(range(68, 87)),                      # (19)
}
# every point must appear exactly once
assert sorted(sum(ZONES.values(), [])) == list(range(100))
