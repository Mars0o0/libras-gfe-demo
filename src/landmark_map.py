"""Proposed Kinect -> MediaPipe map (2026-11-16): contour and eyebrows re-picked to match the average Kinect face.
Eyes, nose, mouth unchanged. Approximate by design: the demo makes no accuracy claim."""

KINECT_TO_MEDIAPIPE_MAP = [133, 157, 159, 161, 33, 163, 145, 154, 362, 384, 386, 388, 263, 390, 374, 381, 55, 65, 66, 52, 53, 224, 223, 222, 28, 221, 285, 296, 334, 293, 300, 276, 283, 282, 295, 441, 193, 122, 217, 48, 64, 97, 326, 294, 278, 437, 351, 417, 61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, 78, 81, 13, 311, 308, 402, 14, 178, 226, 117, 50, 187, 214, 210, 211, 140, 171, 175, 400, 378, 394, 364, 435, 401, 366, 447, 264, 468, 473, 1, 107, 108, 69, 105, 63, 336, 299, 333, 298, 301]
assert len(KINECT_TO_MEDIAPIPE_MAP) == 100
