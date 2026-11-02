"""Shared constants for the Phase 1 feature pipeline.

Landmark indices refer to the MediaPipe Face Landmarker topology (468 base points,
478 with refine_landmarks / iris). Index choices follow the convention used across
the FaceMesh literature; they are listed explicitly here so the geometry module can
be unit-tested without MediaPipe installed.
"""
from __future__ import annotations

# ----------------------------------------------------------------- eye indices
# Ordered p1..p6 following Soukupova & Cech (2016):
#   p1 = outer corner, p2/p3 = upper lid, p4 = inner corner, p5/p6 = lower lid
# EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)   # subject's right eye (left of image)
LEFT_EYE = (362, 385, 387, 263, 373, 380)   # subject's left eye

# ---------------------------------------------------------------- mouth indices
# Vertical openings sampled at three points across the inner lip, normalised by
# the corner-to-corner mouth width. Averaging three pairs makes MAR far less
# jumpy than the single 13/14 pair during speech.
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL_PAIRS = ((81, 178), (13, 14), (311, 402))

# ------------------------------------------------------------- head pose model
# Six-point canonical face model in millimetres, origin at the nose tip.
# Paired with the landmark indices below for cv2.solvePnP.
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
# Axes follow OpenCV's camera frame: X right, Y DOWN, Z away from the camera.
# The widely copied version of this model is Y-up (chin negative, eyes positive),
# which disagrees with image coordinates and makes solvePnP return a pitch near
# 180 degrees for a frontal face. That offset sits right on the angle wrap, so
# pitch jumps between -179 and +179 during an ordinary nod and destroys
# pitch_std and pitch_range. Y and Z are negated here to match the camera.
POSE_MODEL_3D = (
    (0.0, 0.0, 0.0),           # 1   nose tip
    (0.0, 330.0, 65.0),        # 152 chin, below the nose, so +Y
    (-225.0, -170.0, 135.0),   # 33  right eye outer corner, above the nose
    (225.0, -170.0, 135.0),    # 263 left eye outer corner
    (-150.0, 150.0, 125.0),    # 61  right mouth corner
    (150.0, 150.0, 125.0),     # 291 left mouth corner
)

# --------------------------------------------------------------- eye-openness
# Openness is EAR rescaled to [0, 1] against a per-subject baseline, so that
# thresholds mean the same thing for a wide-eyed and a narrow-eyed driver.
#
# PERCLOS P80 is the fraction of time the eye is at least 80% closed, i.e.
# openness < 0.20. P70 (openness < 0.30) is reported alongside it because some
# of the literature uses that variant.
PERCLOS_P80_THRESHOLD = 0.20
PERCLOS_P70_THRESHOLD = 0.30
PERCLOS_WINDOW_S = 60.0
PERCLOS_STEP_S = 1.0

# Percentiles used to estimate a subject's fully-open and fully-closed EAR.
# The open baseline is deliberately taken from the subject's ALERT video and
# reused for their drowsy videos: a drowsy video may never show a fully open
# eye, so a within-video percentile would silently rescale the problem away.
EAR_OPEN_PERCENTILE = 90.0
EAR_CLOSED_PERCENTILE = 2.0

# A subject whose open and closed EAR barely differ has no usable ruler: either
# the landmarker never tracked their eyelid, or the clip contains no blinks.
# Scoring them anyway would turn sensor noise into a drowsiness reading, so the
# subject is reported and skipped instead.
MIN_BASELINE_SPAN = 0.02

# ------------------------------------------------------------ blink detection
# An eye-closure event opens when openness falls below EVENT_ENTER and closes
# when it rises back above EVENT_EXIT. The gap between the two (hysteresis)
# stops a noisy signal hovering at the threshold from registering many events.
EVENT_ENTER = 0.45
EVENT_EXIT = 0.60
MIN_EVENT_FRAMES = 2

# An event must actually reach a near-closed state to count. Without this a
# signal that merely sags to 0.44 and stays there, which happens when the
# baseline drifts or the subject squints, is recorded as one enormous
# microsleep. The eye has to get at least 70% shut.
EVENT_MIN_DEPTH = 0.30

# Duration bands, in seconds. Boundaries are conventions, not physical limits,
# and are configurable per run.
BLINK_MAX_S = 0.40            # normal spontaneous blink
LONG_CLOSURE_MAX_S = 1.00     # prolonged closure
# anything longer is treated as a microsleep candidate

# -------------------------------------------------------------------- quality
# Videos below this face-detection rate are flagged rather than silently used.
# Detection rate is itself a fairness signal: if the landmarker fails more often
# on some subgroups, that is a finding, not a preprocessing nuisance.
MIN_DETECTION_RATE = 0.70

# UTA-RLDD labels its three recordings per subject on the Karolinska scale:
# 0 = alert, 5 = low vigilant, 10 = drowsy.
UTARLDD_CLASS_FROM_STEM = {"0": "alert", "5": "low_vigilant", "10": "drowsy"}
CLASS_ORDER = ("alert", "low_vigilant", "drowsy")
