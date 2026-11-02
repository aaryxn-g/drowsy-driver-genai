"""Per-frame facial geometry: eye aspect ratio, mouth aspect ratio, head pose.

Everything here operates on a plain (N, 3) array of landmark coordinates, so it
can be tested without MediaPipe. Coordinates are expected in pixels; the caller
is responsible for scaling normalised landmarks by the frame size first.

EAR and MAR are ratios, so they are invariant to camera distance. They are NOT
invariant to head rotation: turning the head foreshortens the horizontal eye
width and inflates EAR. `ear_pose_correction` exists for that reason, and the
head-pose angles are carried through to the feature table so the downstream
model can learn the interaction rather than having it hidden.
"""
from __future__ import annotations

import math

import numpy as np

from . import config

try:  # OpenCV is only needed for head pose
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(pts: np.ndarray, eye_indices: tuple[int, ...]) -> float:
    """Soukupova & Cech EAR for one eye.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Returns NaN if the horizontal span collapses, which happens on a profile
    view where the eye is edge-on and the ratio is meaningless anyway.
    """
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in eye_indices)
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return float("nan")
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """Mean inner-lip opening normalised by mouth width.

    Averaging three vertical pairs rather than one makes the signal far steadier
    during speech, where a single mid-lip pair swings wildly.
    """
    left, right = (pts[i] for i in config.MOUTH_CORNERS)
    width = _dist(left, right)
    if width < 1e-6:
        return float("nan")
    openings = [_dist(pts[a], pts[b]) for a, b in config.MOUTH_VERTICAL_PAIRS]
    return float(np.mean(openings) / width)


def both_eyes_ear(pts: np.ndarray) -> tuple[float, float, float]:
    """Return (left EAR, right EAR, mean EAR).

    The mean ignores a NaN side so a partially occluded face still yields a
    usable signal from the visible eye.
    """
    left = eye_aspect_ratio(pts, config.LEFT_EYE)
    right = eye_aspect_ratio(pts, config.RIGHT_EYE)
    both = [v for v in (left, right) if not math.isnan(v)]
    mean = float(np.mean(both)) if both else float("nan")
    return left, right, mean


# ------------------------------------------------------------------ head pose

def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics with focal length approximated by the frame width.

    UTA-RLDD is self-recorded on assorted phones and webcams with no calibration
    data, so this is the standard uncalibrated approximation. Absolute angles
    carry some error; relative change within one video, which is what nodding
    detection needs, is reliable.
    """
    focal = float(width)
    return np.array(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix to (yaw, pitch, roll) in degrees.

    Yaw   = turning left/right
    Pitch = nodding up/down   <- the drowsiness-relevant axis
    Roll  = tilting the head toward a shoulder
    """
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy < 1e-6:  # gimbal lock
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def head_pose(pts: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in degrees via PnP on six canonical points."""
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("opencv is required for head pose estimation")
    image_pts = np.array(
        [pts[i][:2] for i in config.POSE_LANDMARKS], dtype=np.float64
    )
    model_pts = np.array(config.POSE_MODEL_3D, dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        model_pts,
        image_pts,
        camera_matrix(width, height),
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return float("nan"), float("nan"), float("nan")
    rmat, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(rmat)


def ear_pose_correction(ear: float, yaw_deg: float, max_yaw: float = 45.0) -> float:
    """Undo the EAR inflation caused by horizontal foreshortening.

    Turning the head by yaw shrinks the apparent eye width by roughly cos(yaw)
    while leaving the vertical opening largely intact, so EAR rises even though
    the eyelid has not moved. Dividing by 1/cos(yaw) compensates to first order.

    Beyond max_yaw the eye is too oblique to measure and NaN is returned, which
    the downstream code treats as a missing frame rather than as a closed eye.
    """
    if math.isnan(ear) or math.isnan(yaw_deg):
        return float("nan")
    if abs(yaw_deg) > max_yaw:
        return float("nan")
    return ear * math.cos(math.radians(yaw_deg))


def frame_features(pts: np.ndarray, width: int, height: int) -> dict[str, float]:
    """All per-frame geometry for one detected face."""
    left, right, mean = both_eyes_ear(pts)
    yaw, pitch, roll = head_pose(pts, width, height)
    return {
        "ear_left": left,
        "ear_right": right,
        "ear": mean,
        "ear_corrected": ear_pose_correction(mean, yaw),
        "mar": mouth_aspect_ratio(pts),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }
