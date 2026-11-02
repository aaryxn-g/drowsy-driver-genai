"""Geometry tests against hand-computed and synthetically projected ground truth."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import config, geometry


def blank(n: int = 478) -> np.ndarray:
    return np.zeros((n, 3), dtype=float)


def set_eye(pts: np.ndarray, idx: tuple[int, ...], half_open: float,
            x0: float = 0.0, width: float = 30.0) -> None:
    """Place a symmetric synthetic eye so EAR evaluates to half_open*2/width."""
    p1, p2, p3, p4, p5, p6 = idx
    pts[p1] = (x0, 0.0, 0.0)
    pts[p4] = (x0 + width, 0.0, 0.0)
    pts[p2] = (x0 + width / 3, half_open, 0.0)
    pts[p3] = (x0 + 2 * width / 3, half_open, 0.0)
    pts[p6] = (x0 + width / 3, -half_open, 0.0)
    pts[p5] = (x0 + 2 * width / 3, -half_open, 0.0)


class TestEAR:
    def test_matches_hand_computation(self):
        pts = blank()
        set_eye(pts, config.RIGHT_EYE, half_open=3.0, width=30.0)
        # (|p2-p6| + |p3-p5|) / (2|p1-p4|) = (6 + 6) / 60 = 0.2
        assert geometry.eye_aspect_ratio(pts, config.RIGHT_EYE) == pytest.approx(0.2)

    def test_closed_eye_near_zero(self):
        pts = blank()
        set_eye(pts, config.RIGHT_EYE, half_open=0.15, width=30.0)
        assert geometry.eye_aspect_ratio(pts, config.RIGHT_EYE) == pytest.approx(0.01)

    def test_scale_invariant(self):
        """EAR is a ratio, so camera distance must not change it."""
        pts = blank()
        set_eye(pts, config.RIGHT_EYE, half_open=3.0, width=30.0)
        near = geometry.eye_aspect_ratio(pts, config.RIGHT_EYE)
        far = geometry.eye_aspect_ratio(pts * 2.7, config.RIGHT_EYE)
        assert near == pytest.approx(far)

    def test_translation_invariant(self):
        pts = blank()
        set_eye(pts, config.RIGHT_EYE, half_open=3.0)
        base = geometry.eye_aspect_ratio(pts, config.RIGHT_EYE)
        moved = pts.copy()
        moved[:, 0] += 411.0
        moved[:, 1] -= 97.0
        assert geometry.eye_aspect_ratio(moved, config.RIGHT_EYE) == pytest.approx(base)

    def test_degenerate_horizontal_is_nan(self):
        """Edge-on profile view: eye width collapses, EAR is meaningless."""
        pts = blank()
        assert math.isnan(geometry.eye_aspect_ratio(pts, config.RIGHT_EYE))

    def test_mean_survives_one_missing_eye(self):
        pts = blank()
        set_eye(pts, config.LEFT_EYE, half_open=3.0, width=30.0)
        left, right, mean = geometry.both_eyes_ear(pts)
        assert math.isnan(right)
        assert mean == pytest.approx(left)


class TestMAR:
    def test_matches_hand_computation(self):
        pts = blank()
        a, b = config.MOUTH_CORNERS
        pts[a] = (0.0, 0.0, 0.0)
        pts[b] = (50.0, 0.0, 0.0)
        for (top, bot), gap in zip(config.MOUTH_VERTICAL_PAIRS, (5.0, 10.0, 5.0)):
            pts[top] = (25.0, gap / 2, 0.0)
            pts[bot] = (25.0, -gap / 2, 0.0)
        # mean opening 20/3, width 50
        assert geometry.mouth_aspect_ratio(pts) == pytest.approx((20 / 3) / 50)

    def test_closed_mouth_is_zero(self):
        pts = blank()
        a, b = config.MOUTH_CORNERS
        pts[a] = (0.0, 0.0, 0.0)
        pts[b] = (50.0, 0.0, 0.0)
        assert geometry.mouth_aspect_ratio(pts) == pytest.approx(0.0)


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R = Rz(roll) @ Ry(yaw) @ Rx(pitch), matching rotation_to_euler."""
    a, b, c = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


class TestHeadPose:
    @pytest.mark.parametrize("yaw,pitch,roll", [
        (0, 0, 0), (20, 0, 0), (0, 15, 0), (0, 0, 25),
        (-18, 12, -8), (30, -20, 10),
    ])
    def test_euler_roundtrip(self, yaw, pitch, roll):
        y, p, r = geometry.rotation_to_euler(rotation(yaw, pitch, roll))
        assert (y, p, r) == pytest.approx((yaw, pitch, roll), abs=1e-6)

    @pytest.mark.parametrize("yaw,pitch,roll", [
        (0, 0, 0), (15, 0, 0), (0, -20, 0), (0, 0, 12), (-22, 14, -6),
    ])
    def test_solvepnp_recovers_projected_angles(self, yaw, pitch, roll):
        """Project the canonical model at a known pose, then solve for it.

        This is the real check: it exercises the 3D model points, the camera
        matrix, solvePnP and the Euler extraction as one chain.
        """
        w, h = 1280, 720
        rmat = rotation(yaw, pitch, roll)
        tvec = np.array([0.0, 0.0, 1000.0])
        k = geometry.camera_matrix(w, h)

        pts = np.zeros((478, 3), dtype=float)
        for idx, model_pt in zip(config.POSE_LANDMARKS, config.POSE_MODEL_3D):
            cam = rmat @ np.asarray(model_pt) + tvec
            proj = k @ cam
            pts[idx] = (proj[0] / proj[2], proj[1] / proj[2], 0.0)

        got = geometry.head_pose(pts, w, h)
        assert got == pytest.approx((yaw, pitch, roll), abs=0.5)


class TestPoseCorrection:
    def test_no_correction_at_frontal(self):
        assert geometry.ear_pose_correction(0.3, 0.0) == pytest.approx(0.3)

    def test_turned_head_reduces_inflated_ear(self):
        """A 30 degree turn inflates raw EAR; correction pulls it back down."""
        corrected = geometry.ear_pose_correction(0.3, 30.0)
        assert corrected == pytest.approx(0.3 * math.cos(math.radians(30)))
        assert corrected < 0.3

    def test_extreme_yaw_is_discarded(self):
        assert math.isnan(geometry.ear_pose_correction(0.3, 70.0))

    def test_symmetric_in_sign(self):
        assert geometry.ear_pose_correction(0.3, 25.0) == pytest.approx(
            geometry.ear_pose_correction(0.3, -25.0)
        )
