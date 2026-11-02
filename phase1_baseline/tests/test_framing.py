"""Framing: where the face sits, centred pose, and the pitch confound flag."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import aggregate, config, extract, geometry

REPO = Path(__file__).resolve().parents[1]
W, H = 640, 480


def project_face(pitch_deg: float = 0.0, offset_mm=(0.0, 0.0), width: int = W,
                 height: int = H, distance: float = 1500.0) -> np.ndarray:
    """Project the pose model, rotated about camera X and moved in space."""
    a = np.radians(pitch_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    k = geometry.camera_matrix(width, height)
    tvec = np.array([offset_mm[0], offset_mm[1], distance])
    pts = np.zeros((478, 3), dtype=float)
    for idx, point in zip(config.POSE_LANDMARKS, config.POSE_MODEL_3D):
        proj = k @ (rx @ np.asarray(point) + tvec)
        pts[idx] = (proj[0] / proj[2], proj[1] / proj[2], 0.0)
    return pts


def shifted(pts: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = pts.copy()
    out[list(config.POSE_LANDMARKS), 0] += dx
    out[list(config.POSE_LANDMARKS), 1] += dy
    return out


class TestFramingColumns:
    def test_nose_position_tracks_a_face_moved_across_the_frame(self, tmp_path):
        """Through the real frame loop, on a smaller frame than the landmark default."""
        w, h = 320, 240
        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        assert writer.isOpened()
        for _ in range(60):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer.release()

        base = project_face(width=w, height=h, distance=3000.0)

        class Moving:
            backend = "stub"
            calls = 0

            def __call__(self, frame, timestamp_ms=0):
                i, self.calls = self.calls, self.calls + 1
                return shifted(base, -100 + 200 * i / 59, -80 + 160 * i / 59)

        df, _ = extract.process_video(path, Moving())
        for column in ("nose_x_norm", "nose_y_norm", "face_scale",
                       "pitch_centered", "yaw_centered"):
            assert column in extract.FEATURE_COLUMNS and df[column].notna().all()

        x, y = df["nose_x_norm"].to_numpy(), df["nose_y_norm"].to_numpy()
        assert x.min() >= 0 and x.max() <= 1 and y.min() >= 0 and y.max() <= 1
        assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
        assert x[0] == pytest.approx((w / 2 - 100) / w, abs=1e-6)
        assert y[-1] == pytest.approx((h / 2 + 80) / h, abs=1e-6)
        assert np.ptp(df["face_scale"]) < 1e-9   # moved across the frame, not closer

    def test_undetected_frames_are_nan(self, tmp_path):
        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        assert writer.isOpened()
        for _ in range(5):
            writer.write(np.zeros((H, W, 3), dtype=np.uint8))
        writer.release()

        class NoFace:
            backend = "stub"

            def __call__(self, frame, timestamp_ms=0):
                return None

        df, _ = extract.process_video(path, NoFace())
        assert df[["nose_x_norm", "nose_y_norm", "face_scale", "pitch_centered"]].isna().all().all()

    def test_face_scale_is_eye_span_over_width(self):
        pts = np.zeros((478, 3), dtype=float)
        pts[33] = (200.0, 100.0, 0.0)
        pts[263] = (360.0, 100.0, 0.0)
        assert geometry.framing(pts, W, H)["face_scale"] == pytest.approx(0.25)


class TestCenteredPose:
    def test_same_pixels_shifted_vary_raw_pitch_but_not_centered(self):
        """Regression for one portrait reading 24 degrees uncropped, -2.4 cropped.

        A crop moves the same pixels across the frame. The head has not moved at all.
        """
        base = project_face()
        raw, centered = [], []
        for dy in range(-140, 141, 40):
            pts = shifted(base, 0.0, float(dy))
            raw.append(geometry.head_pose(pts, W, H)[1])
            centered.append(geometry.centered_head_pose(pts, W, H)[1])
        assert np.ptp(raw) > 10
        assert np.ptp(centered) < 1e-3

    def test_a_real_nod_survives_centering(self):
        pitches = [geometry.centered_head_pose(project_face(p), W, H)[1]
                   for p in (-20, -10, 0, 10, 20)]
        assert np.all(np.diff(pitches) > 8)

    def test_head_moving_without_rotating_favours_raw_pitch(self):
        """Pins what centring does NOT do. This is the slump case.

        A head that slides down through space without rotating has a constant true
        pose. With correct intrinsics raw pitch stays right, and centring is what
        adds the artifact, because the off-axis view really does look rotated.
        """
        raw, centered = [], []
        for dy in range(-300, 301, 100):
            pts = project_face(offset_mm=(0.0, float(dy)))
            raw.append(geometry.head_pose(pts, W, H)[1])
            centered.append(geometry.centered_head_pose(pts, W, H)[1])
        assert np.ptp(raw) < 0.5
        assert np.ptp(centered) > 5


class TestCorrelation:
    def test_series_moving_together_is_near_one(self):
        rng = np.random.default_rng(0)
        y = np.linspace(0.3, 0.7, 1000)
        pitch = 40 * y + rng.normal(0, 0.05, 1000)
        assert aggregate.pearson(pitch, y) == pytest.approx(1.0, abs=0.01)

    def test_opposite_series_is_near_minus_one(self):
        y = np.linspace(0.3, 0.7, 100)
        assert aggregate.pearson(-y, y) == pytest.approx(-1.0, abs=1e-9)

    def test_independent_series_is_near_zero(self):
        rng = np.random.default_rng(0)
        assert abs(aggregate.pearson(rng.normal(size=5000), rng.normal(size=5000))) < 0.05

    def test_undetected_frames_are_excluded_not_treated_as_values(self):
        y = np.linspace(0.3, 0.7, 100)
        pitch = 40 * y
        pitch[10:20] = np.nan
        assert aggregate.pearson(pitch, y) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("x, y", [
        (np.ones(50), np.linspace(0, 1, 50)),
        (np.linspace(0, 1, 50), np.full(50, 0.5)),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0])),
    ])
    def test_undefined_is_nan(self, x, y):
        assert np.isnan(aggregate.pearson(x, y))


def frames_table(pitch, nose_y, n: int | None = None) -> pd.DataFrame:
    n = n or len(pitch)
    return pd.DataFrame({
        "pitch": pitch,
        "pitch_centered": np.zeros(n),
        "nose_x_norm": np.full(n, 0.5),
        "nose_y_norm": nose_y,
        "face_scale": np.full(n, 0.2),
    })


class TestPitchConfounded:
    N = 3000

    def drifting(self):
        y = np.linspace(0.3, 0.6, self.N)
        return frames_table(30 * y, y)

    def test_fires_on_a_drifting_subject(self):
        f = aggregate.framing_features(self.drifting())
        assert f["pitch_confounded"] is True
        assert f["nose_y_drift"] == pytest.approx(0.27, abs=0.001)   # p95 - p5 of 0.30 span
        assert f["pitch_nose_y_corr"] == pytest.approx(1.0, abs=1e-6)

    def test_silent_on_a_stationary_subject_who_nods(self):
        rng = np.random.default_rng(0)
        pitch = 10 * np.sin(np.linspace(0, 40, self.N))
        y = 0.5 + rng.normal(0, 0.005, self.N)
        f = aggregate.framing_features(frames_table(pitch, y))
        assert f["pitch_confounded"] is False
        assert f["nose_y_drift"] < 0.1

    def test_needs_correlation_as_well_as_drift(self):
        rng = np.random.default_rng(0)
        y = np.linspace(0.3, 0.6, self.N)
        f = aggregate.framing_features(frames_table(rng.normal(0, 5, self.N), y))
        assert f["nose_y_drift"] > 0.1 and f["pitch_confounded"] is False

    @pytest.mark.parametrize("corr, drift, expected", [
        (0.51, 0.11, True), (-0.51, 0.11, True),   # the sign does not matter
        (0.50, 0.11, False), (0.51, 0.10, False),  # both limits are strict
        (float("nan"), 0.5, False), (0.9, float("nan"), False),
    ])
    def test_limits(self, corr, drift, expected):
        assert aggregate.is_pitch_confounded(corr, drift) is expected

    def test_a_real_nod_also_fires_so_the_flag_is_not_proof_of_an_artifact(self):
        """Pins a limitation: the nose moves with the head, pivoting at the neck."""
        pivot = np.array([0.0, 600.0, 500.0])       # below and behind the nose
        pitch, nose_y = [], []
        for angle in 15 * np.sin(np.linspace(0, 12 * np.pi, 600)):
            a = np.radians(angle)
            rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            k = geometry.camera_matrix(W, H)
            tvec = pivot - rx @ pivot + np.array([0.0, 0.0, 2250.0])
            pts = np.zeros((478, 3), dtype=float)
            for idx, point in zip(config.POSE_LANDMARKS, config.POSE_MODEL_3D):
                proj = k @ (rx @ np.asarray(point) + tvec)
                pts[idx] = (proj[0] / proj[2], proj[1] / proj[2], 0.0)
            pitch.append(geometry.head_pose(pts, W, H)[1])
            nose_y.append(pts[config.NOSE_TIP, 1] / H)
        f = aggregate.framing_features(frames_table(np.array(pitch), np.array(nose_y)))
        assert np.ptp(pitch) == pytest.approx(30.0, abs=0.5)   # the nod itself is recovered
        assert f["pitch_confounded"] is True

    def test_all_undetected_video_is_nan_and_not_flagged(self):
        nan = np.full(10, np.nan)
        table = frames_table(nan, nan)
        table["pitch_centered"] = nan
        f = aggregate.framing_features(table)
        assert np.isnan(f["nose_y_drift"]) and np.isnan(f["pitch_centered_std"])
        assert f["pitch_confounded"] is False

    def test_centered_std_and_range(self):
        table = self.drifting()
        table["pitch_centered"] = np.linspace(-3.0, 5.0, self.N)
        f = aggregate.framing_features(table)
        assert f["pitch_centered_range"] == pytest.approx(8.0)
        assert f["pitch_centered_std"] == pytest.approx(np.std(table["pitch_centered"]))


def video_parquet(path: Path, drifting: bool, n: int = 3000) -> None:
    y = np.linspace(0.3, 0.6, n) if drifting else np.full(n, 0.5)
    pitch = 30 * y if drifting else 5 * np.sin(np.linspace(0, 40, n))
    ear = np.where(np.arange(n) % 90 < 8, 0.05, 0.30)
    pd.DataFrame({
        "frame": np.arange(n), "t_s": np.arange(n) / 30.0, "detected": True,
        "ear_left": ear, "ear_right": ear, "ear": ear, "ear_corrected": ear,
        "mar": 0.1, "yaw": 0.0, "pitch": pitch, "roll": 0.0,
        "pitch_centered": pitch / 2, "yaw_centered": 0.0,
        "nose_x_norm": 0.5, "nose_y_norm": y, "face_scale": 0.2,
    }).to_parquet(path, index=False)


def test_aggregate_cli_reports_confounding_by_class(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    # Two drowsy videos of which one drifts; both alert videos stay still.
    rows = []
    for subject, cls, drifting in (("01", "alert", False), ("01", "drowsy", True),
                                   ("02", "alert", False), ("02", "drowsy", False)):
        vid = f"{subject}_{cls}"
        video_parquet(frames / f"{vid}.parquet", drifting)
        rows.append({"video_id": vid, "subject_id": subject, "fold": "1", "class": cls,
                     "relpath": "x.mp4"})
    manifest = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    out = tmp_path / "f.csv"

    proc = subprocess.run(
        [sys.executable, "-m", "src.aggregate", "--manifest", str(manifest),
         "--frames", str(frames), "--out", str(out), "--events-out", str(tmp_path / "e.csv")],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    feats = pd.read_csv(out, dtype={"subject_id": str}).set_index("video_id")
    assert bool(feats.loc["01_drowsy", "pitch_confounded"]) is True
    assert feats["pitch_confounded"].sum() == 1
    cols = list(feats.columns)
    i = cols.index("pitch_range")
    assert cols[i + 1:i + 3] == ["pitch_centered_std", "pitch_centered_range"]
    for column in ("nose_y_drift", "nose_x_drift", "face_scale_drift", "pitch_nose_y_corr"):
        assert column in cols

    report = proc.stdout[proc.stdout.index("pitch_confounded videos by class"):]
    lines = {line.split()[0]: line.split() for line in report.splitlines()[1:] if line.strip()}
    assert lines["alert"][1:] == ["0", "2", "0.0"]
    assert lines["drowsy"][1:] == ["1", "2", "0.5"]
