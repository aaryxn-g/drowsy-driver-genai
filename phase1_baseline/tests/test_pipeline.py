"""End-to-end test of both passes with a stub landmarker.

MediaPipe needs a model file that cannot be fetched in every environment, so
the landmarker is stubbed here. Everything else is real: a real video file on
disk, the real frame loop, real Parquet output, the real baseline logic and the
real aggregation CLI. That covers the plumbing; the geometry and blink maths
are covered by their own suites.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import config, extract

REPO = Path(__file__).resolve().parents[1]


def synth_face(openness: float, mouth: float = 0.05,
               w: int = 640, h: int = 480) -> np.ndarray:
    """A full 478-point landmark array with a controllable eye and mouth."""
    pts = np.zeros((478, 3), dtype=float)

    # pose points: canonical model projected head-on so solvePnP has a solution
    from src import geometry
    k = geometry.camera_matrix(w, h)
    for idx, model_pt in zip(config.POSE_LANDMARKS, config.POSE_MODEL_3D):
        cam = np.asarray(model_pt) + np.array([0.0, 0.0, 1000.0])
        proj = k @ cam
        pts[idx] = (proj[0] / proj[2], proj[1] / proj[2], 0.0)

    # eyes: half-opening scaled so EAR ranges about 0.02 (shut) to 0.30 (open)
    half = 0.15 + openness * 4.35          # EAR = 2*half/30
    for eye, cx in ((config.RIGHT_EYE, 240.0), (config.LEFT_EYE, 400.0)):
        p1, p2, p3, p4, p5, p6 = eye
        pts[p1] = (cx - 15, 200.0, 0.0)
        pts[p4] = (cx + 15, 200.0, 0.0)
        pts[p2] = (cx - 5, 200.0 + half, 0.0)
        pts[p3] = (cx + 5, 200.0 + half, 0.0)
        pts[p6] = (cx - 5, 200.0 - half, 0.0)
        pts[p5] = (cx + 5, 200.0 - half, 0.0)

    a, b = config.MOUTH_CORNERS
    pts[a] = (280.0, 330.0, 0.0)
    pts[b] = (360.0, 330.0, 0.0)
    for top, bot in config.MOUTH_VERTICAL_PAIRS:
        pts[top] = (320.0, 330.0 + mouth * 40, 0.0)
        pts[bot] = (320.0, 330.0 - mouth * 40, 0.0)
    return pts


class StubLandmarker:
    """Returns a face whose eye openness follows a scripted schedule.

    It enforces the real tasks API's VIDEO-mode contract: timestamps must strictly
    increase over the instance's life, and a closed instance cannot be used. Without
    this the suite passed while every real shard died on its second video.
    """

    def __init__(self, schedule, miss_frames: set[int] | None = None):
        self.schedule = schedule
        self.miss = miss_frames or set()
        self.calls = 0
        self.last_timestamp_ms: int | None = None
        self.closed = False

    backend = "stub"

    def __call__(self, frame, timestamp_ms=0):
        if self.closed:
            raise RuntimeError("landmarker used after close()")
        if self.last_timestamp_ms is not None and timestamp_ms <= self.last_timestamp_ms:
            # The message MediaPipe raises, so a test can match on it.
            raise ValueError("Input timestamp must be monotonically increasing.")
        self.last_timestamp_ms = timestamp_ms
        i = self.calls
        self.calls += 1
        if i in self.miss:
            return None
        return synth_face(self.schedule(i))

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_video(path: Path, n_frames: int, fps: int = 30,
                size: tuple[int, int] = (320, 240)) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    assert writer.isOpened(), "could not open a video writer"
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()


@pytest.fixture
def video(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    write_video(p, n_frames=300)
    if not p.exists() or p.stat().st_size == 0:
        pytest.skip("no usable video codec in this environment")
    return p


class TestProcessVideo:
    def test_reads_every_frame_and_builds_the_table(self, video):
        stub = StubLandmarker(lambda i: 1.0)
        df, meta = extract.process_video(video, stub)
        assert len(df) > 250
        assert list(df.columns) == extract.FEATURE_COLUMNS
        assert meta["detection_rate"] == pytest.approx(1.0)
        assert df["ear"].notna().all()

    def test_missing_detections_become_nan_rows_not_dropped(self, video):
        """Dropping undetected frames would compress time and break durations."""
        stub = StubLandmarker(lambda i: 1.0, miss_frames=set(range(50, 100)))
        df, meta = extract.process_video(video, stub)
        assert len(df) > 250                       # rows kept
        assert df["detected"].sum() < len(df)      # some marked undetected
        assert df.loc[~df["detected"], "ear"].isna().all()
        assert meta["detection_rate"] < 1.0

    def test_timestamps_are_monotonic_and_match_fps(self, video):
        df, meta = extract.process_video(video, StubLandmarker(lambda i: 1.0))
        t = df["t_s"].to_numpy()
        assert np.all(np.diff(t) > 0)
        assert np.diff(t).mean() == pytest.approx(1.0 / meta["fps"], rel=0.05)

    def test_eye_closure_shows_up_in_ear(self, video):
        def schedule(i):
            return 0.05 if 100 <= i < 160 else 1.0
        df, _ = extract.process_video(video, StubLandmarker(schedule))
        closed = df.loc[110:150, "ear"].mean()
        open_ = df.loc[0:90, "ear"].mean()
        assert closed < open_ / 2

    def test_stride_subsamples(self, video):
        full, _ = extract.process_video(video, StubLandmarker(lambda i: 1.0))
        strided, meta = extract.process_video(
            video, StubLandmarker(lambda i: 1.0), stride=3
        )
        assert len(strided) == pytest.approx(len(full) / 3, rel=0.1)
        assert meta["effective_fps"] == pytest.approx(meta["fps"] / 3)


class TestAggregateCli:
    def test_two_pass_run_separates_alert_from_drowsy(self, tmp_path, video):
        """The whole pipeline, run as the user runs it, must rank drowsy higher."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        def alert(i):      # a brief blink every second
            return 0.05 if (i % 30) < 6 else 1.0

        def drowsy(i):     # long closures, eyes half shut between them
            return 0.05 if (i % 90) < 55 else 0.6

        rows = []
        for subject in ("01", "02"):
            for cls, sched in (("alert", alert), ("drowsy", drowsy)):
                vid = f"{subject}_{cls}"
                df, _ = extract.process_video(video, StubLandmarker(sched))
                df.to_parquet(frames_dir / f"{vid}.parquet", index=False)
                rows.append({"video_id": vid, "subject_id": subject,
                             "fold": "1", "class": cls, "relpath": "x.mp4"})

        manifest = tmp_path / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        out = tmp_path / "video_features.csv"

        proc = subprocess.run(
            [sys.executable, "-m", "src.aggregate",
             "--manifest", str(manifest),
             "--frames", str(frames_dir),
             "--out", str(out),
             "--events-out", str(tmp_path / "events.csv")],
            cwd=REPO, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

        feats = pd.read_csv(out).set_index("video_id")
        for subject in ("01", "02"):
            a = feats.loc[f"{subject}_alert"]
            d = feats.loc[f"{subject}_drowsy"]
            assert d["perclos80_mean"] > a["perclos80_mean"], subject
            assert d["total_closure_s"] > a["total_closure_s"], subject

        # the baseline must come from the alert video, per subject
        assert feats["baseline_source"].eq("alert").all()
        assert feats.loc["01_alert", "ear_open_baseline"] == pytest.approx(
            feats.loc["01_drowsy", "ear_open_baseline"]
        )

    def test_baseline_is_shared_across_a_subjects_videos(self, tmp_path, video):
        """Both of a subject's videos must be scored on the same ruler."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        rows = []
        alert_sched = lambda i: 0.05 if (i % 30) < 5 else 1.0
        drowsy_sched = lambda i: 0.05 if (i % 60) < 40 else 0.5
        for cls, sched in (("alert", alert_sched), ("drowsy", drowsy_sched)):
            vid = f"07_{cls}"
            df, _ = extract.process_video(video, StubLandmarker(sched))
            df.to_parquet(frames_dir / f"{vid}.parquet", index=False)
            rows.append({"video_id": vid, "subject_id": "07",
                         "fold": "2", "class": cls, "relpath": "x.mp4"})

        manifest = tmp_path / "m.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        out = tmp_path / "f.csv"
        proc = subprocess.run(
            [sys.executable, "-m", "src.aggregate",
             "--manifest", str(manifest), "--frames", str(frames_dir),
             "--out", str(out), "--events-out", str(tmp_path / "e.csv")],
            cwd=REPO, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        feats = pd.read_csv(out)
        assert feats["ear_open_baseline"].nunique() == 1
        assert feats["ear_closed_baseline"].nunique() == 1


class TestManifestScanner:
    def test_discovers_subjects_folds_and_classes(self, tmp_path):
        from src import manifest as man

        # Subject ids are unique across folds, as in UTA-RLDD.
        for fold, subjects in ((1, ("01", "02")), (2, ("13", "14"))):
            for subject in subjects:
                d = tmp_path / f"Fold{fold}_part1" / subject
                d.mkdir(parents=True)
                for stem in ("0", "5", "10"):
                    (d / f"{stem}.mp4").write_bytes(b"x")

        rows, problems, _ = man.scan(tmp_path)
        assert len(rows) == 12
        assert problems == []
        assert {r["class"] for r in rows} == set(config.CLASS_ORDER)
        assert {r["fold"] for r in rows} == {"1", "2"}
        assert {r["subject_id"] for r in rows} == {"01", "02", "13", "14"}

    def test_unknown_stems_are_reported_not_guessed(self, tmp_path):
        from src import manifest as man

        d = tmp_path / "Fold1_part1" / "01"
        d.mkdir(parents=True)
        (d / "0.mp4").write_bytes(b"x")
        (d / "mystery.mp4").write_bytes(b"x")

        rows, problems, _ = man.scan(tmp_path)
        assert len(rows) == 1
        assert len(problems) == 1
        assert problems[0]["reason"] == "unrecognised file stem"
