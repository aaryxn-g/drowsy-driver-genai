"""Pass 1 over several videos in a row, through the same loop the CLI runs.

Each video must get its own landmarker. MediaPipe's tasks API in VIDEO mode needs
strictly increasing timestamps for an instance's whole life, and every video's
clock restarts at 0, so a shared instance runs the first video and raises on all
the rest. The earlier suite stubbed the landmarker without that contract and so
passed while every real shard failed after one video.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import extract
from src.landmarks import DEFAULT_MODEL_DIR

from .test_pipeline import StubLandmarker, write_video


@pytest.fixture
def two_videos(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / "data"
    records = []
    for stem in ("0", "10"):
        rel = f"Fold1_part1/01/{stem}.mp4"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        write_video(root / rel, n_frames=45)
        if (root / rel).stat().st_size == 0:
            pytest.skip("no usable video codec in this environment")
        records.append({"video_id": f"01_{stem}", "subject_id": "01", "fold": "1",
                        "class": "alert" if stem == "0" else "drowsy",
                        "part": float("nan"), "relpath": rel})
    return root, records


def run(root: Path, records: list[dict], out: Path, make_landmarker):
    out.mkdir(parents=True, exist_ok=True)
    log = out / "_extraction_log.csv"
    return extract.extract_videos(records, root, out, log, make_landmarker), log


class TestStubContract:
    def test_rejects_a_timestamp_that_goes_back(self):
        stub = StubLandmarker(lambda i: 1.0)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stub(frame, 0)
        stub(frame, 33)
        with pytest.raises(ValueError, match="monotonically increasing"):
            stub(frame, 0)

    def test_rejects_a_repeated_timestamp(self):
        stub = StubLandmarker(lambda i: 1.0)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stub(frame, 10)
        with pytest.raises(ValueError, match="monotonically increasing"):
            stub(frame, 10)

    def test_rejects_use_after_close(self):
        stub = StubLandmarker(lambda i: 1.0)
        stub.close()
        with pytest.raises(RuntimeError):
            stub(np.zeros((4, 4, 3), dtype=np.uint8), 0)


class TestSequentialVideos:
    def test_two_videos_in_sequence_both_succeed(self, two_videos, tmp_path):
        root, records = two_videos
        made: list[StubLandmarker] = []

        def factory():
            made.append(StubLandmarker(lambda i: 1.0))
            return made[-1]

        (done, skipped, failed), log = run(root, records, tmp_path / "frames", factory)
        assert (done, skipped, failed) == (2, 0, 0)
        for r in records:
            df = pd.read_parquet(tmp_path / "frames" / f"{r['video_id']}.parquet")
            assert len(df) > 0 and df["detected"].all()
        logged = pd.read_csv(log, dtype=str, keep_default_na=False)
        assert list(logged["video_id"]) == ["01_0", "01_10"]
        assert list(logged["part"]) == ["", ""]

    def test_each_video_gets_a_fresh_landmarker_closed_after_use(self, two_videos, tmp_path):
        root, records = two_videos
        made: list[StubLandmarker] = []

        def factory():
            made.append(StubLandmarker(lambda i: 1.0))
            return made[-1]

        run(root, records, tmp_path / "frames", factory)
        assert len(made) == 2
        assert made[0] is not made[1]
        assert all(m.closed for m in made)
        # Both clocks started at zero, which only works on separate instances.
        assert all(m.calls > 0 for m in made)

    def test_a_shared_landmarker_fails_the_second_video(self, two_videos, tmp_path):
        """The old behaviour, reproduced, so this suite is known to catch it."""
        root, records = two_videos
        shared = StubLandmarker(lambda i: 1.0)

        class NoClose:
            backend = "stub"
            __call__ = staticmethod(shared.__call__)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

        (done, _, failed), _ = run(root, records, tmp_path / "frames", NoClose)
        assert (done, failed) == (1, 1)

    def test_a_failed_video_does_not_stop_the_next(self, two_videos, tmp_path):
        root, records = two_videos
        records = [{**records[0], "relpath": "nope.mp4"}, records[1]]
        (done, _, failed), _ = run(root, records, tmp_path / "frames",
                                   lambda: StubLandmarker(lambda i: 1.0))
        assert (done, failed) == (1, 1)


def _real_landmarker_or_skip():
    model = DEFAULT_MODEL_DIR / "face_landmarker.task"
    try:
        import mediapipe  # noqa: F401
        from src.landmarks import FaceLandmarker
        if not (hasattr(mediapipe, "solutions") and hasattr(mediapipe.solutions, "face_mesh")):
            if not model.exists():
                pytest.skip(f"no cached model at {model}; tests do not download it")
        FaceLandmarker(model_path=model if model.exists() else None).close()
    except pytest.skip.Exception:
        raise
    except Exception as exc:
        pytest.skip(f"real landmarker unavailable here: {type(exc).__name__}: {exc}")
    return lambda: FaceLandmarker(model_path=model if model.exists() else None)


def test_real_landmarker_runs_two_videos_in_sequence(two_videos, tmp_path):
    """The real timestamp contract, where MediaPipe and its model are installed.

    The frames are noise, so no face is found; what matters is that the second video
    does not raise. Skipped rather than failed on a machine without MediaPipe.
    """
    factory = _real_landmarker_or_skip()
    root, records = two_videos
    (done, _, failed), _ = run(root, records, tmp_path / "frames", factory)
    assert (done, failed) == (2, 0)
