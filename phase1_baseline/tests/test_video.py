"""Rotation handling and shard selection.

cv2.VideoWriter cannot write a rotation flag, so there is no rotated fixture video.
The rotation functions are tested directly on arrays, and the capture logic against
FakeCapture, which models how the FFmpeg backend behaves, including reporting the
rotated size from CAP_PROP_FRAME_WIDTH while AUTO is on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import aggregate, config, extract, landmarks, manifest, video

from .test_pipeline import StubLandmarker


def marked_frame(w: int = 8, h: int = 4) -> np.ndarray:
    """A frame with one white pixel at the top left, so orientation is visible."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0, 0] = 255
    return frame


def marker_position(frame: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(frame[:, :, 0])
    return int(xs[0]), int(ys[0])


class TestApplyRotation:
    @pytest.mark.parametrize("degrees, shape, position", [
        (0, (4, 8), (0, 0)),
        (90, (8, 4), (3, 0)),      # clockwise: top left goes to top right
        (180, (4, 8), (7, 3)),
        (270, (8, 4), (0, 7)),     # counter-clockwise: top left goes to bottom left
    ])
    def test_right_angles(self, degrees, shape, position):
        out = video.apply_rotation(marked_frame(), degrees)
        assert out.shape[:2] == shape
        assert marker_position(out) == position

    @pytest.mark.parametrize("negative, positive", [
        (-90, 270), (-180, 180), (-270, 90), (-360, 0), (450, 90),
    ])
    def test_angles_outside_the_range_wrap(self, negative, positive):
        assert np.array_equal(video.apply_rotation(marked_frame(), negative),
                              video.apply_rotation(marked_frame(), positive))

    @pytest.mark.parametrize("bad", [45, 89.5, 1])
    def test_other_angles_are_refused(self, bad):
        with pytest.raises(ValueError):
            video.apply_rotation(marked_frame(), bad)

    def test_float_angle_from_cap_get(self):
        assert video.normalise_rotation(270.0) == 270


class TestClassifyRotation:
    def test_unrotated(self):
        assert video.classify_rotation((640, 480), (640, 480), 0) == "none"

    def test_decoder_rotated_it(self):
        assert video.classify_rotation((480, 640), (640, 480), 90) == "auto"

    def test_decoder_ignored_it(self):
        assert video.classify_rotation((640, 480), (640, 480), 270) == "manual"

    @pytest.mark.parametrize("actual, stored, degrees", [
        ((100, 100), (640, 480), 90),     # neither stored nor predicted
        ((100, 100), (640, 480), 0),
        ((480, 640), (640, 480), 0),      # sideways with no flag to explain it
        ((480, 640), (640, 480), 180),
    ])
    def test_disagreement_fires(self, actual, stored, degrees):
        with pytest.raises(video.RotationCheckError):
            video.classify_rotation(actual, stored, degrees)

    @pytest.mark.parametrize("actual, stored, degrees", [
        ((640, 480), (640, 480), 180),
        ((500, 500), (500, 500), 90),
    ])
    def test_shape_cannot_tell(self, actual, stored, degrees):
        assert video.classify_rotation(actual, stored, degrees) is None


class FakeCapture:
    """The parts of cv2.VideoCapture that OrientedVideo uses.

    `honours_auto` False models a build that ignores AUTO and returns stored frames.
    `wrong_shape` models a decoder that returns something nobody predicted.
    """

    def __init__(self, frames, rotation=0, honours_auto=True, default_auto=1, wrong_shape=False):
        self.frames = frames
        self.rotation = rotation
        self.honours_auto = honours_auto
        self.auto = default_auto if honours_auto else 0
        self.wrong_shape = wrong_shape
        self.pos = 0

    def isOpened(self):
        return True

    def release(self):
        pass

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO and self.honours_auto:
            self.auto = int(value)
            return True
        return False

    def get(self, prop):
        h, w = self.frames[0].shape[:2]
        if self.auto and self.rotation % 180:
            w, h = h, w
        return {
            cv2.CAP_PROP_ORIENTATION_META: self.rotation,
            cv2.CAP_PROP_ORIENTATION_AUTO: self.auto,
            cv2.CAP_PROP_FRAME_WIDTH: w,
            cv2.CAP_PROP_FRAME_HEIGHT: h,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_BITRATE: 900.0,
        }[prop]

    def read(self):
        if self.pos >= len(self.frames):
            return False, None
        frame = self.frames[self.pos]
        self.pos += 1
        if self.wrong_shape:
            return True, frame[:2, :2]
        return True, video.apply_rotation(frame, self.rotation) if self.auto else frame


def install(monkeypatch, frames, **kwargs):
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda *a, **k: FakeCapture(frames, **kwargs))


def read_all(path="fake.mp4"):
    with video.OrientedVideo(Path(path)) as source:
        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)
        return source.info, frames


class TestOrientedVideo:
    @pytest.mark.parametrize("degrees, position", [(90, (3, 0)), (270, (0, 7))])
    @pytest.mark.parametrize("honours_auto, method", [(True, "auto"), (False, "manual")])
    def test_quarter_turns_come_out_upright(self, monkeypatch, degrees, position,
                                            honours_auto, method):
        install(monkeypatch, [marked_frame()] * 3, rotation=degrees, honours_auto=honours_auto)
        info, frames = read_all()
        assert info.rotation_method == method
        assert info.rotation_deg == degrees
        assert (info.stored_width, info.stored_height) == (8, 4)
        assert (info.effective_width, info.effective_height) == (4, 8)
        assert len(frames) == 3, "the frame read for the check must not be lost"
        assert all(marker_position(f) == position for f in frames)

    def test_frame_width_is_not_trusted_while_auto_is_on(self, monkeypatch):
        """Predicting from it would swap twice and rotate an upright frame again."""
        install(monkeypatch, [marked_frame()], rotation=90, default_auto=1)
        info, frames = read_all()
        assert info.stored_width == 8 and info.rotation_method == "auto"
        assert frames[0].shape[:2] == (8, 4)

    @pytest.mark.parametrize("honours_auto, method", [(True, "auto"), (False, "manual")])
    def test_180_is_settled_by_pixels(self, monkeypatch, honours_auto, method):
        install(monkeypatch, [marked_frame()] * 2, rotation=180, honours_auto=honours_auto)
        info, frames = read_all()
        assert info.rotation_method == method
        assert all(marker_position(f) == (7, 3) for f in frames)

    def test_flat_opening_frames_do_not_settle_it(self, monkeypatch):
        flat = np.zeros((4, 8, 3), dtype=np.uint8)
        install(monkeypatch, [flat, flat, marked_frame()], rotation=180, honours_auto=False)
        info, frames = read_all()
        assert info.rotation_method == "manual"
        assert len(frames) == 3

    def test_undecidable_180_refuses_to_guess(self, monkeypatch):
        install(monkeypatch, [np.zeros((4, 8, 3), dtype=np.uint8)] * 2, rotation=180)
        with pytest.raises(video.RotationCheckError):
            read_all()

    def test_unexpected_shape_raises(self, monkeypatch):
        install(monkeypatch, [marked_frame()], rotation=90, wrong_shape=True)
        with pytest.raises(video.RotationCheckError):
            read_all()

    def test_shape_change_mid_video_raises(self, monkeypatch):
        install(monkeypatch, [marked_frame(), marked_frame(w=6)], rotation=0)
        with video.OrientedVideo(Path("fake.mp4")) as source:
            source.read()
            with pytest.raises(video.RotationCheckError):
                source.read()

    def test_covariates(self, monkeypatch):
        install(monkeypatch, [marked_frame()] * 60)
        info, _ = read_all()
        assert (info.rotation_method, info.fps, info.duration_s, info.bitrate_kbps) == (
            "none", 30.0, 2.0, 900.0)


class Recording(StubLandmarker):
    """A stub landmarker that remembers the frame sizes it was given."""

    def __init__(self):
        super().__init__(lambda i: 1.0)
        self.seen = []

    def __call__(self, frame, timestamp_ms=0):
        self.seen.append(frame.shape[:2])
        return super().__call__(frame, timestamp_ms)


class TestProcessVideoRotation:
    @pytest.mark.parametrize("honours_auto", [True, False])
    @pytest.mark.parametrize("stored_shape, seen_shape", [
        # Upright width 480 is within the limit. Downscaling first would give 362x640.
        ((480, 848, 3), (848, 480)),
        # Upright width 848 is over it. Stored width 480 is not, so downscaling first
        # would do nothing and hand over 848x480 at full size.
        ((848, 480, 3), (362, 640)),
    ])
    def test_rotation_happens_before_the_downscale(self, monkeypatch, honours_auto,
                                                   stored_shape, seen_shape):
        install(monkeypatch, [np.zeros(stored_shape, dtype=np.uint8)], rotation=90,
                honours_auto=honours_auto)
        stub = Recording()
        extract.process_video(Path("fake.mp4"), stub, max_width=640)
        assert stub.seen == [seen_shape]

    @pytest.mark.parametrize("honours_auto", [True, False])
    def test_landmarker_sees_the_upright_frame_and_meta_says_so(self, monkeypatch, honours_auto):
        stored = [np.zeros((480, 848, 3), dtype=np.uint8)] * 2
        install(monkeypatch, stored, rotation=90, honours_auto=honours_auto)
        stub = Recording()
        df, meta = extract.process_video(Path("fake.mp4"), stub, max_width=640)
        assert stub.seen == [(848, 480), (848, 480)]
        assert len(df) == 2
        assert meta["rotation_method"] == ("auto" if honours_auto else "manual")
        assert (meta["stored_width"], meta["stored_height"]) == (848, 480)
        assert (meta["effective_width"], meta["effective_height"]) == (480, 848)
        assert (meta["inference_width"], meta["inference_height"]) == (480, 848)
        assert not meta["low_resolution"]

    def test_every_recorded_column_is_in_the_meta(self, monkeypatch):
        install(monkeypatch, [marked_frame(640, 240)])
        _, meta = extract.process_video(Path("fake.mp4"), StubLandmarker(lambda i: 1.0))
        assert set(video.INFO_COLUMNS) <= set(meta)
        assert meta["low_resolution"]


class TestManifestCovariates:
    def test_rows_carry_the_probe(self, tmp_path, monkeypatch):
        d = tmp_path / "Fold1_part1" / "01"
        d.mkdir(parents=True)
        (d / "0.mp4").write_bytes(b"x")
        install(monkeypatch, [marked_frame()], rotation=90)
        rows, _, _ = manifest.scan(tmp_path)
        assert rows[0]["rotation_deg"] == 90
        assert rows[0]["effective_width"] == 4
        assert rows[0]["rotation_method"] == "auto"
        assert rows[0]["probe_error"] == ""

    def test_unreadable_video_stays_in_the_manifest(self, tmp_path):
        d = tmp_path / "Fold1_part1" / "01"
        d.mkdir(parents=True)
        (d / "0.mp4").write_bytes(b"x")
        rows, _, _ = manifest.scan(tmp_path)
        assert len(rows) == 1
        assert rows[0]["probe_error"]
        assert rows[0]["rotation_deg"] == ""
        assert "could not probe   1" in manifest.summarise(rows)


class TestLowResolution:
    def test_threshold_is_on_the_short_side(self):
        limit = config.MIN_RESOLUTION_SHORT_SIDE
        assert video.is_low_resolution(240, 426)
        assert video.is_low_resolution(426, 240)
        assert not video.is_low_resolution(848, 480)
        assert not video.is_low_resolution(limit, 2 * limit)
        assert video.is_low_resolution(limit - 1, 2 * limit)

    def test_manifest_strings_are_parsed(self):
        out = aggregate.resolution_features(
            {"effective_width": "240", "effective_height": "426", "rotation_deg": "90"})
        assert out["low_resolution"] is True
        assert out["rotation_deg"] == "90"

    @pytest.mark.parametrize("row", [{}, {"effective_width": "", "effective_height": ""}])
    def test_unknown_is_not_reported_as_fine(self, row):
        assert aggregate.resolution_features(row)["low_resolution"] is None

    def test_low_resolution_subject_is_flagged_and_kept(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(0)
        frames = tmp_path / "frames"
        frames.mkdir()
        n = 900
        ear = 0.30 + rng.normal(0, 0.002, n)
        ear[::20] = 0.05
        columns = {c: np.zeros(n) for c in extract.FEATURE_COLUMNS}
        columns.update(frame=np.arange(n), t_s=np.arange(n) / 30.0, detected=True,
                       ear=ear, ear_corrected=ear, nose_y_norm=rng.normal(0, 0.01, n))
        rows = []
        for subject, (w, h) in {"01": (240, 426), "02": (848, 480)}.items():
            video_id = f"{subject}_0"
            pd.DataFrame(columns).to_parquet(frames / f"{video_id}.parquet", index=False)
            rows.append({"video_id": video_id, "subject_id": subject, "fold": "1",
                         "class": "alert", "relpath": "x.mp4", "rotation_deg": 0,
                         "rotation_method": "none", "stored_width": w, "stored_height": h,
                         "effective_width": w, "effective_height": h, "bitrate_kbps": 900})
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        out_path = tmp_path / "features.csv"
        monkeypatch.setattr(sys, "argv", [
            "aggregate", "--manifest", str(manifest_path), "--frames", str(frames),
            "--out", str(out_path), "--events-out", str(tmp_path / "events.csv")])
        assert aggregate.main() == 0

        out = pd.read_csv(out_path, dtype={"subject_id": str}).set_index("subject_id")
        assert set(out.index) == {"01", "02"}
        assert out.loc["01", "low_resolution"]
        assert not out.loc["02", "low_resolution"]
        assert out.loc["01", "effective_width"] == 240
        columns_out = list(out.columns)
        assert columns_out.index("low_resolution") == columns_out.index("low_quality") + 1
        assert '"low_resolution_subjects": [\n    "01"\n  ]' in (
            out_path.with_suffix(".meta.json").read_text())


class TestSharding:
    @pytest.fixture
    def table(self):
        return pd.DataFrame({"video_id": [f"v{i}" for i in range(180)]})

    @pytest.mark.parametrize("n", [1, 2, 7, 16, 180, 200])
    def test_every_row_exactly_once(self, table, n):
        seen = [v for i in range(1, n + 1)
                for v in extract.select_shard(table, i, n)["video_id"]]
        assert len(seen) == len(table), "no row twice, no row missed"
        assert set(seen) == set(table["video_id"])

    def test_rule_is_row_index_mod_n(self, table):
        positions = extract.select_shard(table, 3, 16).index
        assert all(p % 16 == 2 for p in positions)
        assert list(positions[:2]) == [2, 18]

    def test_n_equal_1_is_the_whole_table_and_the_usual_log(self, table):
        pd.testing.assert_frame_equal(extract.select_shard(table, 1, 1), table)
        assert extract.log_filename(1, 1) == "_extraction_log.csv"

    def test_shard_log_name(self):
        assert extract.log_filename(3, 16) == "_extraction_log_shard3of16.csv"

    def test_parse(self):
        assert extract.parse_shard("3/16") == (3, 16)
        assert extract.parse_shard("1/1") == (1, 1)

    @pytest.mark.parametrize("bad", ["0/16", "17/16", "3", "a/b", "3/0", "-1/4", "1/2/3"])
    def test_parse_refuses(self, bad):
        with pytest.raises(argparse.ArgumentTypeError):
            extract.parse_shard(bad)


class TestMergeLogs:
    def write(self, path, rows):
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_combines_and_keeps_shard_logs(self, tmp_path):
        a = tmp_path / extract.log_filename(1, 2)
        b = tmp_path / extract.log_filename(2, 2)
        self.write(a, [{"video_id": "v2", "detection_rate": 0.5}])
        self.write(b, [{"video_id": "v1", "detection_rate": 0.9}])
        merged = extract.merge_shard_logs(tmp_path)
        assert list(merged["video_id"]) == ["v1", "v2"]
        assert list(pd.read_csv(tmp_path / "_extraction_log.csv")["video_id"]) == ["v1", "v2"]
        assert a.exists() and b.exists()

    def test_redo_keeps_the_newest_row_and_merging_twice_is_stable(self, tmp_path):
        self.write(tmp_path / "_extraction_log.csv", [
            {"video_id": "v0", "detection_rate": 0.1},
            {"video_id": "v1", "detection_rate": 0.1},
        ])
        self.write(tmp_path / extract.log_filename(1, 2), [
            {"video_id": "v1", "detection_rate": 0.2},
            {"video_id": "v1", "detection_rate": 0.3},
        ])
        for _ in range(2):
            merged = extract.merge_shard_logs(tmp_path).set_index("video_id")
            assert merged["detection_rate"].to_dict() == {"v0": "0.1", "v1": "0.3"}

    def test_no_shard_logs_is_an_error(self, tmp_path):
        assert extract.run_merge(tmp_path, None) == 1

    def test_reports_videos_with_no_log_row(self, tmp_path, capsys):
        self.write(tmp_path / extract.log_filename(1, 2), [{"video_id": "v1"}])
        manifest_path = tmp_path / "manifest.csv"
        self.write(manifest_path, [{"video_id": "v1"}, {"video_id": "v2"}])
        assert extract.run_merge(tmp_path, manifest_path) == 1
        assert "v2" in capsys.readouterr().out


class TestShardedCli:
    def test_two_shards_cover_the_manifest_with_separate_logs(self, tmp_path, monkeypatch):
        clip = tmp_path / "root" / "clip.mp4"
        clip.parent.mkdir()
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
        for _ in range(3):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()

        rows = [{"video_id": f"v{i}", "subject_id": "01", "fold": "1", "class": "alert",
                 "relpath": "clip.mp4"} for i in range(5)]
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        out = tmp_path / "frames"

        class Landmarker(StubLandmarker):
            def __init__(self, model_path=None):
                super().__init__(lambda i: 1.0)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

        monkeypatch.setattr(extract, "FaceLandmarker", Landmarker)

        def run(*extra):
            monkeypatch.setattr(sys, "argv", [
                "extract", "--manifest", str(manifest_path), "--root", str(clip.parent),
                "--out", str(out), *extra])
            return extract.main()

        assert run("--shard", "1/2") == 0
        assert sorted(p.stem for p in out.glob("v*.parquet")) == ["v0", "v2", "v4"]
        assert run("--shard", "2/2") == 0
        assert sorted(p.stem for p in out.glob("v*.parquet")) == [f"v{i}" for i in range(5)]
        assert len(pd.read_csv(out / "_extraction_log_shard1of2.csv")) == 3
        assert len(pd.read_csv(out / "_extraction_log_shard2of2.csv")) == 2
        assert not (out / "_extraction_log.csv").exists()

        monkeypatch.setattr(sys, "argv", [
            "extract", "--merge-logs", "--out", str(out), "--manifest", str(manifest_path)])
        assert extract.main() == 0
        assert len(pd.read_csv(out / "_extraction_log.csv")) == 5


class TestModelDownload:
    def test_a_concurrent_reader_never_sees_a_partial_model(self, tmp_path, monkeypatch):
        target = tmp_path / "face_landmarker.task"

        def fake_download(url, dest):
            assert not target.exists(), "must not write to the final path directly"
            Path(dest).write_bytes(b"model")

        monkeypatch.setattr(landmarks.urllib.request, "urlretrieve", fake_download)
        assert landmarks.ensure_model(target) == target
        assert target.read_bytes() == b"model"
        assert list(tmp_path.iterdir()) == [target]
