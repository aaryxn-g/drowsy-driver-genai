"""ITA tests against hand-computed values, and both CLI passes with a stub landmarker.

Patches are built from known L*a*b* values using an inverse conversion written
here, independently of the module under test, and quantised to 8 bits as a real
frame would be.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import config, geometry, subgroups
from src.subgroups import FrameMeasure

REPO = Path(__file__).resolve().parents[1]
W, H = 640, 480


def lab_to_bgr8(l_star: float, a_star: float, b_star: float) -> np.ndarray:
    """Inverse CIELAB to 8-bit BGR, from the CIE and IEC 61966-2-1 definitions."""
    fy = (l_star + 16.0) / 116.0
    f = np.array([fy + a_star / 500.0, fy, fy - b_star / 200.0])
    delta = 6.0 / 29.0
    t = np.where(f > delta, f ** 3, 3.0 * delta ** 2 * (f - 4.0 / 29.0))
    xyz = t * np.array([0.95047, 1.0, 1.08883])
    m = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    linear = m @ xyz
    assert np.all((linear >= 0) & (linear <= 1)), "test colour is out of the sRGB gamut"
    c = np.where(linear <= 0.0031308, 12.92 * linear,
                 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.round(c * 255.0).astype(np.uint8)[::-1]


def face_pts() -> np.ndarray:
    """Landmarks with the three skin patches at fixed spots in a 640x480 frame."""
    pts = np.zeros((478, 3), dtype=float)
    for indices, (cx, cy) in zip(subgroups.SKIN_PATCHES,
                                 ((320, 120), (200, 280), (440, 280))):
        for k, idx in enumerate(indices):
            angle = 2 * math.pi * k / len(indices)
            pts[idx] = (cx + 40 * math.cos(angle), cy + 25 * math.sin(angle), 0.0)
    return pts


def uniform_frame(l_star, a_star, b_star) -> np.ndarray:
    return np.full((H, W, 3), lab_to_bgr8(l_star, a_star, b_star), dtype=np.uint8)


class TestLabConversion:
    def test_white_is_l100_neutral(self):
        lab = subgroups.srgb_to_lab(np.array([255, 255, 255]))
        assert lab == pytest.approx([100.0, 0.0, 0.0], abs=0.01)

    def test_black_is_zero(self):
        assert subgroups.srgb_to_lab(np.array([0, 0, 0])) == pytest.approx([0, 0, 0], abs=1e-9)

    def test_mid_grey_applies_the_gamma_decode(self):
        # By hand: 128/255 = 0.50196, decoded ((c + 0.055) / 1.055) ** 2.4 = 0.21586,
        # cbrt = 0.59985, L* = 116 * 0.59985 - 16 = 53.585.
        # Skipping the decode would give L* = 76.2.
        lab = subgroups.srgb_to_lab(np.array([128, 128, 128]))
        assert lab[0] == pytest.approx(53.585, abs=0.01)

    def test_published_srgb_red(self):
        lab = subgroups.srgb_to_lab(np.array([255, 0, 0]))
        assert lab == pytest.approx([53.24, 80.09, 67.20], abs=0.02)

    def test_dark_values_use_the_linear_segment(self):
        # By hand: 5/255 / 12.92 = 0.0015176, which is Y. It is below (6/29)^3, so
        # f = Y / (3 * (6/29)^2) + 4/29 and L* = 903.3 * Y = 1.371.
        lab = subgroups.srgb_to_lab(np.array([5, 5, 5]))
        assert lab[0] == pytest.approx(1.371, abs=0.01)


class TestItaFormula:
    # (L*, b*, ITA by hand)
    @pytest.mark.parametrize("l_star, b_star, expected", [
        (70.0, 20.0, 45.0),                   # atan(20/20)
        (50.0, 20.0, 0.0),                    # atan(0)
        (30.0, 20.0, -45.0),                  # atan(-20/20)
        (80.0, 10 * math.sqrt(3), 60.0),      # atan(30/17.32) = atan(sqrt 3)
        (60.0, 10 * math.sqrt(3), 30.0),      # atan(10/17.32) = atan(1/sqrt 3)
    ])
    def test_hand_values(self, l_star, b_star, expected):
        assert subgroups.ita_degrees(l_star, b_star) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("b_star", [0.0, -5.0])
    def test_non_positive_b_is_not_a_skin_reading(self, b_star):
        assert math.isnan(subgroups.ita_degrees(70.0, b_star))


class TestBands:
    @pytest.mark.parametrize("ita, label", [
        (55.01, "Very light"), (55.0, "Light"),
        (41.01, "Light"), (41.0, "Intermediate"),
        (28.01, "Intermediate"), (28.0, "Tan"),
        (10.01, "Tan"), (10.0, "Brown"),
        (-29.99, "Brown"), (-30.0, "Dark"),
        (-80.0, "Dark"),
    ])
    def test_protocol_boundaries(self, ita, label):
        assert subgroups.ita_category(ita) == label

    def test_nan_has_no_band(self):
        assert subgroups.ita_category(float("nan")) == ""


class TestMeasureFrame:
    # (L*, a*, b*, ITA by hand, band). 8-bit rounding is why the tolerance is not zero.
    @pytest.mark.parametrize("l_star, a_star, b_star, expected, band", [
        (80.0, 5.0, 10 * math.sqrt(3), 60.0, "Very light"),
        (70.0, 10.0, 20.0, 45.0, "Light"),
        (60.0, 10.0, 10 * math.sqrt(3), 30.0, "Intermediate"),
        (50.0, 12.0, 20.0, 0.0, "Brown"),
        (30.0, 10.0, 20.0, -45.0, "Dark"),
    ])
    def test_known_lab_patch(self, l_star, a_star, b_star, expected, band):
        m = measure_frame_uniform(l_star, a_star, b_star)
        assert m.l_star == pytest.approx(l_star, abs=0.3)
        assert m.b_star == pytest.approx(b_star, abs=0.5)
        assert m.ita == pytest.approx(expected, abs=1.0)
        assert subgroups.ita_category(m.ita) == band
        assert not m.clipped

    def test_tan_band_patch(self):
        # atan(5 / 18.66) = 15 degrees
        m = measure_frame_uniform(55.0, 10.0, 18.66)
        assert m.ita == pytest.approx(15.0, abs=1.0)
        assert subgroups.ita_category(m.ita) == "Tan"

    def test_no_face(self):
        m = subgroups.measure_frame(uniform_frame(70, 10, 20), None)
        assert m.failure == "no_face_in_reference" and math.isnan(m.ita)

    def test_patch_off_frame_is_a_failure_not_a_value(self):
        pts = face_pts()
        pts[list(subgroups.FOREHEAD), 1] -= 400
        m = subgroups.measure_frame(uniform_frame(70, 10, 20), pts)
        assert m.failure == "skin_patch_too_small" and math.isnan(m.ita)

    def test_patch_tracks_landmarks_not_fixed_pixels(self):
        pts = face_pts()
        frame = uniform_frame(30, 10, 20)
        mask = np.logical_or.reduce(
            [subgroups.patch_mask(frame.shape, pts, p) for p in subgroups.SKIN_PATCHES])
        frame[mask] = lab_to_bgr8(70, 10, 20)
        m = subgroups.measure_frame(frame, pts)
        assert m.ita == pytest.approx(45.0, abs=1.0)

    def test_grey_patch_is_not_scored_as_dark(self):
        # A neutral grey, or an IR frame, has b* of about 0.
        frame = np.full((H, W, 3), (200, 200, 200), dtype=np.uint8)
        frame[:, :, 0] = 205  # a slight blue cast makes b* negative
        m = subgroups.measure_frame(frame, face_pts())
        assert m.failure == "non_positive_b_star" and math.isnan(m.ita)


def measure_frame_uniform(l_star, a_star, b_star) -> FrameMeasure:
    return subgroups.measure_frame(uniform_frame(l_star, a_star, b_star), face_pts())


class TestClipping:
    @staticmethod
    def with_clipped_pixels(n: int, channel: int, value: int) -> tuple[np.ndarray, int]:
        """A 45-degree skin frame with n patch pixels clipped. Also returns the patch size."""
        pts = face_pts()
        frame = uniform_frame(70, 10, 20)
        mask = np.logical_or.reduce(
            [subgroups.patch_mask(frame.shape, pts, p) for p in subgroups.SKIN_PATCHES])
        ys, xs = np.nonzero(mask)
        frame[ys[:n], xs[:n], channel] = value
        return frame, len(ys)

    @pytest.mark.parametrize("value, channel", [(255, 0), (255, 1), (255, 2), (0, 0), (0, 2)])
    def test_clipping_is_judged_by_fraction(self, value, channel):
        pts = face_pts()
        _, total = self.with_clipped_pixels(0, channel, value)
        good = subgroups.measure_frame(uniform_frame(70, 10, 20), pts)

        # One stray pixel, as in compressed video: recorded, but still reliable.
        m = subgroups.measure_frame(self.with_clipped_pixels(1, channel, value)[0], pts)
        assert m.clipped_fraction == pytest.approx(1 / total)
        assert not m.clipped
        assert subgroups.summarise_subject("01", [good, good, m])["ita_reliable"] is True

        # Exactly at the limit is allowed; one pixel over it is not.
        at_limit = int(subgroups.MAX_CLIPPED_FRACTION * total)
        m = subgroups.measure_frame(self.with_clipped_pixels(at_limit, channel, value)[0], pts)
        assert not m.clipped
        m = subgroups.measure_frame(self.with_clipped_pixels(at_limit + 1, channel, value)[0], pts)
        assert m.clipped
        row = subgroups.summarise_subject("01", [good, good, m])
        assert row["ita_reliable"] is False
        assert "clipped_patch" in row["ita_unreliable_reason"]
        assert row["ita_spread"] < 1.0        # it is the clipping that trips the flag

        # With most of the patch clipped, ITA still comes from the clean pixels only.
        most = int(0.6 * total)
        m = subgroups.measure_frame(self.with_clipped_pixels(most, channel, value)[0], pts)
        assert m.clipped and m.clipped_fraction == pytest.approx(most / total)
        assert m.ita == pytest.approx(45.0, abs=1.0)

    def test_fully_blown_out_patch(self):
        frame = np.full((H, W, 3), 255, dtype=np.uint8)
        m = subgroups.measure_frame(frame, face_pts())
        assert m.clipped and m.clipped_fraction == 1.0
        assert m.failure == "all_patch_pixels_clipped" and math.isnan(m.ita)
        assert subgroups.summarise_subject("01", [m] * 3)["ita_reliable"] is False

    def test_clipping_outside_the_patch_is_ignored(self):
        frame = uniform_frame(70, 10, 20)
        frame[5, 5] = (255, 255, 255)
        assert not subgroups.measure_frame(frame, face_pts()).clipped


def at(ita: float) -> FrameMeasure:
    return FrameMeasure(ita=ita, l_star=70.0, b_star=20.0, clipped=False, clipped_fraction=0.0)


class TestSummary:
    def test_median_spread_and_band(self):
        row = subgroups.summarise_subject("01", [at(45.0), at(41.0), at(48.0)])
        assert row["ita_median"] == 45.0
        assert row["ita_spread"] == 7.0
        assert row["ita_category"] == "Light"
        assert row["ita_reliable"] is True
        assert (row["ita_0"], row["ita_1"], row["ita_2"]) == (45.0, 41.0, 48.0)

    def test_spread_of_exactly_ten_is_reliable(self):
        assert subgroups.summarise_subject("01", [at(40.0), at(45.0), at(50.0)])["ita_reliable"]

    def test_spread_over_ten_is_unreliable(self):
        row = subgroups.summarise_subject("01", [at(40.0), at(45.0), at(50.5)])
        assert row["ita_reliable"] is False
        assert "spread_exceeds_10_deg" in row["ita_unreliable_reason"]

    def test_too_few_candidates_is_unreliable_but_keeps_the_value(self):
        row = subgroups.summarise_subject("01", [at(45.0), at(47.0)])
        assert row["ita_reliable"] is False
        assert row["ita_median"] == 46.0
        assert math.isnan(row["ita_2"])

    def test_subject_with_no_frames_keeps_a_row(self):
        row = subgroups.summarise_subject("01", [])
        assert row["subject_id"] == "01"
        assert row["ita_reliable"] is False
        assert row["ita_category"] == "" and math.isnan(row["ita_median"])


def scan_table(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["frame", "t_s", "brightness", "detected", "yaw"])


class TestSelectCandidates:
    def test_rejects_turned_and_undetected_frames(self):
        scan = scan_table([
            (0, 0.0, 100.0, True, 30.0),
            (1, 20.0, 100.0, False, float("nan")),
            (2, 40.0, 100.0, True, -10.0),        # the limit is strict
            (3, 60.0, 100.0, True, 9.9),
        ])
        assert subgroups.select_candidates(scan)["frame"].tolist() == [3]

    def test_prefers_median_brightness_over_extremes(self):
        scan = scan_table([(i, i * 30.0, b, True, 0.0)
                           for i, b in enumerate([10, 99, 100, 101, 250])])
        picked = subgroups.select_candidates(scan)
        assert sorted(picked["frame"]) == [1, 2, 3]

    def test_median_includes_frames_that_are_not_eligible(self):
        # Median of all five is 100, though the frontal frames average 150.
        scan = scan_table([
            (0, 0.0, 100.0, True, 40.0), (1, 30.0, 100.0, True, 40.0),
            (2, 60.0, 100.0, False, float("nan")),
            (3, 90.0, 110.0, True, 0.0), (4, 120.0, 190.0, True, 0.0),
        ])
        assert subgroups.select_candidates(scan, n=1)["frame"].tolist() == [3]

    def test_candidates_are_separated_in_time(self):
        scan = scan_table([(i, float(i), 100.0, True, 0.0) for i in range(100)])
        t = sorted(subgroups.select_candidates(scan, min_gap_s=15.0)["t_s"])
        assert len(t) == 3 and min(np.diff(t)) >= 15.0

    def test_empty_scan(self):
        assert subgroups.select_candidates(scan_table([])).empty


def project_at_yaw(yaw_deg: float) -> np.ndarray:
    """A face whose pose points are projected at a known yaw, plus skin patches."""
    pts = face_pts()
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    k = geometry.camera_matrix(W, H)
    for idx, model_pt in zip(config.POSE_LANDMARKS, config.POSE_MODEL_3D):
        proj = k @ (rot @ np.asarray(model_pt) + np.array([0.0, 0.0, 1000.0]))
        pts[idx] = (proj[0] / proj[2], proj[1] / proj[2], 0.0)
    return pts


class ScriptedLandmarker:
    """Frontal except 0-20 s (turned) and 20-30 s (no face). Stills are frontal."""
    backend = "stub"

    def __call__(self, frame, timestamp_ms=0):
        t = timestamp_ms / 1000.0
        if t < 20.0 and timestamp_ms > 0:
            return project_at_yaw(35.0)
        if 20.0 <= t < 30.0:
            return None
        return project_at_yaw(0.0)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class AlwaysTurned(ScriptedLandmarker):
    def __call__(self, frame, timestamp_ms=0):
        return project_at_yaw(40.0)


def write_video(path: Path, seconds: int = 90, fps: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert w.isOpened()
    for i in range(seconds * fps):
        frame = uniform_frame(70, 10, 20)
        w.write(np.clip(frame.astype(int) + (i // fps) // 3 - 15, 0, 255).astype(np.uint8))
    w.release()


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "videos"
    rows = []
    for subject, fold in (("01", "1"), ("02", "1")):
        for stem, cls in (("0", "alert"), ("5", "low_vigilant"), ("10", "drowsy")):
            rel = f"Fold{fold}/{subject}/{stem}.mp4"
            if cls == "alert":
                write_video(root / rel)
            rows.append({"video_id": f"{subject}_{stem}", "subject_id": subject, "fold": fold,
                         "class": cls, "relpath": rel, "size_bytes": "0"})
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, root, tmp_path / "refs"


def run_refs(dataset) -> int:
    manifest, root, refs = dataset
    return subgroups.main(["extract-refs", "--manifest", str(manifest), "--root", str(root),
                           "--out", str(refs)])


def run_ita(tmp_path, *extra) -> int:
    return subgroups.main(["ita", "--refs", str(tmp_path / "refs"),
                           "--out", str(tmp_path / "ita.csv"), *extra])


class TestEndToEnd:
    def test_extract_refs_writes_candidates_and_log(self, dataset, monkeypatch):
        monkeypatch.setattr(subgroups, "FaceLandmarker", lambda **k: ScriptedLandmarker())
        assert run_refs(dataset) == 0
        refs = dataset[2]
        names = sorted(p.name for p in refs.glob("*.png"))
        assert names == [f"{s}_{i}.png" for s in ("01", "02") for i in range(3)]
        assert cv2.imread(str(refs / "01_0.png")).shape == (H, W, 3)

        log = pd.read_csv(refs / subgroups.REFS_LOG_NAME, dtype=str).set_index("subject_id")
        assert log.loc["01", "alert_video"] == "01_0"
        assert log.loc["01", "clip_ids"] == "01_0;01_10;01_5"
        frames = sorted(int(f) for f in log.loc["01", "frames"].split(";"))
        # Frames from the turned and no-face stretches (under 30 s at 5 fps) are out.
        assert all(f >= 30 * 5 for f in frames)
        assert min(np.diff(frames)) >= 15 * 5

    def test_extract_refs_resumes_without_rescanning(self, dataset, monkeypatch):
        monkeypatch.setattr(subgroups, "FaceLandmarker", lambda **k: ScriptedLandmarker())
        assert run_refs(dataset) == 0

        def must_not_build(**k):
            raise AssertionError("a logged subject was reprocessed")
        monkeypatch.setattr(subgroups, "FaceLandmarker", must_not_build)
        assert run_refs(dataset) == 0

    def test_subject_without_frontal_frames_is_kept_and_flagged(self, dataset, monkeypatch, tmp_path):
        monkeypatch.setattr(subgroups, "FaceLandmarker", lambda **k: AlwaysTurned())
        assert run_refs(dataset) == 1            # the shortfall fails loudly
        log = pd.read_csv(dataset[2] / subgroups.REFS_LOG_NAME, dtype=str)
        assert set(log["status"]) == {"too_few_frontal_frames"}
        assert not list(dataset[2].glob("*.png"))

        assert run_ita(tmp_path) == 0
        ita = pd.read_csv(tmp_path / "ita.csv", dtype={"subject_id": str})
        assert ita["subject_id"].tolist() == ["01", "02"]
        assert not ita["ita_reliable"].any()
        assert ita["ita_median"].isna().all()

    def test_ita_and_sheets_from_real_files(self, dataset, monkeypatch, tmp_path):
        monkeypatch.setattr(subgroups, "FaceLandmarker", lambda **k: ScriptedLandmarker())
        assert run_refs(dataset) == 0
        assert run_ita(tmp_path) == 0

        ita = pd.read_csv(tmp_path / "ita.csv", dtype={"subject_id": str})
        assert len(ita) == 2
        for col in ("ita_0", "ita_1", "ita_2", "ita_median", "ita_spread", "ita_reliable"):
            assert col in ita.columns
        assert ita["ita_median"].notna().all()

        sheet = pd.read_csv(tmp_path / "subgroup_annotations_prefilled.csv",
                            dtype=str, keep_default_na=False)
        assert list(sheet.columns[:len(subgroups.SCHEMA_COLUMNS)]) == subgroups.SCHEMA_COLUMNS
        assert sheet.loc[0, "subject_id"] == "01"
        assert sheet.loc[0, "dataset_source"] == "UTA-RLDD"
        assert sheet.loc[0, "clip_ids"] == "01_0;01_10;01_5"
        assert sheet.loc[0, "ita_value_deg"] != ""
        assert sheet.loc[0, "reference_frames"] == "01_0.png;01_1.png;01_2.png"

    def test_rerun_will_not_overwrite_hand_labels(self, dataset, monkeypatch, tmp_path):
        monkeypatch.setattr(subgroups, "FaceLandmarker", lambda **k: ScriptedLandmarker())
        assert run_refs(dataset) == 0
        assert run_ita(tmp_path) == 0
        sheet = tmp_path / "subgroup_annotations_prefilled.csv"
        sheet.write_text("hand labels")

        assert run_ita(tmp_path) == 2
        assert sheet.read_text() == "hand labels"
        assert run_ita(tmp_path, "--overwrite-sheets") == 0
        assert sheet.read_text() != "hand labels"

    def test_missing_refs_folder_stops_without_output(self, tmp_path):
        assert run_ita(tmp_path) == 2
        assert not (tmp_path / "ita.csv").exists()


class TestSheets:
    def build(self, n=60):
        ita = pd.DataFrame([
            subgroups.summarise_subject(f"{i:02d}", [at(45.0), at(46.0), at(47.0)])
            for i in range(1, n + 1)
        ])
        log = pd.DataFrame({"subject_id": ita["subject_id"], "clip_ids": "c", "status": "ok"})
        return subgroups.build_sheets(ita, log, Path("/nonexistent"), seed=0)

    def test_schema_constant_matches_the_schema_file(self):
        header = (REPO / "data" / "subgroup_annotation_schema.csv").read_text().strip().split(",")
        assert subgroups.SCHEMA_COLUMNS == header

    def test_human_judgements_are_all_blank(self):
        master, blind = self.build()
        blank = (["age_bracket", "gender_presentation", "glasses", "annotator_1",
                  "annotator_2", "agreement_flag", "notes"]
                 + subgroups.ANNOTATOR_1_COLUMNS + subgroups.ANNOTATOR_2_COLUMNS)
        assert (master[blank] == "").all().all()
        assert (blind.drop(columns=["subject_id", "reference_frames"]) == "").all().all()

        # Skin tone is computed: nobody labels it on the sheet that shows the ITA
        # value, and the only human input is the second annotator's spot-check.
        assert not [c for c in subgroups.ANNOTATOR_1_COLUMNS if "skin" in c]
        assert "skin_tone_spot_check" in blind.columns

    def test_every_subject_on_master_and_twenty_percent_on_blind(self):
        master, blind = self.build()
        assert len(master) == 60 and len(blind) == 12
        assert set(blind["subject_id"]) == set(
            master.loc[master["second_annotator_sample"], "subject_id"])

    def test_blind_sheet_shows_no_ita(self):
        _, blind = self.build()
        assert not [c for c in blind.columns if "ita" in c or "skin_tone_ita" in c]
        assert not [c for c in blind.columns if c.endswith("_a1")]

    def test_sample_is_reproducible(self):
        assert self.build()[1]["subject_id"].tolist() == self.build()[1]["subject_id"].tolist()
