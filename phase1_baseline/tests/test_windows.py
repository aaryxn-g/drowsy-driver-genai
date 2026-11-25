"""Window-level features (aggregate --window-s) and the window-level detector path."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import aggregate, blinks, detector

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0


def frames(seconds: float, open_ear: float, nan_spans=()) -> pd.DataFrame:
    """A per-frame table: eye at `open_ear`, a 0.2 s blink to 0.05 every 3 s."""
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    ear = np.where((np.arange(n) % 90) < 6, 0.05, open_ear)
    for a, b in nan_spans:
        ear[(t >= a) & (t < b)] = np.nan
    return pd.DataFrame({
        "frame": np.arange(n), "t_s": t, "detected": np.isfinite(ear),
        "ear": ear, "ear_corrected": ear, "mar": np.full(n, 0.05),
        "pitch": np.zeros(n), "pitch_centered": np.zeros(n),
        "nose_x_norm": np.full(n, 0.5), "nose_y_norm": np.full(n, 0.5),
        "face_scale": np.full(n, 0.2),
    })


def write_subject(tmp_path: Path) -> Path:
    """Subject 07: an alert video with wide-open eyes and a drowsy one that never opens fully."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames(120, 0.30).to_parquet(frames_dir / "07_0.parquet")
    frames(120, 0.15).to_parquet(frames_dir / "07_10.parquet")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"video_id": "07_0", "subject_id": "07", "fold": "1", "class": "alert", "relpath": "x"},
        {"video_id": "07_10", "subject_id": "07", "fold": "1", "class": "drowsy", "relpath": "x"},
    ]).to_csv(manifest, index=False)
    return manifest


def run_aggregate(tmp_path, manifest, *extra):
    proc = subprocess.run(
        [sys.executable, "-m", "src.aggregate", "--manifest", str(manifest),
         "--frames", str(tmp_path / "frames"), "--out", str(tmp_path / "video_features.csv"),
         *extra], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


# ----------------------------------------------------------------- cutting
@pytest.mark.parametrize("duration, w, s, expected", [
    (100, 30, 15, [0, 15, 30, 45, 60]),   # 75 + 30 > 100: trailing partial not cut
    (60, 30, 15, [0, 15, 30]),            # exact fit keeps the last window
    (20, 30, 15, []),                     # shorter than one window
])
def test_window_starts(duration, w, s, expected):
    np.testing.assert_allclose(aggregate.window_starts(duration, w, s), expected)


def test_window_over_whole_video_matches_video_features():
    df = frames(60, 0.30)
    base = blinks.estimate_baseline(df["ear_corrected"].to_numpy(), source="alert")
    rows, dropped = aggregate.window_features(df, base, 60, 60)
    video, _ = aggregate.summarise_frames(df, base, aggregate.video_fps(df), True)
    assert dropped == 0 and len(rows) == 1
    for k, v in video.items():
        assert rows[0][k] == pytest.approx(v, nan_ok=True), k


# --------------------------------------------------------- baseline invariant
def test_windows_use_the_subjects_alert_baseline(tmp_path):
    out = run_aggregate(tmp_path, write_subject(tmp_path), "--window-s", "30")
    assert "dropped 0 windows" in out
    w = pd.read_csv(tmp_path / "window_features_w30_s15.csv", dtype={"subject_id": str})
    alert = blinks.estimate_baseline(frames(120, 0.30)["ear"].to_numpy(), source="alert")
    assert w["ear_open_baseline"].unique() == pytest.approx([alert.ear_open])
    assert (w["baseline_source"] == "alert").all()
    # A per-window baseline would put the drowsy eye back near fully open (~0.9).
    # Against the alert ruler it sits about 40% open.
    drowsy = w[w["class"] == "drowsy"]
    assert drowsy["mean_openness"].max() < 0.5
    assert (w[w["class"] == "alert"]["mean_openness"] > 0.8).all()


def test_baseline_assertion_fires_on_a_per_window_baseline():
    base = blinks.EyeBaseline(0.30, 0.05, "alert", 1000)
    rows = pd.DataFrame({"subject_id": ["07"] * 3, "baseline_source": ["alert"] * 3,
                         "ear_open_baseline": [0.30, 0.30, 0.17],
                         "ear_closed_baseline": [0.05, 0.05, 0.05]})
    with pytest.raises(aggregate.BaselineInvariantError, match="07"):
        aggregate.assert_subject_baselines(rows, {"07": base})
    rows["ear_open_baseline"] = 0.17  # consistent, but not the subject's
    with pytest.raises(aggregate.BaselineInvariantError):
        aggregate.assert_subject_baselines(rows, {"07": base})


# --------------------------------------------------------------- coverage
def test_low_coverage_windows_are_dropped_and_counted():
    base = blinks.EyeBaseline(0.30, 0.05, "alert", 1000)
    # 90 s in three 30 s windows: the first is 80% untracked, the second 20%.
    df = frames(90, 0.30, nan_spans=[(0, 24), (30, 36)])
    rows, dropped = aggregate.window_features(df, base, 30, 30)
    assert dropped == 1
    assert [r["window_start_s"] for r in rows] == [30.0, 60.0]
    assert rows[0]["coverage"] == pytest.approx(0.8, abs=0.01)


def test_cli_reports_drops_in_meta(tmp_path):
    manifest = write_subject(tmp_path)
    frames(120, 0.15, nan_spans=[(0, 30)]).to_parquet(tmp_path / "frames" / "07_10.parquet")
    out = run_aggregate(tmp_path, manifest, "--window-s", "30", "--stride-s", "30")
    meta = json.loads((tmp_path / "window_features_w30_s30.meta.json").read_text())
    assert meta["windows_dropped_low_coverage"] == 1
    assert meta["dropped_by_subject"]["07"]["dropped"] == 1
    assert "dropped 1 windows" in out


def test_window_mode_leaves_video_features_alone(tmp_path):
    run_aggregate(tmp_path, write_subject(tmp_path), "--window-s", "30", "15")
    assert not (tmp_path / "video_features.csv").exists()
    assert (tmp_path / "window_features_w30_s15.csv").exists()
    assert (tmp_path / "window_features_w15_s7.5.csv").exists()


# ------------------------------------------------------- detector, windows
def test_video_vote_majority_and_tie_break():
    pred = pd.DataFrame({
        "video_id": ["a"] * 3 + ["b"] * 2, "subject_id": "01", "fold": 1,
        "class": ["drowsy"] * 3 + ["alert"] * 2,
        "pred": ["drowsy", "drowsy", "alert", "alert", "drowsy"],
        "p_alert": [0.2, 0.3, 0.6, 0.55, 0.1], "p_drowsy": [0.8, 0.7, 0.4, 0.45, 0.9],
    })
    v = detector.video_vote(pred, ["alert", "drowsy"]).set_index("video_id")
    assert v.loc["a", "pred"] == "drowsy" and v.loc["a", "n_windows"] == 3
    # b is a 1-1 tie; mean p_drowsy 0.675 beats mean p_alert 0.325
    assert v.loc["b", "pred"] == "drowsy"


def test_choose_window_picks_shortest_within_tolerance():
    t = pd.DataFrame({"window_s": [60, 30, 15, 10],
                      "window_macro_f1": [0.55, 0.545, 0.54, 0.50],
                      "video_accuracy": [0.62, 0.61, 0.605, 0.62]})
    assert detector.choose_window(t, 0.02)["window_s"] == 15
    t.loc[2, "video_accuracy"] = 0.59  # 15 s now 3 points short on video accuracy
    assert detector.choose_window(t, 0.02)["window_s"] == 30


def window_table(n_subjects=10, per_video=4, seed=0):
    rows = []
    for _, r in pd.DataFrame(
            __import__("tests.test_detector", fromlist=["x"]).synthetic_table(n_subjects, seed)
    ).iterrows():
        for k in range(per_video):
            rows.append({**r.to_dict(), "window_start_s": 15.0 * k, "window_s": 30.0,
                         "stride_s": 15.0, "duration_s": 30.0})
    return pd.DataFrame(rows)


def test_window_leaky_split_still_raises():
    df = window_table()
    df.loc[df.video_id == "02_1", "fold"] = 5
    with pytest.raises(detector.SubjectLeakageError):
        detector.cross_validate(df, "3class", "hgb", 0)


def test_run_windows_logs_window_and_video_metrics(tmp_path):
    path = tmp_path / "window_features_w30_s15.csv"
    window_table().to_csv(path, index=False)
    table = detector.run_windows([path], annotations_path=tmp_path / "none.csv",
                                 results_dir=tmp_path, seeds=(0,), n_boot=20)
    runs = pd.read_csv(tmp_path / "detector_window_runs.csv")
    assert len(runs) == 4 and runs.run_id.is_unique  # 2 tasks x 2 models
    three = runs[runs.task == "3class"].iloc[0]
    assert three.n_windows == 120 and three.n_videos == 30
    votes = pd.read_csv(tmp_path / "detector_window_votes.csv")
    assert (votes.n_windows == 4).all()
    assert set(table.columns) >= {"window_macro_f1", "video_accuracy"}
