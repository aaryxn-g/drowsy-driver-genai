"""Detector tests: the leakage guard, feature selection, metrics, and the subgroup hook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src import detector
from src.detector import SubjectLeakageError


def synthetic_table(n_subjects: int = 10, seed: int = 0) -> pd.DataFrame:
    """A features table shaped like video_features.csv, with every real column."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for k, cls in enumerate(detector.CLASSES["3class"]):
            row = {"video_id": f"{s:02d}_{k}", "subject_id": f"{s:02d}",
                   "fold": s % 5 + 1, "class": cls, "duration_s": 600.0 + 60 * k}
            for c in detector.RATE_FEATURES + detector.COUNT_FEATURES:
                row[c] = rng.normal(k, 0.3)
            for c in detector.EXCLUDED:
                row.setdefault(c, rng.normal())
            row["n_blinks"] = 100
            row["baseline_source"] = "alert"
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------- leakage guard
def test_leaky_split_raises():
    df = synthetic_table()
    df.loc[df.video_id == "03_2", "fold"] = 1  # subject 03 is in fold 4; now also in fold 1
    with pytest.raises(SubjectLeakageError, match="03"):
        detector.assert_subject_level_folds(df)


def test_leaky_split_raises_through_cross_validate():
    df = synthetic_table()
    df.loc[df.video_id == "00_0", "fold"] = 2
    with pytest.raises(SubjectLeakageError):
        detector.cross_validate(df, "3class", "logreg", seed=0)


def test_clean_split_passes_and_four_video_subjects_are_fine():
    df = synthetic_table()
    extra = df[df.video_id == "01_2"].assign(video_id="01_2_2")  # split drowsy recording
    detector.assert_subject_level_folds(pd.concat([df, extra]))


def test_missing_fold_is_refused():
    df = synthetic_table()
    df.loc[0, "fold"] = np.nan
    with pytest.raises(SubjectLeakageError):
        detector.assert_subject_level_folds(df)


def test_synthetic_in_test_fold_is_refused():
    df = synthetic_table().assign(is_synthetic=False)
    df.loc[0, "is_synthetic"] = True
    with pytest.raises(ValueError, match="real data only"):
        detector.cross_validate(df, "3class", "logreg", seed=0)


# ------------------------------------------------------- feature selection
def test_feature_selection_excludes_every_listed_column():
    X = detector.build_features(synthetic_table())
    required = {
        "pitch_range", "pitch_centered_range", "rotation_deg", "stored_width",
        "stored_height", "effective_width", "effective_height", "bitrate_kbps",
        "detection_rate", "low_quality", "low_resolution", "ear_open_baseline",
        "ear_closed_baseline", "baseline_source", "duration_s", "fps", "n_frames",
    }
    assert required <= set(detector.EXCLUDED)
    assert not required & set(X.columns)
    assert list(X.columns) == list(detector.FEATURES)


def test_raw_counts_are_per_minute():
    df = synthetic_table()
    X = detector.build_features(df)
    for c in detector.COUNT_FEATURES:
        np.testing.assert_allclose(X[f"{c}_per_min"], df[c] / df["duration_s"] * 60)
        assert c not in X.columns
    assert "n_blinks" not in X.columns and "blink_rate_per_min" in X.columns


def test_nan_survives_feature_building():
    df = synthetic_table()
    df.loc[0, "mean_closing_speed"] = np.nan
    assert np.isnan(detector.build_features(df).loc[0, "mean_closing_speed"])


# ------------------------------------------------------------------ metrics
def test_metrics_match_hand_computed_and_sklearn():
    labels = ["a", "b", "c"]
    t = ["a", "a", "b", "b", "c", "c"]
    p = ["a", "b", "b", "b", "c", "a"]
    m = detector.compute_metrics(t, p, labels)
    # a: tp1 fp1 fn1 -> F1 .5; b: tp2 fp1 fn0 -> P 2/3 R 1 -> .8; c: tp1 fp0 fn1 -> 2/3
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["f1_a"] == pytest.approx(0.5)
    assert m["f1_b"] == pytest.approx(0.8)
    assert m["f1_c"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m["confusion"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["accuracy"] == pytest.approx(accuracy_score(t, p))
    assert m["macro_f1"] == pytest.approx(f1_score(t, p, average="macro"))
    assert m["confusion"] == confusion_matrix(t, p, labels=labels).tolist()


def test_bootstrap_resamples_subjects_not_videos():
    # Subject A is always right, subject B always wrong. Any resample of subjects
    # gives accuracy 0, .5 or 1, never anything in between; a video-level
    # bootstrap over six videos would produce values like 1/3.
    pred = pd.DataFrame({
        "subject_id": ["A"] * 3 + ["B"] * 3,
        "class": ["alert", "drowsy", "alert"] * 2,
        "pred": ["alert", "drowsy", "alert", "drowsy", "alert", "drowsy"],
    })
    rng_acc = []
    for seed in range(20):
        ci = detector.bootstrap_ci(pred, ["alert", "drowsy"], n_boot=50, seed=seed)
        rng_acc.extend(ci["accuracy_ci"])
    assert set(np.round(rng_acc, 6)) <= {0.0, 0.5, 1.0}


def test_cross_validate_gives_one_prediction_per_video():
    df = synthetic_table()
    for task in detector.CLASSES:
        for model in detector.MODELS:
            pred, latency = detector.cross_validate(df, task, model, seed=0)
            assert len(pred) == len(detector.select_task(df, task))
            assert pred.video_id.is_unique and latency >= 0
    binary, _ = detector.cross_validate(df, "binary", "hgb", 0)
    assert "low_vigilant" not in set(binary["class"]) | set(binary["pred"])


# ------------------------------------------------------------ subgroup hook
def _annotations(tmp_path, ages):
    path = tmp_path / "ann.csv"
    pd.DataFrame({"subject_id": [f"{s:02d}" for s in range(len(ages))],
                  "age_bracket": ages,
                  "gender_presentation": ["male_presenting", "female_presenting"] * (len(ages) // 2)}
                 ).to_csv(path, index=False)
    return path


def test_subgroup_hook_not_yet_annotated(tmp_path):
    pred, _ = detector.cross_validate(synthetic_table(), "3class", "logreg", 0)
    rep = detector.subgroup_report(pred, _annotations(tmp_path, [""] * 10), detector.CLASSES["3class"])
    assert not rep.ready and "not yet annotated" in rep.message


def test_subgroup_hook_refuses_partial_annotation(tmp_path):
    pred, _ = detector.cross_validate(synthetic_table(), "3class", "logreg", 0)
    path = _annotations(tmp_path, ["18-25", "26-35"] + [""] * 8)
    rep = detector.subgroup_report(pred, path, detector.CLASSES["3class"])
    assert not rep.ready and "2 of 10" in rep.message


def test_subgroup_hook_reports_gap_when_annotated(tmp_path):
    pred, _ = detector.cross_validate(synthetic_table(), "3class", "logreg", 0)
    path = _annotations(tmp_path, ["18-25"] * 5 + ["26-35"] * 5)
    rep = detector.subgroup_report(pred, path, detector.CLASSES["3class"])
    assert rep.ready
    ages = rep.table[rep.table.attribute == "age_bracket"]
    assert set(ages.group) == {"18-25", "26-35"}
    assert rep.gaps["age_bracket"]["gap"] == pytest.approx(ages.macro_f1.max() - ages.macro_f1.min())


# ------------------------------------------------------------------ logging
def test_run_appends_and_never_overwrites(tmp_path):
    feats = tmp_path / "f.csv"
    synthetic_table().to_csv(feats, index=False)
    kw = dict(annotations_path=tmp_path / "none.csv", results_dir=tmp_path,
              seeds=(0, 1), n_boot=20)
    detector.run(feats, **kw)
    first = pd.read_csv(tmp_path / "detector_runs.csv")
    detector.run(feats, **kw)
    both = pd.read_csv(tmp_path / "detector_runs.csv")
    assert len(both) == 2 * len(first) == 2 * 2 * 2 * 2  # runs x tasks x models x seeds
    assert both.run_id.is_unique
    assert (both.mix_ratio == 0.0).all()
