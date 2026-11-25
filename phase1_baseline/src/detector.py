"""Phase 1.3 baseline detector on per-video features.

    python -m src.detector                       # from phase1_baseline/

Reads data/video_features.csv (one row per video), trains a gradient-boosted tree
model and a logistic-regression reference with leave-one-fold-out over the five
official UTA-RLDD folds, and reports pooled out-of-fold metrics with subject-level
bootstrap confidence intervals. Two tasks:

  * 3class: alert / low_vigilant / drowsy. Matches the task of the published
    UTA-RLDD baseline (65.2% video accuracy, Ghoddoosian et al., CVPRW 2019).
  * binary: alert vs drowsy, low_vigilant dropped.

Every (task, model, seed) is appended to results/detector_runs.csv and its
out-of-fold predictions to results/detector_predictions.csv. Nothing is overwritten.

Hyperparameters are fixed in advance, never tuned: with 60 subjects there is no
spare data for a validation split, and tuning on the held-out folds would leak.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data" / "video_features.csv"
DEFAULT_ANNOTATIONS = REPO / "data" / "subgroup_annotations_prefilled.csv"
DEFAULT_RESULTS = REPO / "results"

PUBLISHED_BASELINE_ACC = 0.652   # Ghoddoosian et al., CVPRW 2019, 3-class video accuracy
HUMAN_OBSERVER_ACC = 0.578       # same paper, human observers

CLASSES = {
    "3class": ("alert", "low_vigilant", "drowsy"),
    "binary": ("alert", "drowsy"),
}

# ------------------------------------------------------------------ features
# Used as they are: already rates, fractions, means or per-video spreads.
RATE_FEATURES = (
    "perclos80_mean", "perclos80_max", "perclos70_mean", "mean_openness",
    "blink_rate_per_min", "mean_blink_duration_s", "mean_reopening_speed",
    "mean_closing_speed", "mar_mean", "mar_p95", "yawn_frame_fraction",
    "pitch_std", "pitch_centered_std", "nose_y_drift", "face_scale_drift",
)
# Raw counts or totals, which scale with video length (split drowsy parts run as
# short as 3 minutes, full recordings up to 19). Divided by duration_s and
# expressed per minute. n_blinks is not listed: its per-minute form is exactly
# blink_rate_per_min, already in RATE_FEATURES, so normalising it would add a
# duplicate column.
COUNT_FEATURES = ("n_long_closures", "n_microsleeps", "total_closure_s")
NORMALISED_BY_RATE_COLUMN = {"n_blinks": "blink_rate_per_min"}

FEATURES = RATE_FEATURES + tuple(f"{c}_per_min" for c in COUNT_FEATURES)

# Never model inputs. The selection below is a whitelist, so anything not listed
# above is already out; this list exists so the reasons are written down and the
# test can check each one by name.
EXCLUDED = {
    # Dominated by solvePnP outliers: the cohort mean is ~119 degrees, which a
    # seated person cannot nod through. These measure tracking failure, not nodding.
    "pitch_range": "solvePnP outliers, not nodding",
    "pitch_centered_range": "solvePnP outliers, not nodding",
    # Camera properties, not drowsiness. Kept in the table for covariate analysis.
    "rotation_deg": "camera property",
    "stored_width": "camera property",
    "stored_height": "camera property",
    "effective_width": "camera property",
    "effective_height": "camera property",
    "bitrate_kbps": "camera property",
    # Tracking quality. detection_rate is itself a fairness metric; a model that
    # consumes it would fold tracking failure into its drowsiness score.
    "detection_rate": "tracking quality, a fairness metric in its own right",
    "low_quality": "tracking quality",
    "low_resolution": "tracking quality",
    # Calibration and bookkeeping. duration_s is used only as the denominator for
    # COUNT_FEATURES, never as an input: video length should not predict class.
    "ear_open_baseline": "per-subject calibration constant",
    "ear_closed_baseline": "per-subject calibration constant",
    "baseline_source": "bookkeeping",
    "duration_s": "bookkeeping, denominator only",
    "fps": "bookkeeping",
    "n_frames": "bookkeeping",
}
_overlap = set(EXCLUDED) & (set(FEATURES) | set(COUNT_FEATURES))
if _overlap:  # pragma: no cover - guards edits to the lists above
    raise RuntimeError(f"columns both used and excluded: {sorted(_overlap)}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs only, in FEATURES order. NaN stays NaN."""
    missing = [c for c in RATE_FEATURES + COUNT_FEATURES + ("duration_s",) if c not in df]
    if missing:
        raise KeyError(f"video features table is missing {missing}")
    duration = df["duration_s"].astype(float)
    if (duration <= 0).any() or duration.isna().any():
        bad = df.loc[(duration <= 0) | duration.isna(), "video_id"].tolist()
        raise ValueError(f"non-positive or missing duration_s for {bad}")
    X = df.loc[:, list(RATE_FEATURES)].astype(float).copy()
    for c in COUNT_FEATURES:
        X[f"{c}_per_min"] = df[c].astype(float) / duration * 60.0
    return X.loc[:, list(FEATURES)]


# ------------------------------------------------------------------ splits
class SubjectLeakageError(AssertionError):
    """A subject appears on both sides of a split. Results would be invalid."""


def assert_subject_level_folds(df: pd.DataFrame) -> None:
    """Every subject_id must belong to exactly one fold. Raises, never warns.

    Explicit raise rather than `assert`, which `python -O` strips.
    """
    if df["subject_id"].isna().any() or df["fold"].isna().any():
        raise SubjectLeakageError("rows with no subject_id or no fold; cannot verify the split")
    folds_per_subject = df.groupby("subject_id")["fold"].nunique()
    leaky = folds_per_subject[folds_per_subject > 1]
    if len(leaky):
        detail = {s: sorted(df.loc[df.subject_id == s, "fold"].unique().tolist())
                  for s in leaky.index}
        raise SubjectLeakageError(
            f"{len(leaky)} subject(s) appear in more than one fold: {detail}. "
            "Identity leakage inflates accuracy; refusing to evaluate.")


def _check_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared = set(train.subject_id) & set(test.subject_id)
    if shared:
        raise SubjectLeakageError(f"subjects in both train and test: {sorted(shared)}")
    if "is_synthetic" in test and test["is_synthetic"].astype(bool).any():
        raise ValueError("synthetic samples in a test fold; test on real data only")


# ------------------------------------------------------------------ models
def make_model(name: str, seed: int):
    if name == "hgb":
        # max_features < 1 subsamples columns per split. It regularises on 145
        # training rows and is the only thing that makes the seed matter: with
        # early stopping off (auto, n < 10000) HGB is otherwise deterministic.
        return HistGradientBoostingClassifier(max_features=0.8, random_state=seed)
    if name == "logreg":
        # lbfgs on a multiclass target is multinomial. It is deterministic, so
        # the five seeds give identical results; they are run anyway for a
        # uniform log. Imputation and scaling are fitted inside each fold.
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=seed),
        )
    raise ValueError(f"unknown model {name!r}")


MODELS = ("hgb", "logreg")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:  # torch is not needed here; seeded when present
        pass


def select_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    keep = CLASSES[task]
    unknown = set(df["class"]) - set(CLASSES["3class"])
    if unknown:
        raise ValueError(f"unexpected class labels {sorted(unknown)}")
    return df[df["class"].isin(keep)].reset_index(drop=True)


def cross_validate(df: pd.DataFrame, task: str, model_name: str, seed: int) -> tuple[pd.DataFrame, float]:
    """Leave-one-fold-out. Returns one out-of-fold prediction per video, and
    classifier latency in ms per video (features to label, extraction excluded)."""
    assert_subject_level_folds(df)
    data = select_task(df, task)
    labels = CLASSES[task]
    y = data["class"].map({c: i for i, c in enumerate(labels)}).to_numpy()
    X = build_features(data)
    pred = np.full(len(data), -1)
    elapsed, n_pred = 0.0, 0
    for fold in sorted(data["fold"].unique()):
        test = (data["fold"] == fold).to_numpy()
        _check_split(data[~test], data[test])
        set_seeds(seed)
        model = make_model(model_name, seed).fit(X[~test], y[~test])
        t0 = time.perf_counter()
        pred[test] = model.predict(X[test])
        elapsed += time.perf_counter() - t0
        n_pred += int(test.sum())
    if (pred < 0).any():
        raise RuntimeError("some videos received no out-of-fold prediction")
    out = data[["video_id", "subject_id", "fold", "class"]].copy()
    out["pred"] = [labels[i] for i in pred]
    return out, 1000.0 * elapsed / n_pred


# ------------------------------------------------------------------ metrics
def compute_metrics(y_true, y_pred, labels) -> dict:
    labels = list(labels)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        **{f"f1_{c}": float(v) for c, v in zip(labels, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def bootstrap_ci(pred: pd.DataFrame, labels, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """Percentile CIs for accuracy and macro F1, resampling SUBJECTS with replacement.

    A subject's videos (and, when `pred` stacks several seeds, all of that
    subject's predictions) travel together, because they are not independent.
    """
    rng = np.random.default_rng(seed)
    labels = list(labels)
    groups = {s: g for s, g in pred.groupby("subject_id")}
    subjects = np.array(sorted(groups))
    true_by = {s: g["class"].to_numpy() for s, g in groups.items()}
    pred_by = {s: g["pred"].to_numpy() for s, g in groups.items()}
    acc, f1 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        t = np.concatenate([true_by[s] for s in draw])
        p = np.concatenate([pred_by[s] for s in draw])
        acc[b] = np.mean(t == p)
        f1[b] = f1_score(t, p, labels=labels, average="macro", zero_division=0)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"accuracy_ci": tuple(np.percentile(acc, [lo, hi])),
            "macro_f1_ci": tuple(np.percentile(f1, [lo, hi]))}


# ------------------------------------------------------------------ subgroups
SUBGROUP_ATTRIBUTES = ("skin_tone_ita_category", "gender_presentation", "age_bracket", "glasses")
MIN_SUBJECTS_FOR_GAP = 3


@dataclass
class SubgroupReport:
    ready: bool
    message: str
    metadata_version: str
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    gaps: dict = field(default_factory=dict)


def _file_version(path: Path) -> str:
    return "sha1:" + hashlib.sha1(path.read_bytes()).hexdigest()[:12]


# ============================================================================
# PER-SUBGROUP HOOK. Gated on the human annotation pass: returns ready=False
# with a "not yet annotated" message until age_bracket is filled for every
# subject. Labels are read from the annotation file, never re-derived here.
# ============================================================================
def subgroup_report(pred: pd.DataFrame, annotations_path: Path, labels) -> SubgroupReport:
    """Per-group macro F1 and the max-to-min gap, for each subgroup attribute.

    `pred` is out-of-fold predictions (video_id, subject_id, class, pred), and
    may stack several seeds. Groups with fewer than MIN_SUBJECTS_FOR_GAP
    subjects are reported but left out of the gap, which they would dominate.
    """
    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        return SubgroupReport(False, f"not yet annotated: {annotations_path} does not exist", "none")
    version = _file_version(annotations_path)
    ann = pd.read_csv(annotations_path, dtype=str)
    ann["subject_id"] = ann["subject_id"].str.strip()
    age = ann.get("age_bracket", pd.Series(dtype=str)).fillna("").str.strip()
    n_filled, n_subjects = int((age != "").sum()), int(pred["subject_id"].nunique())
    if n_filled == 0:
        return SubgroupReport(False, "not yet annotated: the age_bracket column is empty", version)
    unlabelled = sorted(set(pred["subject_id"]) - set(ann.loc[age != "", "subject_id"]))
    if unlabelled:
        return SubgroupReport(
            False, f"not yet annotated: age_bracket is filled for {n_filled} of {n_subjects} "
                   f"subjects; per-subgroup F1 is not computed on a partial annotation", version)

    labels = list(labels)
    rows, gaps = [], {}
    for attr in SUBGROUP_ATTRIBUTES:
        if attr not in ann:
            continue
        m = pred.merge(ann[["subject_id", attr]], on="subject_id", how="left")
        m[attr] = m[attr].fillna("unlabelled")
        for value, g in m.groupby(attr):
            rows.append({
                "attribute": attr, "group": value,
                "n_subjects": g["subject_id"].nunique(), "n_videos": g["video_id"].nunique(),
                "macro_f1": f1_score(g["class"], g["pred"], labels=labels,
                                     average="macro", zero_division=0),
                "accuracy": float(np.mean(g["class"] == g["pred"])),
            })
        sub = pd.DataFrame([r for r in rows if r["attribute"] == attr])
        eligible = sub[(sub.n_subjects >= MIN_SUBJECTS_FOR_GAP)
                       & (~sub.group.isin(["unlabelled", "unsure"]))]
        if len(eligible) >= 2:
            best, worst = eligible.loc[eligible.macro_f1.idxmax()], eligible.loc[eligible.macro_f1.idxmin()]
            gaps[attr] = {"gap": float(best.macro_f1 - worst.macro_f1),
                          "best": best.group, "worst": worst.group}
    return SubgroupReport(True, "ok", version, pd.DataFrame(rows), gaps)


# ------------------------------------------------------------------ logging
RUN_COLUMNS = [
    "run_id", "batch_id", "timestamp", "git_commit", "task", "model", "seed", "mix_ratio",
    "n_videos", "n_subjects", "features", "subgroup_metadata_version",
    "accuracy", "accuracy_ci_lo", "accuracy_ci_hi", "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
    "f1_alert", "f1_low_vigilant", "f1_drowsy", "confusion_matrix", "confusion_labels",
    "per_subgroup_f1", "max_min_subgroup_gap", "latency_ms_per_video", "n_boot",
]


def append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows. Refuses to write into a file whose header has drifted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=columns)
    if path.exists() and path.stat().st_size > 0:
        existing = list(pd.read_csv(path, nrows=0).columns)
        if existing != columns:
            raise ValueError(f"{path} has a different header; refusing to append. "
                             "Move it aside rather than overwrite it.")
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ------------------------------------------------------------------ main
def load_features(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run the extraction and aggregation passes first; "
            "no results are produced without real data.")
    return pd.read_csv(path, dtype={"subject_id": str, "video_id": str})


def _fmt(mean, std, ci=None) -> str:
    s = f"{100 * mean:5.1f} ± {100 * std:3.1f}"
    return s + (f"  [{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]" if ci is not None else "")


def run(features_path=DEFAULT_FEATURES, annotations_path=DEFAULT_ANNOTATIONS,
        results_dir=DEFAULT_RESULTS, seeds=(0, 1, 2, 3, 4), n_boot=2000,
        tasks=("3class", "binary"), models=MODELS, log=True) -> pd.DataFrame:
    df = load_features(features_path)
    assert_subject_level_folds(df)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    commit = _git_commit()
    results_dir = Path(results_dir)
    summary, run_rows = [], []

    for task in tasks:
        labels = CLASSES[task]
        for model_name in models:
            per_seed, preds, latencies = [], [], []
            sub_version = "none"
            for seed in seeds:
                pred, latency = cross_validate(df, task, model_name, seed)
                m = compute_metrics(pred["class"], pred["pred"], labels)
                ci = bootstrap_ci(pred, labels, n_boot=n_boot, seed=seed)
                sub = subgroup_report(pred, annotations_path, labels)
                sub_version = sub.metadata_version
                run_id = f"{batch_id}-{task}-{model_name}-s{seed}"
                run_rows.append({
                    "run_id": run_id, "batch_id": batch_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": commit, "task": task, "model": model_name, "seed": seed,
                    "mix_ratio": 0.0, "n_videos": len(pred), "n_subjects": pred.subject_id.nunique(),
                    "features": ";".join(FEATURES), "subgroup_metadata_version": sub.metadata_version,
                    "accuracy": m["accuracy"], "accuracy_ci_lo": ci["accuracy_ci"][0],
                    "accuracy_ci_hi": ci["accuracy_ci"][1], "macro_f1": m["macro_f1"],
                    "macro_f1_ci_lo": ci["macro_f1_ci"][0], "macro_f1_ci_hi": ci["macro_f1_ci"][1],
                    **{f"f1_{c}": m[f"f1_{c}"] for c in labels},
                    "confusion_matrix": json.dumps(m["confusion"]),
                    "confusion_labels": ";".join(labels),
                    "per_subgroup_f1": (sub.table.to_json(orient="records") if sub.ready
                                        else sub.message),
                    "max_min_subgroup_gap": json.dumps(sub.gaps) if sub.ready else "",
                    "latency_ms_per_video": latency, "n_boot": n_boot,
                })
                per_seed.append(m)
                latencies.append(latency)
                preds.append(pred.assign(run_id=run_id, seed=seed, task=task, model=model_name))

            stacked = pd.concat(preds, ignore_index=True)
            pooled_ci = bootstrap_ci(stacked, labels, n_boot=n_boot, seed=0)
            row = {"task": task, "model": model_name, "n_videos": len(preds[0]),
                   "seeds": len(seeds), "subgroup": subgroup_report(stacked, annotations_path, labels),
                   "confusion_seed0": per_seed[0]["confusion"], "labels": labels,
                   "latency_ms": float(np.mean(latencies)), "pred": stacked}
            for k in ["accuracy", "macro_f1"] + [f"f1_{c}" for c in labels]:
                vals = np.array([m[k] for m in per_seed])
                row[k] = (vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0)
            row["accuracy_ci"], row["macro_f1_ci"] = pooled_ci["accuracy_ci"], pooled_ci["macro_f1_ci"]
            summary.append(row)

    if log:
        append_csv(results_dir / "detector_runs.csv", run_rows, RUN_COLUMNS)
        append_csv(results_dir / "detector_predictions.csv",
                   [r for s in summary for r in s["pred"].to_dict("records")],
                   ["run_id", "task", "model", "seed", "video_id", "subject_id", "fold", "class", "pred"])
    _print_summary(summary, batch_id, n_boot, results_dir if log else None)
    return pd.DataFrame(run_rows)


def _print_summary(summary, batch_id, n_boot, results_dir) -> None:
    print(f"\nPhase 1.3 baseline detector, batch {batch_id}")
    print("Leave-one-fold-out over the 5 official UTA-RLDD folds, subject-disjoint (asserted).")
    print(f"Mean ± sd over seeds; [95% CI] from {n_boot} subject-level bootstrap resamples "
          "of the pooled out-of-fold predictions.\n")
    for task in CLASSES:
        rows = [s for s in summary if s["task"] == task]
        if not rows:
            continue
        labels = rows[0]["labels"]
        print(f"== {task}: {' / '.join(labels)}  ({rows[0]['n_videos']} videos)")
        head = f"{'model':8} {'accuracy %':>27} {'macro F1 %':>27} " + " ".join(
            f"{'F1 ' + c:>15}" for c in labels) + f" {'ms/video':>9}"
        print(head)
        for s in rows:
            print(f"{s['model']:8} {_fmt(*s['accuracy'], s['accuracy_ci']):>27} "
                  f"{_fmt(*s['macro_f1'], s['macro_f1_ci']):>27} "
                  + " ".join(f"{_fmt(*s['f1_' + c]):>15}" for c in labels)
                  + f" {s['latency_ms']:9.4f}")
        if task == "3class":
            print(f"{'ref':8} {'published baseline (HM-LSTM)':>27}  {100 * PUBLISHED_BASELINE_ACC:.1f}% accuracy")
            print(f"{'ref':8} {'human observers':>27}  {100 * HUMAN_OBSERVER_ACC:.1f}% accuracy")
        for s in rows:
            cm = pd.DataFrame(s["confusion_seed0"], index=[f"true {c}" for c in labels],
                              columns=[f"pred {c}" for c in labels])
            print(f"\n  {s['model']} confusion matrix, seed 0 (rows true, columns predicted):")
            print("  " + cm.to_string().replace("\n", "\n  "))
        print()
    sub = summary[0]["subgroup"]
    print(f"Per-subgroup breakdown: {sub.message if not sub.ready else 'see detector_runs.csv'}")
    if sub.ready:
        for s in summary:
            print(f"\n{s['task']} {s['model']}:\n{s['subgroup'].table.to_string(index=False)}")
            print(f"max-min macro-F1 gaps: {s['subgroup'].gaps}")
    if results_dir is not None:
        print(f"\nAppended to {results_dir / 'detector_runs.csv'} and detector_predictions.csv")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--no-log", action="store_true", help="print only, append nothing")
    a = p.parse_args(argv)
    try:
        run(a.features, a.annotations, a.results_dir, tuple(a.seeds), a.n_boot, log=not a.no_log)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
