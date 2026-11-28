"""Phase 1.3 subgroup audit: accuracy and tracking fairness per demographic group.

Reads the out-of-fold majority-vote predictions already logged by the window-level
detector (no retraining), merges subgroup labels by subject_id, collapses bands by
rules fixed in advance (see COLLAPSE and the 2026-11-28 decision row in
PROJECT_TRACKER.md), and reports per group:

  * macro F1 and accuracy over the pooled seeds, with 95% CIs from resampling SUBJECTS
    within the group (a subject's videos and seeds travel together);
  * group size in subjects and videos;
  * mean per-video detection_rate and the fraction of windows dropped for low coverage,
    tracking-fairness metrics kept separate from accuracy;
  * per axis, the max-to-min gap over groups with >= MIN_SUBJECTS_FOR_GAP subjects,
    excluding "unknown"/"unsure", with a bootstrap CI.

    python -m src.audit                        # 60 s, 3-class HGB, the chosen config
    python -m src.audit --no-log               # print only
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .detector import (CLASSES, DEFAULT_ANNOTATIONS, DEFAULT_RESULTS, MIN_SUBJECTS_FOR_GAP, REPO,
                       _file_version, _git_commit, append_csv)

DEFAULT_VOTES = DEFAULT_RESULTS / "detector_window_votes.csv"
DEFAULT_WINDOWS = REPO / "data" / "window_features_w60_s30.csv"
DEFAULT_VIDEO_FEATURES = REPO / "data" / "video_features.csv"
DEFAULT_BATCH = "20261125T065741Z-04b301"  # the window sweep behind the 60 s decision

UNKNOWN = "unknown"
UNSURE = "unsure"
NOT_IN_GAP = {UNKNOWN, UNSURE}

# Fixed 2026-11-28, before the audit was first run. Changing a rule after seeing results
# needs a new decision row that says so.
COLLAPSE = {
    "skin_tone": ("skin_tone_ita_category", {
        "Very light": "Light", "Light": "Light",
        "Intermediate": "Medium", "Tan": "Medium",
        "Brown": "Dark", "Dark": "Dark",
    }),
    "age": ("age_bracket", {
        "18-25": "18-25", "26-35": "26+", "36-50": "26+", "51+": "26+", UNSURE: UNSURE,
    }),
    "gender_presentation": ("gender_presentation", {
        "male_presenting": "male_presenting", "female_presenting": "female_presenting",
        UNSURE: UNSURE,
    }),
    "glasses": ("glasses", {"glasses": "glasses", "none": "none", UNSURE: UNSURE}),
}

AUDIT_COLUMNS = [
    "run_id", "audit_batch_id", "timestamp", "git_commit", "source_batch_id", "task", "model",
    "window_s", "seeds", "subgroup_metadata_version", "axis", "group", "in_gap",
    "n_subjects", "n_videos", "n_predictions",
    "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
    "accuracy", "accuracy_ci_lo", "accuracy_ci_hi",
    "mean_detection_rate", "detection_rate_ci_lo", "detection_rate_ci_hi",
    "windows_dropped", "windows_total", "frac_windows_dropped",
    "frac_dropped_ci_lo", "frac_dropped_ci_hi",
    "gap_macro_f1", "gap_macro_f1_ci_lo", "gap_macro_f1_ci_hi", "gap_best", "gap_worst",
    "gap_accuracy", "gap_accuracy_ci_lo", "gap_accuracy_ci_hi",
    "n_boot", "notes",
]


class AuditInputError(ValueError):
    pass


# ------------------------------------------------------------------ inputs
def collapse_labels(ann: pd.DataFrame) -> pd.DataFrame:
    """One row per subject with one collapsed column per axis.

    Missing skin tone becomes "unknown"; any value outside the fixed rules raises
    instead of being dropped or guessed.
    """
    out = pd.DataFrame({"subject_id": ann["subject_id"].astype(str).str.strip()})
    for axis, (col, rule) in COLLAPSE.items():
        raw = ann[col].astype("string").str.strip()
        mapped = raw.map(rule)
        missing = raw.isna() | (raw == "")
        if axis == "skin_tone":
            mapped = mapped.mask(missing, UNKNOWN)
        elif missing.any():
            raise AuditInputError(f"{col} is blank for subjects "
                                  f"{sorted(out.subject_id[missing])}; finish the annotation")
        bad = mapped.isna() & ~missing
        if bad.any():
            raise AuditInputError(f"{col} has values outside the fixed collapse rules: "
                                  f"{sorted(set(raw[bad]))}")
        out[axis] = mapped.astype(str)
    if out.subject_id.duplicated().any():
        raise AuditInputError("duplicate subject_id in the annotation file")
    return out


def load_votes(path: Path, batch_id: str, window_s: float, task: str, model: str) -> pd.DataFrame:
    v = pd.read_csv(path, dtype={"subject_id": str, "video_id": str})
    v = v[v.run_id.str.startswith(batch_id) & (v.window_s == window_s)
          & (v.task == task) & (v.model == model)]
    if v.empty:
        raise AuditInputError(f"no votes for batch {batch_id}, {window_s:g}s, {task}, {model}")
    per_seed = v.groupby("seed").size()
    if per_seed.nunique() != 1:
        raise AuditInputError(f"seeds have different video counts: {per_seed.to_dict()}")
    return v


def tracking_by_subject(windows_path: Path, video_features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-video detection_rate, and per-subject kept/dropped window counts."""
    meta_path = windows_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    win = pd.read_csv(windows_path, dtype={"subject_id": str}, usecols=["subject_id"])
    kept = win.groupby("subject_id").size().rename("kept")
    dropped = pd.Series({s: d["dropped"] for s, d in meta["dropped_by_subject"].items()},
                        name="dropped", dtype=float)
    for s, d in meta["dropped_by_subject"].items():
        if kept.get(s, 0) != d["kept"]:
            raise AuditInputError(f"{meta_path.name} and {windows_path.name} disagree on "
                                  f"subject {s}'s kept windows")
    subj = pd.concat([kept, dropped], axis=1).fillna(0).astype(int)
    if int(subj.dropped.sum()) != meta["windows_dropped_low_coverage"]:
        raise AuditInputError("per-subject dropped counts do not sum to the meta total")
    subj.index.name = "subject_id"
    vf = pd.read_csv(video_features_path, dtype={"subject_id": str, "video_id": str},
                     usecols=["video_id", "subject_id", "detection_rate"])
    return vf, subj.reset_index()


# ------------------------------------------------------------------ metrics
def _f1_acc(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Macro F1 and accuracy from confusion matrices, shape (..., k, k)."""
    tp = np.einsum("...ii->...i", cm)
    fp, fn = cm.sum(axis=-2) - tp, cm.sum(axis=-1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return f1.mean(axis=-1), tp.sum(axis=-1) / cm.sum(axis=(-2, -1))


def audit(votes: pd.DataFrame, groups: pd.DataFrame, videos: pd.DataFrame,
          windows: pd.DataFrame, labels, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per (axis, group), plus the axis gap repeated on every row of that axis.

    Every bootstrap draw resamples subjects with replacement inside each group, so group
    sizes stay fixed and one subject's videos, seeds and windows move together. The gap
    CI uses the same per-group draws, so it reflects the joint uncertainty.
    """
    labels = list(labels)
    unknown_subjects = set(votes.subject_id) - set(groups.subject_id)
    if unknown_subjects:
        raise AuditInputError(f"subjects with predictions but no annotation row: "
                              f"{sorted(unknown_subjects)}")
    rng = np.random.default_rng(seed)
    lo_q, hi_q = 2.5, 97.5

    # Per-subject sufficient statistics, so each draw is a sum.
    cms = {s: confusion_matrix(g["class"], g["pred"], labels=labels)
           for s, g in votes.groupby("subject_id")}
    det = videos.groupby("subject_id")["detection_rate"].agg(["sum", "count"])
    win = windows.set_index("subject_id")

    rows = []
    for axis in COLLAPSE:
        boot_f1, boot_acc, axis_rows = {}, {}, []
        for group, gsub in groups.groupby(axis):
            subj = sorted(set(gsub.subject_id) & set(cms))
            n = len(subj)
            cm = np.stack([cms[s] for s in subj])
            d_sum = det.reindex(subj)["sum"].to_numpy()
            d_cnt = det.reindex(subj)["count"].to_numpy()
            w_drop = win.reindex(subj)["dropped"].fillna(0).to_numpy()
            w_tot = w_drop + win.reindex(subj)["kept"].fillna(0).to_numpy()

            draws = rng.integers(0, n, size=(n_boot, n))
            counts = np.stack([np.bincount(d, minlength=n) for d in draws])
            f1_b, acc_b = _f1_acc(np.einsum("bs,sij->bij", counts, cm).astype(float))
            det_b = (counts @ d_sum) / (counts @ d_cnt)
            drop_b = (counts @ w_drop) / (counts @ w_tot)
            f1, acc = _f1_acc(cm.sum(axis=0).astype(float))
            boot_f1[group], boot_acc[group] = f1_b, acc_b

            g_votes = votes[votes.subject_id.isin(subj)]
            in_gap = n >= MIN_SUBJECTS_FOR_GAP and group not in NOT_IN_GAP
            note = "" if in_gap else (
                f"listed, not in gap: {group}" if group in NOT_IN_GAP
                else f"listed, not in gap: fewer than {MIN_SUBJECTS_FOR_GAP} subjects")
            axis_rows.append({
                "axis": axis, "group": group, "in_gap": in_gap,
                "n_subjects": n, "n_videos": g_votes.video_id.nunique(),
                "n_predictions": len(g_votes),
                "macro_f1": float(f1), "macro_f1_ci_lo": np.percentile(f1_b, lo_q),
                "macro_f1_ci_hi": np.percentile(f1_b, hi_q),
                "accuracy": float(acc), "accuracy_ci_lo": np.percentile(acc_b, lo_q),
                "accuracy_ci_hi": np.percentile(acc_b, hi_q),
                "mean_detection_rate": d_sum.sum() / d_cnt.sum(),
                "detection_rate_ci_lo": np.percentile(det_b, lo_q),
                "detection_rate_ci_hi": np.percentile(det_b, hi_q),
                "windows_dropped": int(w_drop.sum()), "windows_total": int(w_tot.sum()),
                "frac_windows_dropped": w_drop.sum() / w_tot.sum(),
                "frac_dropped_ci_lo": np.percentile(drop_b, lo_q),
                "frac_dropped_ci_hi": np.percentile(drop_b, hi_q),
                "notes": note,
            })

        eligible = [r for r in axis_rows if r["in_gap"]]
        gap = {}
        if len(eligible) >= 2:
            names = [r["group"] for r in eligible]
            for metric, boot in (("macro_f1", boot_f1), ("accuracy", boot_acc)):
                point = {r["group"]: r[metric] for r in eligible}
                b = np.stack([boot[g] for g in names])
                gb = b.max(axis=0) - b.min(axis=0)
                gap[f"gap_{metric}"] = max(point.values()) - min(point.values())
                gap[f"gap_{metric}_ci_lo"] = np.percentile(gb, lo_q)
                gap[f"gap_{metric}_ci_hi"] = np.percentile(gb, hi_q)
                if metric == "macro_f1":
                    gap["gap_best"] = max(point, key=point.get)
                    gap["gap_worst"] = min(point, key=point.get)
        for r in axis_rows:
            r.update(gap)
        rows.extend(axis_rows)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def _ci(v, lo, hi, pct=True) -> str:
    k = 100 if pct else 1
    return f"{k * v:5.1f} [{k * lo:5.1f}, {k * hi:5.1f}]"


def print_tables(table: pd.DataFrame, header: str) -> None:
    print(header)
    for axis, t in table.groupby("axis", sort=False):
        print(f"\n== {axis}")
        print(f"{'group':<18} {'subj':>4} {'vids':>4}  {'macro F1 % [95% CI]':>21}  "
              f"{'accuracy % [95% CI]':>21}  {'detection % [95% CI]':>21}  "
              f"{'windows dropped % [CI]':>23}")
        for _, r in t.iterrows():
            mark = "" if r.in_gap else " *"
            print(f"{r.group + mark:<18} {r.n_subjects:4d} {r.n_videos:4d}  "
                  f"{_ci(r.macro_f1, r.macro_f1_ci_lo, r.macro_f1_ci_hi):>21}  "
                  f"{_ci(r.accuracy, r.accuracy_ci_lo, r.accuracy_ci_hi):>21}  "
                  f"{_ci(r.mean_detection_rate, r.detection_rate_ci_lo, r.detection_rate_ci_hi):>21}  "
                  f"{_ci(r.frac_windows_dropped, r.frac_dropped_ci_lo, r.frac_dropped_ci_hi):>16}"
                  f" {r.windows_dropped:>3d}/{r.windows_total:<4d}")
        r = t.iloc[0]
        if pd.notna(r.get("gap_macro_f1")):
            print(f"gap (max-min, eligible groups): macro F1 "
                  f"{_ci(r.gap_macro_f1, r.gap_macro_f1_ci_lo, r.gap_macro_f1_ci_hi)} points, "
                  f"best {r.gap_best}, worst {r.gap_worst}; accuracy "
                  f"{_ci(r.gap_accuracy, r.gap_accuracy_ci_lo, r.gap_accuracy_ci_hi)}")
        else:
            print("gap: fewer than two eligible groups")
    print(f"\n* listed but excluded from the gap (unknown/unsure, or < {MIN_SUBJECTS_FOR_GAP} "
          "subjects). A bootstrapped max-min gap is biased upward; read its CI as a range, "
          "not a test.")


def run(votes_path=DEFAULT_VOTES, annotations_path=DEFAULT_ANNOTATIONS,
        windows_path=DEFAULT_WINDOWS, video_features_path=DEFAULT_VIDEO_FEATURES,
        results_dir=DEFAULT_RESULTS, batch_id=DEFAULT_BATCH, window_s=60.0, task="3class",
        model="hgb", n_boot=2000, seed=0, log=True) -> pd.DataFrame:
    for p in (votes_path, annotations_path, windows_path, video_features_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} not found; no audit is produced without real data")
    labels = CLASSES[task]
    votes = load_votes(Path(votes_path), batch_id, window_s, task, model)
    ann = pd.read_csv(annotations_path, dtype=str)
    groups = collapse_labels(ann)
    videos, windows = tracking_by_subject(Path(windows_path), Path(video_features_path))
    videos = videos[videos.subject_id.isin(votes.subject_id)]
    table = audit(votes, groups, videos, windows, labels, n_boot=n_boot, seed=seed)

    audit_batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    seeds = sorted(int(s) for s in votes.seed.unique())
    table.insert(0, "run_id", [f"{audit_batch}-{a}-{g}" for a, g in zip(table.axis, table.group)])
    meta = {"audit_batch_id": audit_batch,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(), "source_batch_id": batch_id, "task": task,
            "model": model, "window_s": window_s, "seeds": ";".join(map(str, seeds)),
            "subgroup_metadata_version": _file_version(Path(annotations_path)), "n_boot": n_boot}
    table = table.assign(**meta)
    header = (f"Phase 1.3 subgroup audit {audit_batch}: {window_s:g}s windows, {task} {model}, "
              f"majority vote per video, source batch {batch_id}, seeds {seeds} pooled.\n"
              f"Annotations {meta['subgroup_metadata_version']}. 95% CIs from {n_boot} "
              "bootstrap resamples of subjects within each group.")
    print_tables(table, header)
    if log:
        out = Path(results_dir) / "subgroup_audit.csv"
        append_csv(out, table.to_dict("records"), AUDIT_COLUMNS)
        print(f"\nAppended {len(table)} rows to {out}")
    return table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--votes", type=Path, default=DEFAULT_VOTES)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    p.add_argument("--video-features", type=Path, default=DEFAULT_VIDEO_FEATURES)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--batch", default=DEFAULT_BATCH)
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--task", choices=list(CLASSES), default="3class")
    p.add_argument("--model", default="hgb")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args(argv)
    try:
        run(a.votes, a.annotations, a.windows, a.video_features, a.results_dir, a.batch,
            a.window_s, a.task, a.model, a.n_boot, log=not a.no_log)
    except (FileNotFoundError, AuditInputError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
