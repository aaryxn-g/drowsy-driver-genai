"""Pass 2: per-frame tables -> per-subject baselines -> per-video features.

The ordering matters and is the reason this is a separate pass.

An eye-openness threshold only means something relative to how open that
person's eye is when they are awake. Eye shape varies enormously between
people, so a single global EAR cut-off would label narrow-eyed subjects as
permanently drowsy. We therefore:

  1. estimate each subject's open/closed EAR from their ALERT video, then
  2. apply that one baseline to all three of their videos.

Taking the baseline from each video separately would be the obvious shortcut
and it is wrong: a drowsy recording may never contain a fully open eye, so a
within-video percentile would rescale the drowsiness away and flatten exactly
the signal we are trying to measure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import blinks, config, video


def load_frames(frames_dir: Path, video_id: str) -> pd.DataFrame | None:
    path = frames_dir / f"{video_id}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def ear_column(df: pd.DataFrame, use_corrected: bool) -> np.ndarray:
    col = "ear_corrected" if use_corrected else "ear"
    series = df[col].to_numpy(dtype=float)
    if use_corrected and not np.isfinite(series).any():
        return df["ear"].to_numpy(dtype=float)
    return series


def build_baselines(
    manifest: pd.DataFrame, frames_dir: Path, use_corrected: bool
) -> tuple[dict[str, blinks.EyeBaseline], list[str]]:
    """One eye baseline per subject, taken from their alert recording."""
    baselines: dict[str, blinks.EyeBaseline] = {}
    problems: list[str] = []

    for subject, group in manifest.groupby("subject_id"):
        alert = group[group["class"] == "alert"]
        source_rows = alert if len(alert) else group
        source_label = "alert" if len(alert) else "fallback:any-class"
        if not len(alert):
            problems.append(f"{subject}: no alert video, baseline from {source_label}")

        pooled: list[np.ndarray] = []
        for vid in source_rows["video_id"]:
            df = load_frames(frames_dir, vid)
            if df is not None:
                pooled.append(ear_column(df, use_corrected))
        if not pooled:
            problems.append(f"{subject}: no extracted frames, skipped")
            continue

        try:
            candidate = blinks.estimate_baseline(
                np.concatenate(pooled), source=source_label
            )
        except ValueError as exc:
            problems.append(f"{subject}: {exc}")
            continue

        if candidate.span < config.MIN_BASELINE_SPAN:
            problems.append(
                f"{subject}: eye baseline is degenerate "
                f"(open {candidate.ear_open:.4f}, closed {candidate.ear_closed:.4f}, "
                f"span {candidate.span:.4f}); subject skipped. Check that the "
                f"landmarker tracked this face and that the clip contains blinks."
            )
            continue
        baselines[subject] = candidate
    return baselines, problems


def _p95_minus_p5(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    return float(np.percentile(finite, 95) - np.percentile(finite, 5))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r over frames where both are finite. NaN if either does not vary."""
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def is_pitch_confounded(corr: float, nose_y_drift: float) -> bool:
    """Whether apparent nodding MAY be movement in the frame.

    This cannot tell a slump from a real head drop, since both move pitch and
    nose position together. False when either input is NaN: an unmeasurable video
    is not a flagged one.
    """
    return bool(abs(corr) > config.PITCH_CONFOUND_MIN_CORR
                and nose_y_drift > config.PITCH_CONFOUND_MIN_DRIFT)


def framing_features(df: pd.DataFrame) -> dict[str, float | bool]:
    """How much the subject moved in the frame, and whether pitch moved with them."""
    pitch = df["pitch"].to_numpy(dtype=float)
    centered = df["pitch_centered"].to_numpy(dtype=float)
    nose_y = df["nose_y_norm"].to_numpy(dtype=float)

    corr = pearson(pitch, nose_y)
    nose_y_drift = _p95_minus_p5(nose_y)
    has_centered = np.isfinite(centered).any()
    return {
        "nose_y_drift": nose_y_drift,
        "nose_x_drift": _p95_minus_p5(df["nose_x_norm"].to_numpy(dtype=float)),
        "face_scale_drift": _p95_minus_p5(df["face_scale"].to_numpy(dtype=float)),
        "pitch_nose_y_corr": corr,
        "pitch_centered_std": float(np.nanstd(centered)) if has_centered else float("nan"),
        "pitch_centered_range": (
            float(np.nanmax(centered) - np.nanmin(centered)) if has_centered else float("nan")
        ),
        "pitch_confounded": is_pitch_confounded(corr, nose_y_drift),
    }


def confounded_by_class(features: pd.DataFrame) -> pd.DataFrame:
    """Share of videos flagged pitch_confounded, per class, in class order."""
    present = [c for c in config.CLASS_ORDER if c in set(features["class"])]
    table = features.groupby("class")["pitch_confounded"].agg(["sum", "count", "mean"])
    return table.reindex(present).rename(
        columns={"sum": "confounded", "count": "videos", "mean": "fraction"})


def resolution_features(row: dict) -> dict:
    """Quality covariates copied from the manifest, plus the low_resolution flag.

    A flag and a covariate, never a filter: low-resolution subjects stay in. The flag
    is None rather than False when the manifest cannot say, so an unmeasured video
    cannot pass as a measured one.
    """
    columns = ["rotation_deg", "rotation_method", "stored_width", "stored_height",
               "effective_width", "effective_height", "bitrate_kbps"]
    out: dict = {c: row.get(c, "") for c in columns}
    width = pd.to_numeric(out["effective_width"], errors="coerce")
    height = pd.to_numeric(out["effective_height"], errors="coerce")
    known = bool(np.isfinite(width) and np.isfinite(height))
    out["low_resolution"] = video.is_low_resolution(width, height) if known else None
    return out


def video_fps(df: pd.DataFrame) -> float:
    if len(df) > 1 and df["t_s"].iloc[-1] > 0:
        return float(len(df)) / float(df["t_s"].iloc[-1])
    return 30.0


def summarise_frames(df: pd.DataFrame, baseline: blinks.EyeBaseline, fps: float,
                     use_corrected: bool) -> tuple[dict, np.ndarray]:
    """The feature set for any stretch of frames: a whole video or one window.

    Both paths go through here so a window row and a video row mean the same
    thing. `baseline` is always the subject's, from their alert video.
    """
    open_series = blinks.openness(ear_column(df, use_corrected), baseline)
    summary = blinks.summarise(
        open_series,
        df["mar"].to_numpy(dtype=float),
        df["pitch"].to_numpy(dtype=float),
        fps,
        baseline,
    )
    summary.update(framing_features(df))
    return summary, open_series


# ------------------------------------------------------------------ windows
# A window is a fixed-length slice of one video, summarised exactly as a whole
# video is. Each window is cut and summarised on its own, the way a short
# generated clip will be in Phase 2, so a blink straddling a boundary is
# truncated rather than borrowed from the neighbouring frames.
#
# With windows of 60 s or less, the rolling PERCLOS window (60 s) covers the
# whole slice, so perclos80_max equals perclos80_mean for that window.

def window_starts(duration_s: float, window_s: float, stride_s: float) -> np.ndarray:
    """Start times of the full windows that fit. A trailing partial window is not cut."""
    if window_s <= 0 or stride_s <= 0:
        raise ValueError("window_s and stride_s must be positive")
    n = int(np.floor((duration_s - window_s) / stride_s + 1e-9)) + 1
    return np.arange(max(n, 0)) * stride_s


def window_features(
    df: pd.DataFrame,
    baseline: blinks.EyeBaseline,
    window_s: float,
    stride_s: float,
    use_corrected: bool = True,
    min_coverage: float = config.MIN_WINDOW_COVERAGE,
) -> tuple[list[dict], int]:
    """Feature rows for one video's windows, and how many were dropped for coverage.

    Takes the subject's baseline as an argument and never estimates one: a short
    window may contain no fully open eye, and a per-window baseline would rescale
    the drowsiness away.
    """
    fps = video_fps(df)
    t = df["t_s"].to_numpy(dtype=float)
    # Last timestamp plus one frame, so a window that exactly fits is kept.
    # len(df) / video_fps(df) is one frame short of that.
    frame_dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / fps
    duration = float(t[-1]) + frame_dt if len(t) else 0.0
    rows, dropped = [], 0
    for start in window_starts(duration, window_s, stride_s):
        seg = df[(t >= start) & (t < start + window_s)]
        if seg.empty:
            dropped += 1
            continue
        coverage = float(np.isfinite(blinks.openness(ear_column(seg, use_corrected),
                                                     baseline)).mean())
        if coverage < min_coverage:
            dropped += 1
            continue
        summary, _ = summarise_frames(seg, baseline, fps, use_corrected)
        summary.update({"window_start_s": float(start), "window_s": float(window_s),
                        "stride_s": float(stride_s), "coverage": coverage})
        rows.append(summary)
    return rows, dropped


class BaselineInvariantError(AssertionError):
    """A window was scored against something other than its subject's alert baseline."""


def assert_subject_baselines(rows: pd.DataFrame,
                             baselines: dict[str, blinks.EyeBaseline]) -> None:
    """Every row of a subject carries that subject's one baseline. Raises, never warns.

    This is the check that windows did not get a baseline of their own: one
    subject, one (open, closed) pair, equal to the pair from build_baselines.
    """
    per_subject = rows.groupby("subject_id")[["ear_open_baseline", "ear_closed_baseline"]].nunique()
    several = per_subject[(per_subject > 1).any(axis=1)]
    if len(several):
        raise BaselineInvariantError(
            f"subjects with more than one eye baseline across their rows: {list(several.index)}")
    for subject, g in rows.groupby("subject_id"):
        expected = baselines.get(subject)
        if expected is None:
            raise BaselineInvariantError(f"{subject}: rows exist but no subject baseline was built")
        got = (g["ear_open_baseline"].iloc[0], g["ear_closed_baseline"].iloc[0])
        if not np.allclose(got, (expected.ear_open, expected.ear_closed)):
            raise BaselineInvariantError(
                f"{subject}: rows use baseline {got}, subject baseline is "
                f"{(expected.ear_open, expected.ear_closed)}")
        if (g["baseline_source"] != expected.source).any():
            raise BaselineInvariantError(f"{subject}: baseline_source differs from {expected.source}")


def window_output_path(out_dir: Path, window_s: float, stride_s: float) -> Path:
    return out_dir / f"window_features_w{window_s:g}_s{stride_s:g}.csv"


def run_windows(manifest: pd.DataFrame, frames_dir: Path, out_dir: Path,
                window_lengths: list[float], stride_s: float | None,
                use_corrected: bool, min_coverage: float) -> int:
    """Window mode: one CSV per window length, each parquet read once."""
    if stride_s is not None and len(window_lengths) > 1:
        raise SystemExit("--stride-s applies to one --window-s; omit it to use half of each")
    configs = [(w, stride_s if stride_s is not None else w / 2.0) for w in window_lengths]

    print("[aggregate] estimating per-subject eye baselines from whole alert videos")
    baselines, problems = build_baselines(manifest, frames_dir, use_corrected)
    print(f"[aggregate] baselines for {len(baselines)} subjects")
    for p in problems:
        print(f"  WARNING {p}")

    rows = {c: [] for c in configs}
    dropped = {c: {} for c in configs}
    missing = 0
    for row in manifest.to_dict("records"):
        subject = row["subject_id"]
        df = load_frames(frames_dir, row["video_id"]) if subject in baselines else None
        if df is None:
            missing += 1
            continue
        ident = {"video_id": row["video_id"], "subject_id": subject,
                 "fold": row.get("fold", ""), "class": row.get("class", ""),
                 "part": row["part"] if isinstance(row.get("part"), str) else ""}
        for c in configs:
            win_rows, n_drop = window_features(df, baselines[subject], c[0], c[1],
                                               use_corrected, min_coverage)
            rows[c].extend({**ident, **r} for r in win_rows)
            dropped[c][row["video_id"]] = (subject, n_drop, len(win_rows))

    if not any(rows.values()):
        print("no windows produced; run the extraction pass first")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    lead = ["video_id", "subject_id", "fold", "class", "part",
            "window_start_s", "window_s", "stride_s", "coverage"]
    all_subjects = sorted(set(manifest["subject_id"]))
    for (w, s), win_rows in rows.items():
        out = pd.DataFrame(win_rows)
        out = out[lead + [c for c in out.columns if c not in lead]]
        assert_subject_baselines(out, baselines)
        path = window_output_path(out_dir, w, s)
        out.to_csv(path, index=False)

        per_video = pd.DataFrame.from_dict(dropped[(w, s)], orient="index",
                                           columns=["subject_id", "dropped", "kept"])
        per_subject = per_video.groupby("subject_id")[["dropped", "kept"]].sum()
        no_windows_v = sorted(per_video.index[per_video.kept == 0])
        no_windows_s = sorted(set(all_subjects) - set(out["subject_id"]))
        meta = {
            "window_s": w, "stride_s": s, "min_coverage": min_coverage,
            "n_windows": len(out), "n_videos": int(out["video_id"].nunique()),
            "n_subjects": int(out["subject_id"].nunique()),
            "windows_dropped_low_coverage": int(per_video.dropped.sum()),
            "videos_without_windows": no_windows_v,
            "subjects_without_windows": no_windows_s,
            # Coverage drops are a fairness signal too: per subject, not only in total.
            "dropped_by_subject": per_subject[per_subject.dropped > 0].to_dict("index"),
            "baseline_problems": problems, "videos_without_frames": missing,
            "baseline": "per subject, whole alert video (asserted)",
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=int))
        print(f"\nwindow {w:g}s stride {s:g}s: {len(out)} windows from "
              f"{meta['n_videos']} videos -> {path}")
        print(f"  dropped {meta['windows_dropped_low_coverage']} windows below "
              f"{min_coverage:.0%} coverage")
        if no_windows_v:
            print(f"  WARNING videos with no window left: {', '.join(no_windows_v)}")
        if no_windows_s:
            print(f"  WARNING subjects with no window left: {', '.join(no_windows_s)}")
    if missing:
        print(f"WARNING {missing} manifest rows had no extracted frames or no baseline")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate per-frame features into per-video drowsiness features."
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--frames", type=Path, default=Path("data/frames"))
    ap.add_argument("--out", type=Path, default=Path("data/video_features.csv"))
    ap.add_argument("--events-out", type=Path, default=Path("data/closure_events.csv"))
    ap.add_argument("--raw-ear", action="store_true",
                    help="use uncorrected EAR instead of the pose-corrected column")
    ap.add_argument("--window-s", type=float, nargs="+", default=None,
                    help="window mode: cut each video into windows of this many seconds "
                         "(several values allowed) and write one window_features CSV each "
                         "next to --out, instead of per-video features")
    ap.add_argument("--stride-s", type=float, default=None,
                    help="window stride in seconds; default half the window")
    ap.add_argument("--min-coverage", type=float, default=config.MIN_WINDOW_COVERAGE,
                    help="drop windows with a smaller fraction of tracked frames")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, dtype=str)
    use_corrected = not args.raw_ear
    if args.window_s:
        return run_windows(manifest, args.frames, args.out.parent, args.window_s,
                           args.stride_s, use_corrected, args.min_coverage)

    print("[aggregate] estimating per-subject eye baselines from alert videos")
    baselines, problems = build_baselines(manifest, args.frames, use_corrected)
    print(f"[aggregate] baselines for {len(baselines)} subjects")
    for p in problems:
        print(f"  WARNING {p}")

    rows: list[dict] = []
    event_rows: list[dict] = []
    missing = 0

    for row in manifest.to_dict("records"):
        subject = row["subject_id"]
        if subject not in baselines:
            missing += 1
            continue
        df = load_frames(args.frames, row["video_id"])
        if df is None:
            missing += 1
            continue

        fps = video_fps(df)
        summary, open_series = summarise_frames(df, baselines[subject], fps, use_corrected)
        summary.update(resolution_features(row))
        summary.update({
            "video_id": row["video_id"],
            "subject_id": subject,
            "fold": row.get("fold", ""),
            "class": row.get("class", ""),
            # Which half of a split recording, blank if unsplit (NaN from read_csv).
            "part": row["part"] if isinstance(row.get("part"), str) else "",
        })
        rows.append(summary)

        for ev in blinks.events_to_records(blinks.detect_closures(open_series, fps)):
            ev.update({"video_id": row["video_id"], "subject_id": subject,
                       "class": row.get("class", "")})
            event_rows.append(ev)

    if not rows:
        print("no videos aggregated; run the extraction pass first")
        return 1

    out = pd.DataFrame(rows)
    lead = ["video_id", "subject_id", "fold", "class", "part"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    for k, name in enumerate(("pitch_centered_std", "pitch_centered_range")):
        out.insert(out.columns.get_loc("pitch_range") + 1 + k, name, out.pop(name))
    out.insert(out.columns.get_loc("low_quality") + 1, "low_resolution",
               out.pop("low_resolution"))
    assert_subject_baselines(out, baselines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.DataFrame(event_rows).to_csv(args.events_out, index=False)

    confounded = confounded_by_class(out)
    low_res = out[out["low_resolution"] == True]  # noqa: E712, the column may hold None
    unmeasured = int(out["low_resolution"].isna().sum())
    meta = {
        "n_videos": len(out),
        "n_subjects": int(out["subject_id"].nunique()),
        "ear_column": "ear_corrected" if use_corrected else "ear",
        "baseline_problems": problems,
        "videos_without_features": missing,
        "pitch_confounded_by_class": confounded.to_dict("index"),
        "min_resolution_short_side": config.MIN_RESOLUTION_SHORT_SIDE,
        "low_resolution_videos": sorted(low_res["video_id"]),
        "low_resolution_subjects": sorted(low_res["subject_id"].unique()),
        "videos_without_resolution": unmeasured,
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {len(out)} videos to {args.out}")
    print(f"wrote {len(event_rows)} closure events to {args.events_out}")
    if missing:
        print(f"WARNING {missing} manifest rows had no extracted frames")

    present = [c for c in config.CLASS_ORDER if c in set(out["class"])]
    if present:
        cols = ["perclos80_mean", "n_microsleeps", "blink_rate_per_min",
                "mean_reopening_speed", "detection_rate"]
        print("\nmean by class (sanity check: PERCLOS should rise left to right)")
        print(out.groupby("class")[cols].mean().reindex(present).round(3).to_string())

    print("\npitch_confounded videos by class (pitch moves with vertical position; a "
          "slump or a real head drop)")
    print(confounded.round(3).to_string())

    print(f"\nlow_resolution (short side < {config.MIN_RESOLUTION_SHORT_SIDE}px): "
          f"{len(low_res)} videos, {low_res['subject_id'].nunique()} subjects, flagged "
          f"and kept")
    if len(low_res):
        print("  subjects: " + ", ".join(sorted(low_res["subject_id"].unique())))
    if unmeasured:
        print(f"WARNING {unmeasured} videos have no resolution in the manifest; "
              f"rebuild it with src.manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
