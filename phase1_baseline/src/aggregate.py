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

from . import blinks, config


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
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, dtype=str)
    use_corrected = not args.raw_ear

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

        baseline = baselines[subject]
        fps = float(len(df)) / float(df["t_s"].iloc[-1]) if (
            len(df) > 1 and df["t_s"].iloc[-1] > 0
        ) else 30.0

        open_series = blinks.openness(ear_column(df, use_corrected), baseline)
        summary = blinks.summarise(
            open_series,
            df["mar"].to_numpy(dtype=float),
            df["pitch"].to_numpy(dtype=float),
            fps,
            baseline,
        )
        summary.update({
            "video_id": row["video_id"],
            "subject_id": subject,
            "fold": row.get("fold", ""),
            "class": row.get("class", ""),
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
    lead = ["video_id", "subject_id", "fold", "class"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.DataFrame(event_rows).to_csv(args.events_out, index=False)

    meta = {
        "n_videos": len(out),
        "n_subjects": int(out["subject_id"].nunique()),
        "ear_column": "ear_corrected" if use_corrected else "ear",
        "baseline_problems": problems,
        "videos_without_features": missing,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
