"""One-command check that the landmarker works on this machine.

Run this on a single real video before launching the full extraction. It
confirms MediaPipe loads, the face is being tracked, and the EAR signal has
the shape it should, and it saves a plot so you can see blinks with your own
eyes rather than trusting a number.

    python -m src.selfcheck --video "path/to/one/video.mp4"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import blinks, geometry
from .extract import process_video
from .landmarks import FaceLandmarker


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the landmarker on one video.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="how much of the video to check")
    ap.add_argument("--max-width", type=int, default=640)
    ap.add_argument("--plot", type=Path, default=Path("data/selfcheck_ear.png"))
    args = ap.parse_args()

    if not args.video.exists():
        print(f"video not found: {args.video}")
        return 1

    print(f"checking {args.video.name}\n")
    with FaceLandmarker(model_path=args.model) as fl:
        print(f"landmarker backend : {fl.backend}")
        df, meta = process_video(
            args.video, fl,
            max_width=args.max_width,
            max_frames=int(args.seconds * 30),
        )

    fps = meta["fps"]
    ear = df["ear"].to_numpy(dtype=float)
    finite = ear[np.isfinite(ear)]

    print(f"rotation_deg       : {meta['rotation_deg']} ({meta['rotation_method']})")
    print(f"stored resolution  : {meta['stored_width']}x{meta['stored_height']}")
    print(f"effective resol.   : {meta['effective_width']}x{meta['effective_height']}"
          f"{'  LOW-RESOLUTION, flagged not dropped' if meta['low_resolution'] else ''}")
    print(f"frames read        : {len(df)}")
    print(f"reported fps       : {fps:.1f}")
    print(f"seconds per frame  : {meta['seconds_elapsed'] / max(1, len(df)):.4f}")
    print(f"detection rate     : {meta['detection_rate']:.1%}")

    checks: list[tuple[str, bool, str]] = []
    checks.append((
        "face detected on most frames",
        meta["detection_rate"] > 0.70,
        f"{meta['detection_rate']:.1%}",
    ))

    if finite.size < 30:
        print("\nnot enough detected frames to judge the signal")
        return 1

    lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 90))
    print(f"EAR p2 / p90       : {lo:.3f} / {hi:.3f}")
    checks.append((
        "EAR in a plausible range",
        0.02 < hi < 0.60,
        f"p90={hi:.3f} (expect roughly 0.20 to 0.40 for an open eye)",
    ))
    checks.append((
        "EAR actually varies",
        (hi - lo) > 0.03,
        f"span={hi - lo:.3f} (a flat signal means the eye landmarks are stuck)",
    ))

    baseline = blinks.estimate_baseline(ear, source=args.video.name)
    open_series = blinks.openness(ear, baseline)
    events = blinks.detect_closures(open_series, fps)
    blink_events = [e for e in events if e.kind == "blink"]
    minutes = len(df) / fps / 60.0
    rate = len(blink_events) / minutes if minutes > 0 else 0.0
    p80 = blinks.perclos_series(open_series, fps)

    print(f"closure events     : {len(events)} "
          f"({len(blink_events)} blinks, "
          f"{sum(1 for e in events if e.kind == 'microsleep')} microsleeps)")
    print(f"blink rate         : {rate:.1f} per minute")
    print(f"mean PERCLOS80     : {np.nanmean(p80['perclos']):.3f}")

    checks.append((
        "blinks detected at a human rate",
        2.0 <= rate <= 60.0,
        f"{rate:.1f}/min (resting humans blink roughly 15 to 20 times a minute)",
    ))

    print("\n" + "-" * 58)
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  [{verdict(ok)}] {name:<34} {detail}")
    print("-" * 58)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = df["t_s"].to_numpy(dtype=float)
        axes[0].plot(t, ear, lw=0.8)
        axes[0].set_ylabel("EAR")
        axes[0].set_title(f"{args.video.name}  (dips should be blinks)")
        axes[1].plot(t, open_series, lw=0.8, color="tab:orange")
        axes[1].axhline(0.2, ls="--", lw=0.8, color="grey")
        axes[1].set_ylabel("openness")
        axes[1].set_xlabel("seconds")
        for e in events:
            axes[1].axvspan(e.start_s, e.start_s + e.duration_s, alpha=0.2,
                            color="tab:red")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"\nplot saved to {args.plot}")
        print("Open it. Each shaded band should line up with a real blink.")
    except ImportError:
        print("\n(install matplotlib to also get the EAR plot)")

    if not all_ok:
        print("\nSomething looks wrong. Check the plot before running the full job.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
