"""Pass 1: video -> per-frame feature table.

Reads the manifest, runs the face landmarker over every frame of every video,
and writes one Parquet file per video holding EAR, MAR and head pose per frame.

Design notes that matter when this runs for hours:

* Resumable. A video whose output already exists is skipped, so an interrupted
  run costs you one video, not the whole job.
* Frames are downscaled before inference. Landmark accuracy saturates well
  below 1080p and the speedup is large. EAR and MAR are ratios, so downscaling
  does not change them.
* A frame where no face is found is written with NaNs, never dropped. Dropping
  it would silently compress the timeline and corrupt every duration we later
  measure.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import geometry
from .landmarks import FaceLandmarker

FEATURE_COLUMNS = [
    "frame", "t_s", "detected",
    "ear_left", "ear_right", "ear", "ear_corrected",
    "mar", "yaw", "pitch", "roll",
]


def process_video(
    path: Path,
    landmarker: FaceLandmarker,
    max_width: int = 640,
    stride: int = 1,
    max_frames: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or fps > 240:
        fps = 30.0  # some phone-recorded files report nonsense

    rows: list[dict] = []
    idx = 0
    kept = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue

        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        fh, fw = frame.shape[:2]

        timestamp_ms = int(round(1000.0 * idx / fps))
        pts = landmarker(frame, timestamp_ms)

        row = {"frame": idx, "t_s": idx / fps, "detected": pts is not None}
        if pts is None:
            row.update({c: float("nan") for c in FEATURE_COLUMNS[3:]})
        else:
            row.update(geometry.frame_features(pts, fw, fh))
        rows.append(row)

        kept += 1
        idx += 1
        if max_frames is not None and kept >= max_frames:
            break

    cap.release()
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    detected = float(df["detected"].mean()) if len(df) else 0.0
    meta = {
        "path": str(path),
        "fps": fps,
        "effective_fps": fps / stride,
        "n_frames_declared": n_declared,
        "n_frames_read": len(df),
        "detection_rate": detected,
        "stride": stride,
        "max_width": max_width,
        "seconds_elapsed": round(time.time() - t_start, 1),
        "backend": landmarker.backend,
    }
    return df, meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract per-frame facial features from the video manifest."
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path,
                    help="dataset root that manifest relpaths are relative to")
    ap.add_argument("--out", type=Path, default=Path("data/frames"))
    ap.add_argument("--model", type=Path, default=None,
                    help="path to face_landmarker.task (downloaded if omitted)")
    ap.add_argument("--max-width", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame; keep at 1 for blink timing")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per video, for smoke tests")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N videos")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, dtype=str)
    if args.limit:
        manifest = manifest.head(args.limit)
    args.out.mkdir(parents=True, exist_ok=True)

    log_path = args.out / "_extraction_log.csv"
    log_exists = log_path.exists()

    if args.stride != 1:
        print(f"WARNING: stride={args.stride} lowers the effective frame rate to "
              f"{30 / args.stride:.1f} fps. Blink durations below "
              f"{args.stride / 30:.2f}s become unmeasurable.")

    done = skipped = failed = 0
    with FaceLandmarker(model_path=args.model) as fl:
        print(f"[extract] landmarker backend: {fl.backend}")
        records = manifest.to_dict("records")
        for i, row in enumerate(records, 1):
            out_path = args.out / f"{row['video_id']}.parquet"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            video_path = args.root / row["relpath"]
            label = f"[{i}/{len(records)}] {row['video_id']}"
            if not video_path.exists():
                print(f"{label}  MISSING {video_path}")
                failed += 1
                continue

            try:
                df, meta = process_video(
                    video_path, fl,
                    max_width=args.max_width,
                    stride=args.stride,
                    max_frames=args.max_frames,
                )
            except Exception as exc:
                print(f"{label}  FAILED {type(exc).__name__}: {exc}")
                failed += 1
                continue

            df.to_parquet(out_path, index=False)
            meta.update({
                "video_id": row["video_id"],
                "subject_id": row["subject_id"],
                "fold": row.get("fold", ""),
                "class": row.get("class", ""),
            })
            with log_path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(meta.keys()))
                if not log_exists:
                    w.writeheader()
                    log_exists = True
                w.writerow(meta)

            flag = "  LOW-DETECTION" if meta["detection_rate"] < 0.70 else ""
            print(f"{label}  {meta['n_frames_read']} frames  "
                  f"det {meta['detection_rate']:.1%}  "
                  f"{meta['seconds_elapsed']}s{flag}")
            done += 1

    print(f"\ndone {done}   skipped {skipped}   failed {failed}")
    print(f"log: {log_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
