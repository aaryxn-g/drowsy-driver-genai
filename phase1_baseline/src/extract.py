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
* Frames are put the right way up first, from the container's rotation flag, and
  only then downscaled. See video.py for why that is checked rather than assumed.
* One landmarker per video. MediaPipe's VIDEO mode needs strictly increasing
  timestamps for the life of an instance, and each video starts again at 0.
* Shardable. `--shard i/n` handles every row where row_index % n == i-1, so n
  processes can share one manifest. Parquet names are disjoint because video_id is,
  but each shard keeps its own log since concurrent appends to one CSV interleave.
  Combine the logs afterwards with `--merge-logs`.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import config, geometry, video
from .landmarks import FaceLandmarker

FEATURE_COLUMNS = [
    "frame", "t_s", "detected",
    "ear_left", "ear_right", "ear", "ear_corrected",
    "mar", "yaw", "pitch", "roll",
    "pitch_centered", "yaw_centered",
    "nose_x_norm", "nose_y_norm", "face_scale",
]


def process_video(
    path: Path,
    landmarker: FaceLandmarker,
    max_width: int = 640,
    stride: int = 1,
    max_frames: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    idx = 0
    kept = 0
    t_start = time.time()
    inference_wh = (0, 0)

    with video.OrientedVideo(path) as source:
        info = source.info
        n_declared = source.n_frames_declared
        fps = info.fps

        while True:
            frame = source.read()
            if frame is None:
                break
            if idx % stride != 0:
                idx += 1
                continue

            # Rotation has already happened inside read(), before this downscale.
            frame = video.downscale(frame, max_width)
            fh, fw = frame.shape[:2]
            inference_wh = (fw, fh)

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

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    detected = float(df["detected"].mean()) if len(df) else 0.0
    meta = {
        "path": str(path),
        **info.as_row(),
        "inference_width": inference_wh[0],
        "inference_height": inference_wh[1],
        "low_resolution": video.is_low_resolution(
            info.effective_width, info.effective_height),
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


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def extract_videos(
    records: list[dict],
    root: Path,
    out_dir: Path,
    log_path: Path,
    make_landmarker: Callable[[], FaceLandmarker],
    max_width: int = 640,
    stride: int = 1,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Run pass 1 over manifest rows. Returns (done, skipped, failed).

    Each video gets its own landmarker from `make_landmarker`, closed when that
    video ends. The tasks API in VIDEO mode requires strictly increasing timestamps
    over an instance's life, and every video's clock starts at 0, so a shared
    instance accepts the first video and raises on every one after it. Offsetting
    timestamps across videos would avoid the error but would carry tracking state
    from one subject's video into the next. Reloading the cached .task file costs a
    second or two per video.
    """
    root, out_dir, log_path = Path(root), Path(out_dir), Path(log_path)
    log_exists = log_path.exists()
    done = skipped = failed = 0
    backend_reported = False
    for i, row in enumerate(records, 1):
        out_path = out_dir / f"{row['video_id']}.parquet"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        video_path = root / row["relpath"]
        label = f"[{i}/{len(records)}] {row['video_id']}"
        if not video_path.exists():
            print(f"{label}  MISSING {video_path}")
            failed += 1
            continue

        try:
            with make_landmarker() as fl:
                if not backend_reported:
                    print(f"[extract] landmarker backend: {fl.backend}")
                    backend_reported = True
                df, meta = process_video(
                    video_path, fl,
                    max_width=max_width,
                    stride=stride,
                    max_frames=max_frames,
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
            # Blank for an unsplit video. pandas reads a blank cell as NaN.
            "part": _text(row.get("part")),
        })
        with log_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(meta.keys()))
            if not log_exists:
                w.writeheader()
                log_exists = True
            w.writerow(meta)

        flag = "  LOW-DETECTION" if meta["detection_rate"] < config.MIN_DETECTION_RATE else ""
        if meta["low_resolution"]:
            flag += "  LOW-RESOLUTION"
        if meta["rotation_deg"]:
            flag += f"  rotated {meta['rotation_deg']} ({meta['rotation_method']})"
        print(f"{label}  {meta['n_frames_read']} frames  "
              f"{meta['effective_width']}x{meta['effective_height']}  "
              f"det {meta['detection_rate']:.1%}  "
              f"{meta['seconds_elapsed']}s{flag}")
        done += 1
    return done, skipped, failed


def parse_shard(text: str) -> tuple[int, int]:
    """Parse "3/16" into (3, 16). Shards count from 1, as in "shard 3 of 16"."""
    match = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"expected i/n such as 3/16, got {text!r}")
    i, n = int(match.group(1)), int(match.group(2))
    if not 1 <= i <= n:
        raise argparse.ArgumentTypeError(f"need 1 <= i <= n, got {text!r}")
    return i, n


def select_shard(manifest: pd.DataFrame, i: int, n: int) -> pd.DataFrame:
    """Rows where row_index % n == i-1, counted by position in the manifest."""
    return manifest.iloc[i - 1::n]


def log_filename(i: int, n: int) -> str:
    return "_extraction_log.csv" if n == 1 else f"_extraction_log_shard{i}of{n}.csv"


def merge_shard_logs(out_dir: Path) -> pd.DataFrame:
    """Combine the shard logs into `_extraction_log.csv`, one row per video.

    Any existing combined log is read first, so rows from an earlier run survive. A
    video logged more than once, after a redo with --overwrite, keeps its newest row,
    which is why shard logs are read oldest first. The shard logs are left in place.
    """
    out_dir = Path(out_dir)
    combined = out_dir / log_filename(1, 1)
    shards = sorted(out_dir.glob("_extraction_log_shard*of*.csv"),
                    key=lambda p: p.stat().st_mtime)
    if not shards:
        raise FileNotFoundError(f"no shard logs in {out_dir}")
    sources = ([combined] if combined.exists() else []) + shards
    merged = pd.concat([pd.read_csv(p, dtype=str) for p in sources], ignore_index=True)
    merged = merged.drop_duplicates("video_id", keep="last").sort_values("video_id")
    temp = combined.with_suffix(".csv.tmp")
    merged.to_csv(temp, index=False)
    temp.replace(combined)
    return merged


def run_merge(out_dir: Path, manifest_path: Path | None) -> int:
    try:
        merged = merge_shard_logs(out_dir)
    except FileNotFoundError as exc:
        print(f"[extract] {exc}")
        return 1
    print(f"[extract] merged {len(merged)} videos into {out_dir / log_filename(1, 1)}")
    if manifest_path is None:
        return 0
    manifest = pd.read_csv(manifest_path, dtype=str)
    absent = sorted(set(manifest["video_id"]) - set(merged["video_id"]))
    if absent:
        print(f"WARNING {len(absent)} manifest videos have no log row: {absent}")
    return 1 if absent else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract per-frame facial features from the video manifest."
    )
    ap.add_argument("--manifest", type=Path, default=None,
                    help="required, except with --merge-logs where it is optional and "
                         "reports videos that have no log row")
    ap.add_argument("--root", type=Path, default=None,
                    help="dataset root that manifest relpaths are relative to; required "
                         "unless --merge-logs")
    ap.add_argument("--out", type=Path, default=Path("data/frames"))
    ap.add_argument("--model", type=Path, default=None,
                    help="path to face_landmarker.task (downloaded if omitted)")
    ap.add_argument("--max-width", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame; keep at 1 for blink timing")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per video, for smoke tests")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N videos; applied before --shard")
    ap.add_argument("--shard", type=parse_shard, default=(1, 1), metavar="I/N",
                    help="process only rows where row_index %% N == I-1, e.g. 3/16")
    ap.add_argument("--merge-logs", action="store_true",
                    help="combine the shard logs in --out into _extraction_log.csv, "
                         "then exit; run it after every shard has finished")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.merge_logs:
        return run_merge(args.out, args.manifest)
    if args.manifest is None or args.root is None:
        ap.error("--manifest and --root are required unless --merge-logs is given")

    manifest = pd.read_csv(args.manifest, dtype=str)
    if args.limit:
        manifest = manifest.head(args.limit)
    shard_i, shard_n = args.shard
    manifest = select_shard(manifest, shard_i, shard_n)
    args.out.mkdir(parents=True, exist_ok=True)

    log_path = args.out / log_filename(shard_i, shard_n)

    if args.stride != 1:
        print(f"WARNING: stride={args.stride} lowers the effective frame rate to "
              f"{30 / args.stride:.1f} fps. Blink durations below "
              f"{args.stride / 30:.2f}s become unmeasurable.")

    model_path = args.model
    done, skipped, failed = extract_videos(
        manifest.to_dict("records"), args.root, args.out, log_path,
        make_landmarker=lambda: FaceLandmarker(model_path=model_path),
        max_width=args.max_width, stride=args.stride, max_frames=args.max_frames,
        overwrite=args.overwrite,
    )

    print(f"\ndone {done}   skipped {skipped}   failed {failed}")
    print(f"log: {log_path}")
    if shard_n > 1:
        print("once every shard has finished, combine the logs with "
              f"`python -m src.extract --merge-logs --out {args.out}`")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
