"""Stage 3 of the Phase 2 pilot: eight 60 s driving clips spanning the PERCLOS range.

Selection is deterministic and fixed before generation:
  * candidates: Phase 1 60 s windows (`window_features_w60_s30.csv`) with coverage and
    detection_rate >= 0.95, not low_quality, not pitch_confounded, from videos not flagged
    low_resolution;
  * eight target perclos80_mean values from the alert floor to the drowsy tail (the 95th
    percentile of all windows is 0.75); windows near 1.0 are excluded as likely tracking or
    baseline artefacts rather than usable driving motion;
  * for each target in turn, the closest window from a subject not yet used;
  * the result must hold all three classes, or this refuses to continue.

Start nudge. LivePortrait applies expression relative to the FIRST driving frame, so a clip
that starts mid-blink offsets every later eye state. Each start is moved to the nearest frame
whose openness (Phase 1 scale, subject's alert-video baseline) is open and stays open from
0.1 s before to 0.4 s after (no blink starting or ending). Two tiers, first match wins:
  tier 1: openness >= 0.85, hold >= 0.70, searched +-5 s then +-15 s;
  tier 2: openness >= 0.75, hold >= 0.60, within +-15 s.
Tier 2 exists because a deep-drowsy minute may never reach 0.85: drv_13_10 has median openness
0.01 over its window, and the nearest tier-1 frame is 62 s away, in a different minute (PERCLOS
0.92). A tier-2 start is slightly less open, so with relative motion the synthetic eyes may run
slightly more open than the driving ones. Tier and start openness are recorded per clip.

Two outputs per clip:
  * an archival segment cut with ffmpeg stream copy from the keyframe at or before the nudged
    start (stream copy can only start on a keyframe);
  * the LivePortrait driving input: frames [start_frame, start_frame + n_frames) of the source,
    decoded upright through phase1 `video.OrientedVideo` (the path verified on 182 videos),
    fixed square crop around the tracked face, 512x512, libx264 CRF 12.
Ground truth is the measured window perclos80_mean, as specified. `perclos80_cut` re-measures
exactly those frames with the subject's alert-video baseline through unchanged Phase 1 code.

Runs in the `drowsy` env (pandas, Phase 1 code):
    python -m gen.pilot_driving --root /home/Shivang/data/UTA-RLDD
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
P1 = HERE.parent / "phase1_baseline"
sys.path.insert(0, str(P1))
from src import aggregate, blinks, config, video  # noqa: E402  phase1_baseline/src, unchanged

TARGETS = (0.01, 0.03, 0.06, 0.10, 0.18, 0.30, 0.50, 0.75)
MAX_PERCLOS = 0.95
MIN_QUALITY = 0.95
CLIP_S = 60.0
NUDGE_TIERS = ((1, 0.85, 0.70, (5.0, 15.0)), (2, 0.75, 0.60, (15.0,)))
NUDGE_PRE_S, NUDGE_POST_S = 0.1, 0.4
CROP_SIZE = 512
CROP_SIDE_IOD = 4.2       # crop side in outer-eye-corner distances: LivePortrait's driving
                          # crop is 2.2 x landmark bbox, brow-to-chin ~1.9 eye distances
CROP_UP_IOD = 0.35        # centre this far above the nose tip
OUT = HERE / "pilot" / "driving"


def candidates(windows: pd.DataFrame, video_features: pd.DataFrame) -> pd.DataFrame:
    lowres = set(video_features.loc[video_features.low_resolution.astype(str).str.lower()
                                    == "true", "video_id"])
    w = windows[(windows.coverage >= MIN_QUALITY) & (windows.detection_rate >= MIN_QUALITY)
                & ~windows.low_quality.astype(bool) & ~windows.pitch_confounded.astype(bool)
                & (windows.perclos80_mean < MAX_PERCLOS) & ~windows.video_id.isin(lowres)]
    return w.sort_values(["video_id", "window_start_s"]).reset_index(drop=True)


def select(w: pd.DataFrame, targets=TARGETS) -> pd.DataFrame:
    used, picks = set(), []
    for t in targets:
        pool = w[~w.subject_id.isin(used)]
        if pool.empty:
            raise ValueError("ran out of subjects")
        d = (pool.perclos80_mean - t).abs()
        best = pool.loc[d.sort_values(kind="stable").index[0]]
        used.add(best.subject_id)
        picks.append(best.to_dict() | {"target_perclos": t})
    out = pd.DataFrame(picks)
    if set(out["class"]) != {"alert", "low_vigilant", "drowsy"}:
        raise ValueError(f"selection does not cover all three classes: {out['class'].tolist()}")
    if out.subject_id.nunique() != len(out):
        raise ValueError("a subject appears twice")
    return out



def keyframe_before(src: Path, t: float) -> float:
    """Time of the last video keyframe at or before t, read from packet flags."""
    lo = max(t - 15.0, 0.0)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-read_intervals", f"{lo}%{t + 0.5}", "-show_entries",
                        "packet=pts_time,flags", "-of", "csv=p=0", str(src)],
                       capture_output=True, text=True, check=True)
    keys = [float(p) for p, f in (ln.split(",")[:2] for ln in r.stdout.split() if "," in ln)
            if "K" in f and p != "N/A" and float(p) <= t + 1e-3]
    if not keys:
        raise ValueError(f"no keyframe found before {t}s in {src}")
    return max(keys)


def cut(src: Path, dst: Path, start: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src),
                    "-t", f"{CLIP_S:.3f}", "-map", "0:v:0", "-c", "copy",
                    "-avoid_negative_ts", "make_zero", str(dst)], check=True)


def subject_baseline(subject: str, manifest: pd.DataFrame, frames: Path) -> blinks.EyeBaseline:
    baselines, problems = aggregate.build_baselines(manifest[manifest.subject_id == subject],
                                                    frames, use_corrected=True)
    if subject not in baselines:
        raise ValueError(f"no baseline for {subject}: {problems}")
    return baselines[subject]


def nudge_start(open_series: np.ndarray, fps: float, start_frame: int, n_frames: int,
                tiers=NUDGE_TIERS) -> tuple[int, int]:
    """(frame, tier): nearest frame to start_frame where the eye is open and no blink is near.

    Ties go to the earlier frame. The clip must still fit inside the video. Raises rather than
    starting on a closed or untracked eye.
    """
    pre, post = int(round(NUDGE_PRE_S * fps)), int(round(NUDGE_POST_S * fps))
    o = np.nan_to_num(open_series, nan=-1.0)  # an untracked frame never qualifies
    last = len(o) - n_frames
    for tier, open_min, hold_min, radii in tiers:
        for radius_s in radii:
            r = int(round(radius_s * fps))
            for d in sorted(range(-r, r + 1), key=lambda k: (abs(k), k)):
                i = start_frame + d
                if i - pre < 0 or i > last:
                    continue
                if o[i] >= open_min and (o[i - pre:i + post + 1] >= hold_min).all():
                    return i, tier
    raise ValueError(f"no open-eye start in any tier near frame {start_frame}")


def face_box(df: pd.DataFrame, width: int, height: int) -> tuple[int, int, int]:
    """Fixed square crop (x0, y0, side) from the median tracked face over the clip."""
    d = df[df.detected.astype(bool)]
    iod = float(np.median(d.face_scale)) * width
    cx = float(np.median(d.nose_x_norm)) * width
    cy = float(np.median(d.nose_y_norm)) * height - CROP_UP_IOD * iod
    side = int(round(CROP_SIDE_IOD * iod))
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def crop_frame(frame: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """Square crop, padded with black where it leaves the frame, resized to CROP_SIZE."""
    import cv2
    h, w = frame.shape[:2]
    out = np.zeros((side, side, 3), dtype=frame.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def write_crop(src: Path, dst: Path, start_frame: int, n_frames: int, fps: float,
               box: tuple[int, int, int]) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{CROP_SIZE}x{CROP_SIZE}", "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p",
         str(dst)], stdin=subprocess.PIPE)
    written, info = 0, None
    with video.OrientedVideo(src) as v:
        info = v.info
        i = 0
        while written < n_frames and (frame := v.read()) is not None:
            if i >= start_frame:
                enc.stdin.write(crop_frame(frame, *box).tobytes())
                written += 1
            i += 1
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {dst}")
    if written != n_frames:
        raise RuntimeError(f"{src}: wrote {written} of {n_frames} frames")
    return {"rotation_method": info.rotation_method, "effective_width": info.effective_width,
            "effective_height": info.effective_height}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    frames_dir = P1 / "data" / "frames"
    windows = pd.read_csv(P1 / "data" / "window_features_w60_s30.csv",
                          dtype={"subject_id": str, "video_id": str})
    manifest = pd.read_csv(P1 / "data" / "manifest.csv", dtype=str)
    vf = pd.read_csv(P1 / "data" / "video_features.csv", dtype={"video_id": str})
    picks = select(candidates(windows, vf))

    rows = []
    for _, p in picks.iterrows():
        m = manifest[manifest.video_id == p.video_id].iloc[0]
        src = a.root / m.relpath
        df = aggregate.load_frames(frames_dir, p.video_id)
        fps = aggregate.video_fps(df)
        base = subject_baseline(p.subject_id, manifest, frames_dir)
        op = blinks.openness(aggregate.ear_column(df, True), base)
        n_frames = int(round(CLIP_S * fps))
        win_frame = int(np.searchsorted(df.t_s.to_numpy(), float(p.window_start_s)))
        start, tier = nudge_start(op, fps, win_frame, n_frames)
        seg = df.iloc[start:start + n_frames].reset_index(drop=True)
        summary, _ = aggregate.summarise_frames(seg, base, fps, use_corrected=True)
        start_s = float(df.t_s.iloc[start])

        clip_id = f"drv_{p.video_id}_t{int(p.window_start_s)}"
        key = keyframe_before(src, start_s)
        cut(src, a.out / "segments" / f"{clip_id}.mp4", key)
        width = int(m.effective_width); height = int(m.effective_height)
        box = face_box(seg, width, height)
        vinfo = write_crop(src, a.out / f"{clip_id}.mp4", start, n_frames, fps, box)
        rows.append({
            "driving_clip_id": clip_id, "video_id": p.video_id, "subject_id": p.subject_id,
            "fold": int(p.fold), "class": p["class"], "window_start_s": float(p.window_start_s),
            "perclos80_mean": float(p.perclos80_mean), "target_perclos": p.target_perclos,
            "nudged_start_s": start_s, "nudge_s": start_s - float(p.window_start_s),
            "start_frame": start, "n_frames": n_frames, "fps": fps,
            "start_openness": float(op[start]), "nudge_tier": tier,
            "perclos80_cut": float(summary["perclos80_mean"]),
            "blink_rate_cut": float(summary["blink_rate_per_min"]),
            "mean_reopening_speed_cut": float(summary["mean_reopening_speed"]),
            "detection_rate_cut": float(summary["detection_rate"]),
            "segment_keyframe_s": key, "rotation_deg": int(float(m.rotation_deg)),
            **vinfo, "crop_x0": box[0], "crop_y0": box[1], "crop_side": box[2],
            "source_relpath": m.relpath, "file": f"{clip_id}.mp4",
            "segment_file": f"segments/{clip_id}.mp4",
        })
        r = rows[-1]
        print(f"  {clip_id}: {p['class']}, window {p.perclos80_mean:.3f} -> cut "
              f"{r['perclos80_cut']:.3f}, nudge {r['nudge_s']:+.2f}s, "
              f"start openness {r['start_openness']:.2f} (tier {tier})")
    pd.DataFrame(rows).to_csv(a.out / "driving_clips.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
