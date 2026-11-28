"""Stage 4 of the Phase 2 pilot: every chosen identity against every driving clip (4 x 8 = 32).

LivePortrait runs with its defaults except as listed in LP_FLAGS. The driving clips are already
upright 512x512 face crops made through phase1 `video.OrientedVideo` (gen.pilot_driving), so
LivePortrait's own driving cropper and video loader rotation handling are not relied on.

One subprocess per pair, cwd in the LivePortrait checkout (its package is also named `src`).
Each output gets a provenance JSON next to it; gen.pilot_manifest turns those into the manifest.

Refuses to generate more than the 32 pilot clips.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m gen.pilot_generate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LP = HERE / "external" / "LivePortrait"
IDENT = HERE / "pilot" / "identities"
DRIVE = HERE / "pilot" / "driving"
OUT = HERE / "pilot" / "synthetic"
MAX_CLIPS = 32
LP_FLAGS = ["--flag-use-half-precision",          # fp16; V100 has no bf16
            "--no-flag-crop-driving-video",       # input is already a face crop
            "--flag-relative-motion", "--flag-stitching", "--flag-pasteback",
            "--no-flag-eye-retargeting", "--no-flag-lip-retargeting",
            "--animation-region", "all", "--driving-option", "expression-friendly",
            "--driving-multiplier", "1.0", "--device-id", "0"]


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_id(identity_id: str, driving_clip_id: str) -> str:
    return f"p2pilot-lp-{identity_id}-{driving_clip_id}"


def lp_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=LP, capture_output=True,
                          text=True, check=True).stdout.strip()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", default=None, help="provenance_ids to (re)run")
    a = ap.parse_args(argv)

    import torch
    idents = read_csv(IDENT / "chosen.csv")
    drives = read_csv(DRIVE / "driving_clips.csv")
    pairs = [(i, d) for i in idents for d in drives]
    if len(pairs) > MAX_CLIPS:
        raise SystemExit(f"{len(pairs)} pairs requested; the pilot is capped at {MAX_CLIPS}")
    a.out.mkdir(parents=True, exist_ok=True)
    commit = lp_commit()
    flags_hash = hashlib.sha1(" ".join(LP_FLAGS).encode()).hexdigest()[:10]

    for ident, drv in pairs:
        pid = provenance_id(ident["identity_id"], drv["driving_clip_id"])
        if a.only is not None and pid not in a.only:
            continue
        final = a.out / f"{pid}.mp4"
        if final.exists() and a.only is None:
            print(f"  skip {pid} (exists)")
            continue
        src_img = IDENT / ident["file"]
        drv_mp4 = DRIVE / drv["file"]
        tmp = a.out / "_lp_tmp" / pid
        shutil.rmtree(tmp, ignore_errors=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "inference.py", "-s", str(src_img), "-d", str(drv_mp4),
                            "-o", str(tmp), *LP_FLAGS], cwd=LP, capture_output=True, text=True)
        (a.out / f"{pid}.log").write_text(r.stdout + "\n" + r.stderr)
        made = [p for p in tmp.glob("*.mp4") if not p.stem.endswith("_concat")]
        if r.returncode != 0 or len(made) != 1:
            print(f"  FAILED {pid}: exit {r.returncode}, outputs {[p.name for p in made]}")
            continue
        shutil.move(str(made[0]), final)
        shutil.rmtree(tmp, ignore_errors=True)
        prov = {
            "provenance_id": pid, "is_synthetic": True, "file": final.name,
            "identity_id": ident["identity_id"], "identity_file": ident["file"],
            "identity_seed": int(ident["seed"]), "identity_sha1": sha1(src_img),
            "driving_clip_id": drv["driving_clip_id"], "driving_video_id": drv["video_id"],
            "driving_subject_id": drv["subject_id"], "driving_fold": int(drv["fold"]),
            "driving_class": drv["class"], "driving_sha1": sha1(drv_mp4),
            "generator": "LivePortrait", "generator_commit": commit,
            "generator_flags": LP_FLAGS, "generator_flags_hash": flags_hash,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "seconds": round(time.time() - t0, 1),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (a.out / f"{pid}.json").write_text(json.dumps(prov, indent=2))
        print(f"  {pid}  {prov['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
