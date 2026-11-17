"""Scan a UTA-RLDD extraction and build a manifest of videos.

The expected layout after unzipping the official folds is

    <root>/Fold1_part1/01/0.mp4
    <root>/Fold1_part1/01/5.mp4
    <root>/Fold1_part1/01/10.mp4
    ...

where the folder name is the participant id and the file stem is the Karolinska
label (0 alert, 5 low vigilant, 10 drowsy).

Nothing here assumes that layout is exactly right. The scanner discovers what is
actually on disk and reports anything it could not classify instead of guessing,
because a silently mislabelled video would poison every downstream number.

Each video is also probed for its rotation flag, resolution, frame rate, duration
and bitrate. Resolution is a quality covariate for Phase 1.3. A video that cannot be
probed stays in the manifest with `probe_error` set.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from . import config, video

# Compared lowercased: the dataset already mixes .mov and .MOV. Subject 46 records
# in .m4v, which was once missing here and cost them two of three videos silently.
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
FOLD_RE = re.compile(r"fold[\s_-]*(\d+)", re.IGNORECASE)
# A recording cut in two is saved as 10_1.mp4 and 10_2.mp4 (subjects 32 and 49).
PART_RE = re.compile(r"^(?P<base>.+?)_(?P<part>\d+)$")


def infer_fold(path: Path, root: Path) -> str | None:
    for part in path.relative_to(root).parts:
        m = FOLD_RE.search(part)
        if m:
            return m.group(1)
    return None


def split_stem(stem: str) -> tuple[str, str]:
    """Split "10_2" into ("10", "2"). An unsplit stem comes back with a blank part."""
    stem = stem.strip()
    m = PART_RE.match(stem)
    return (m.group("base"), m.group("part")) if m else (stem, "")


def infer_class(path: Path) -> str | None:
    """Label from the stem, ignoring a trailing _<part>. Never guessed beyond that."""
    base, _ = split_stem(path.stem)
    return config.UTARLDD_CLASS_FROM_STEM.get(base)


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def probe_columns(path: Path) -> dict:
    """The video's quality covariates, blank plus a reason if it cannot be read."""
    blank = {column: "" for column in video.INFO_COLUMNS}
    try:
        return {**video.probe(path).as_row(), "probe_error": ""}
    except Exception as exc:
        return {**blank, "probe_error": f"{type(exc).__name__}: {exc}"}


def scan(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (classified rows, unclassified rows, skipped files).

    Every file under root lands in exactly one of the three. A file skipped on its
    extension is reported, never silently discarded.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    rows: list[dict] = []
    problems: list[dict] = []
    skipped: list[dict] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relpath = str(path.relative_to(root)).replace("\\", "/")
        if not is_video(path):
            skipped.append({
                "relpath": relpath,
                "size_bytes": path.stat().st_size,
                "reason": f"extension {path.suffix or '(none)'!r} not in VIDEO_SUFFIXES",
            })
            continue
        subject = path.parent.name
        fold = infer_fold(path, root)
        label = infer_class(path)
        _, part = split_stem(path.stem)
        record = {
            "video_id": f"{subject}_{path.stem}",
            "subject_id": subject,
            "fold": fold or "",
            "class": label or "",
            "part": part,
            "relpath": relpath,
            "size_bytes": path.stat().st_size,
            **probe_columns(path),
        }
        if label is None or fold is None:
            record["reason"] = (
                "unrecognised file stem" if label is None else "no fold in path"
            )
            problems.append(record)
        else:
            rows.append(record)

    # Two files with one stem in one folder (0.mp4 beside 0.MOV) would share a
    # video_id and so overwrite each other's parquet. Neither is picked for you.
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["video_id"]] = counts.get(r["video_id"], 0) + 1
    for r in [r for r in rows if counts[r["video_id"]] > 1]:
        rows.remove(r)
        problems.append({**r, "reason": "duplicate video_id"})
    return rows, problems, skipped


def summarise(rows: list[dict], skipped: list[dict] | None = None) -> str:
    subjects = sorted({r["subject_id"] for r in rows})
    folds: dict[str, set[str]] = {}
    classes: dict[str, int] = {}
    for r in rows:
        folds.setdefault(r["fold"], set()).add(r["subject_id"])
        classes[r["class"]] = classes.get(r["class"], 0) + 1

    lines = [
        f"videos            {len(rows)}",
        f"subjects          {len(subjects)}",
        f"folds             {len(folds)}",
    ]
    for f in sorted(folds, key=lambda x: (len(x), x)):
        lines.append(f"  fold {f:<12} {len(folds[f])} subjects")
    for c in config.CLASS_ORDER:
        lines.append(f"  {c:<17} {classes.get(c, 0)} videos")

    split = sorted({r["subject_id"] for r in rows if r.get("part")})
    if split:
        lines.append(f"split recordings  {sum(1 for r in rows if r.get('part'))} parts, "
                     f"subjects {split} (kept as separate videos)")

    incomplete = [s for s in subjects
                  if len({r['class'] for r in rows if r['subject_id'] == s}) < 3]
    if incomplete:
        lines.append(f"subjects missing a class: {len(incomplete)} -> {incomplete[:10]}")

    probed = [r for r in rows if not r.get("probe_error")]
    rotated = sum(1 for r in probed if r["rotation_deg"])
    low_res = [r for r in probed
               if video.is_low_resolution(r["effective_width"], r["effective_height"])]
    lines.append(f"rotated videos    {rotated}")
    lines.append(f"low resolution    {len(low_res)} videos, "
                 f"{len({r['subject_id'] for r in low_res})} subjects (flagged, kept)")
    failed = len(rows) - len(probed)
    if failed:
        lines.append(f"could not probe   {failed} videos, see probe_error in the manifest")
    if skipped is not None:
        lines.append(f"skipped files     {len(skipped)} (not a video extension)")
    return "\n".join(lines)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a UTA-RLDD video manifest.")
    ap.add_argument("--root", required=True, type=Path,
                    help="folder containing the extracted Fold* directories")
    ap.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    args = ap.parse_args()

    rows, problems, skipped = scan(args.root)
    write_csv(rows, args.out)
    print(summarise(rows, skipped))
    print(f"\nwrote {len(rows)} rows to {args.out}")

    skip_path = args.out.with_name(args.out.stem + "_skipped.csv")
    if skipped:
        write_csv(skipped, skip_path)
        by_ext: dict[str, int] = {}
        for r in skipped:
            ext = Path(r["relpath"]).suffix.lower() or "(none)"
            by_ext[ext] = by_ext.get(ext, 0) + 1
        print(f"NOTE: {len(skipped)} files skipped on extension "
              f"{dict(sorted(by_ext.items()))} -> {skip_path}")
        print("Check it holds no recordings; a real video here means VIDEO_SUFFIXES is short.")
    elif skip_path.exists():
        skip_path.unlink()  # a stale list from an earlier scan would mislead

    prob_path = args.out.with_name(args.out.stem + "_unclassified.csv")
    if not problems and prob_path.exists():
        prob_path.unlink()
    if problems:
        write_csv(problems, prob_path)
        print(f"WARNING: {len(problems)} files could not be classified "
              f"-> {prob_path}")
        print("Inspect that file before continuing; do not assume a label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
