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
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from . import config

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"}
FOLD_RE = re.compile(r"fold[\s_-]*(\d+)", re.IGNORECASE)


def infer_fold(path: Path, root: Path) -> str | None:
    for part in path.relative_to(root).parts:
        m = FOLD_RE.search(part)
        if m:
            return m.group(1)
    return None


def infer_class(path: Path) -> str | None:
    return config.UTARLDD_CLASS_FROM_STEM.get(path.stem.strip())


def scan(root: Path) -> tuple[list[dict], list[dict]]:
    """Return (classified rows, unclassified rows)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    rows: list[dict] = []
    problems: list[dict] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in VIDEO_SUFFIXES:
            continue
        subject = path.parent.name
        fold = infer_fold(path, root)
        label = infer_class(path)
        record = {
            "video_id": f"{subject}_{path.stem}",
            "subject_id": subject,
            "fold": fold or "",
            "class": label or "",
            "relpath": str(path.relative_to(root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
        }
        if label is None or fold is None:
            record["reason"] = (
                "unrecognised file stem" if label is None else "no fold in path"
            )
            problems.append(record)
        else:
            rows.append(record)
    return rows, problems


def summarise(rows: list[dict]) -> str:
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

    incomplete = [s for s in subjects
                  if len({r['class'] for r in rows if r['subject_id'] == s}) < 3]
    if incomplete:
        lines.append(f"subjects missing a class: {len(incomplete)} -> {incomplete[:10]}")
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

    rows, problems = scan(args.root)
    write_csv(rows, args.out)
    print(summarise(rows))
    print(f"\nwrote {len(rows)} rows to {args.out}")

    if problems:
        prob_path = args.out.with_name(args.out.stem + "_unclassified.csv")
        write_csv(problems, prob_path)
        print(f"WARNING: {len(problems)} files could not be classified "
              f"-> {prob_path}")
        print("Inspect that file before continuing; do not assume a label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
