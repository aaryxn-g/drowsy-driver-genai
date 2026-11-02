"""Prepare subgroup annotation: reference frames, ITA, and a sheet for humans.

    python -m src.subgroups extract-refs --manifest data/manifest.csv --root <ROOT>
    python -m src.subgroups ita --refs data/refs --out data/ita.csv

The method is in data/subgroup_annotation_protocol.md. This tool only measures
skin tone and prepares frames. Age bracket, gender presentation and glasses are
human judgements, so their columns are always written blank.

Every subject in the manifest gets a row in every output, including subjects
with no usable frame. They are flagged, never dropped: the subgroups that are
hardest to measure are the ones the project exists to study, and a tracking
failure is itself a finding.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import geometry
from .landmarks import FaceLandmarker

# ---------------------------------------------------------------- selection
N_CANDIDATES = 3
MAX_ABS_YAW_DEG = 10.0
SAMPLE_EVERY_S = 1.0
# Candidates from adjacent frames would agree trivially and make the spread
# check meaningless, so they have to be separated in time.
MIN_GAP_S = 15.0

# ---------------------------------------------------------------- skin patches
# MediaPipe face mesh indices. The cheek lists are mirror pairs of each other.
# The forehead patch stops well short of the hairline (landmark 10) to avoid hair.
FOREHEAD = (108, 151, 337, 336, 9, 107)
RIGHT_CHEEK = (117, 118, 101, 36, 205, 187, 123)   # subject's right, image left
LEFT_CHEEK = (346, 347, 330, 266, 425, 411, 352)   # subject's left, image right
SKIN_PATCHES = (FOREHEAD, RIGHT_CHEEK, LEFT_CHEEK)
MIN_PATCH_PIXELS = 25

# ---------------------------------------------------------------- reliability
MAX_ITA_SPREAD_DEG = 10.0
# Compressed video puts a few stray pixels at 0 or 255 in most frames, so a single
# clipped pixel says nothing. Above this fraction the frame counts as clipped.
MAX_CLIPPED_FRACTION = 0.02

# (exclusive lower bound, label), from the protocol. Below the last bound is Dark.
ITA_BANDS = (
    (55.0, "Very light"),
    (41.0, "Light"),
    (28.0, "Intermediate"),
    (10.0, "Tan"),
    (-30.0, "Brown"),
)
ITA_DARKEST_BAND = "Dark"

# ---------------------------------------------------------------- colour
# IEC 61966-2-1 linear sRGB to XYZ, with the matching D65 reference white.
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
D65_WHITE = np.array([0.95047, 1.0, 1.08883])

# ---------------------------------------------------------------- outputs
REFS_LOG_NAME = "_refs_log.csv"
REF_NAME_RE = re.compile(r"^(?P<subject>.+)_(?P<index>\d+)\.png$")
DATASET_SOURCE = "UTA-RLDD"
SECOND_ANNOTATOR_FRACTION = 0.20

# Must match data/subgroup_annotation_schema.csv, column for column.
SCHEMA_COLUMNS = [
    "subject_id", "dataset_source", "clip_ids", "skin_tone_ita_category",
    "ita_value_deg", "age_bracket", "gender_presentation", "glasses",
    "annotator_1", "annotator_2", "agreement_flag", "notes",
]
# The attributes that get Cohen's kappa. Skin tone is computed, so it is not one of
# them: its only human input is the blind spot-check by the second annotator, in its
# own column so that it cannot be mistaken for inter-annotator reliability.
HUMAN_ATTRIBUTES = ["age_bracket", "gender_presentation", "glasses"]
ANNOTATOR_1_COLUMNS = [f"{a}_a1" for a in HUMAN_ATTRIBUTES]
ANNOTATOR_2_COLUMNS = [f"{a}_a2" for a in HUMAN_ATTRIBUTES] + ["skin_tone_spot_check"]


# ==================================================================== ITA maths

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert 8-bit sRGB to CIELAB (D65). Input and output are shaped (..., 3)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (linear @ SRGB_TO_XYZ.T) / D65_WHITE

    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3.0 * delta ** 2) + 4.0 / 29.0)
    return np.stack([
        116.0 * f[..., 1] - 16.0,
        500.0 * (f[..., 0] - f[..., 1]),
        200.0 * (f[..., 1] - f[..., 2]),
    ], axis=-1)


def ita_degrees(l_star: float, b_star: float) -> float:
    """ITA = atan((L* - 50) / b*) in degrees, NaN when b* <= 0.

    Skin has positive b*. At b* <= 0 the formula still returns a number, and a
    bright blue-cast or grey patch would land in a dark band. NaN keeps that
    from passing as a skin-tone reading.
    """
    if not b_star > 0:
        return float("nan")
    return math.degrees(math.atan((l_star - 50.0) / b_star))


def ita_category(ita: float) -> str:
    """Bucket an ITA into the protocol's six bands. Blank for NaN."""
    if math.isnan(ita):
        return ""
    for lower, label in ITA_BANDS:
        if ita > lower:
            return label
    return ITA_DARKEST_BAND


# ================================================================ measuring ITA

@dataclass(frozen=True)
class FrameMeasure:
    """One candidate frame's skin measurement. `failure` is set if there is none."""
    ita: float = float("nan")
    l_star: float = float("nan")
    b_star: float = float("nan")
    clipped: bool = False
    clipped_fraction: float = float("nan")
    failure: str = ""


def patch_mask(shape: tuple[int, ...], pts: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of the convex hull of the landmarks, clipped to the frame."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    polygon = np.round(pts[list(indices), :2]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 1)
    return mask.astype(bool)


def measure_frame(frame_bgr: np.ndarray, pts: np.ndarray | None) -> FrameMeasure:
    """Measure ITA on the pooled, non-clipped forehead and cheek pixels of one frame.

    Medians, not means, so a stray highlight or a glasses rim inside the patch
    does not shift the value.
    """
    if pts is None:
        return FrameMeasure(failure="no_face_in_reference")

    masks = [patch_mask(frame_bgr.shape, pts, p) for p in SKIN_PATCHES]
    if any(int(m.sum()) < MIN_PATCH_PIXELS for m in masks):
        return FrameMeasure(failure="skin_patch_too_small")

    union = np.logical_or.reduce(masks)
    pixels_rgb = frame_bgr[union][:, ::-1]
    clipped_pixels = np.any((pixels_rgb == 0) | (pixels_rgb == 255), axis=1)
    clipped_fraction = float(clipped_pixels.mean())
    clipped = clipped_fraction > MAX_CLIPPED_FRACTION
    if clipped_pixels.all():
        return FrameMeasure(clipped=clipped, clipped_fraction=clipped_fraction,
                            failure="all_patch_pixels_clipped")

    lab = srgb_to_lab(pixels_rgb[~clipped_pixels])
    l_star = float(np.median(lab[:, 0]))
    b_star = float(np.median(lab[:, 2]))
    ita = ita_degrees(l_star, b_star)
    return FrameMeasure(
        ita=ita,
        l_star=l_star,
        b_star=b_star,
        clipped=clipped,
        clipped_fraction=clipped_fraction,
        failure="non_positive_b_star" if math.isnan(ita) else "",
    )


def summarise_subject(subject_id: str, measures: list[FrameMeasure]) -> dict:
    """Combine a subject's candidates into one row of ita.csv."""
    values = [m.ita for m in measures if not math.isnan(m.ita)]
    median = float(np.median(values)) if values else float("nan")
    spread = float(max(values) - min(values)) if len(values) >= 2 else float("nan")

    reasons: list[str] = [m.failure for m in measures if m.failure]
    if len(values) < N_CANDIDATES:
        reasons.append("fewer_than_3_candidates")
    if spread > MAX_ITA_SPREAD_DEG:
        reasons.append("spread_exceeds_10_deg")
    if any(m.clipped for m in measures):
        reasons.append("clipped_patch")

    row = {
        "subject_id": subject_id,
        "n_candidates": len(values),
        "ita_median": median,
        "ita_spread": spread,
        "ita_category": ita_category(median),
        "ita_reliable": not reasons,
        "ita_unreliable_reason": ";".join(dict.fromkeys(reasons)),
    }
    for i in range(N_CANDIDATES):
        m = measures[i] if i < len(measures) else FrameMeasure()
        row[f"ita_{i}"] = m.ita
        row[f"l_star_{i}"] = m.l_star
        row[f"b_star_{i}"] = m.b_star
        row[f"clipped_fraction_{i}"] = m.clipped_fraction
    return row


# =========================================================== choosing reference frames

def scan_video(
    path: Path,
    landmarker,
    sample_every_s: float = SAMPLE_EVERY_S,
    max_width: int = 640,
) -> tuple[pd.DataFrame, float]:
    """Record brightness, detection and yaw on evenly spaced frames, in time order.

    The landmarker tracks through a video and needs rising timestamps, which is
    why frames are scanned in order and ranked afterwards.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 or fps > 240:
        fps = 30.0  # same fallback as extract.py: some phone files report nonsense
    step = max(1, int(round(sample_every_s * fps)))

    rows: list[dict] = []
    idx = 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(round(h * max_width / w))),
                                   interpolation=cv2.INTER_AREA)
            fh, fw = frame.shape[:2]
            pts = landmarker(frame, int(round(1000.0 * idx / fps)))
            yaw = geometry.head_pose(pts, fw, fh)[0] if pts is not None else float("nan")
            rows.append({
                "frame": idx,
                "t_s": idx / fps,
                "brightness": float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()),
                "detected": pts is not None,
                "yaw": yaw,
            })
        idx += 1
    cap.release()
    return pd.DataFrame(rows, columns=["frame", "t_s", "brightness", "detected", "yaw"]), fps


def select_candidates(
    scan: pd.DataFrame,
    n: int = N_CANDIDATES,
    max_abs_yaw: float = MAX_ABS_YAW_DEG,
    min_gap_s: float = MIN_GAP_S,
) -> pd.DataFrame:
    """Pick up to n frontal, detected frames nearest the video's median brightness.

    The median is over every scanned frame, because it describes the video's
    overall exposure and not just the frames that happened to be frontal.
    """
    if scan.empty:
        return scan
    median = scan["brightness"].median()
    eligible = scan[scan["detected"] & (scan["yaw"].abs() < max_abs_yaw)].copy()
    eligible["distance"] = (eligible["brightness"] - median).abs()
    eligible = eligible.sort_values(["distance", "frame"])

    chosen: list[pd.Series] = []
    for _, row in eligible.iterrows():
        if all(abs(row["t_s"] - c["t_s"]) >= min_gap_s for c in chosen):
            chosen.append(row)
            if len(chosen) == n:
                break
    return pd.DataFrame(chosen, columns=eligible.columns)


def save_frames(path: Path, frames: dict[int, Path]) -> None:
    """Write the given frame numbers from a video as lossless, full-resolution PNGs."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    remaining = dict(frames)
    idx = 0
    while remaining and cap.grab():
        if idx in remaining:
            ok, frame = cap.retrieve()
            if not ok:
                break
            dest = remaining.pop(idx)
            tmp = dest.with_name(dest.stem + ".tmp.png")
            if not cv2.imwrite(str(tmp), frame):
                raise RuntimeError(f"could not write {tmp}")
            tmp.replace(dest)
        idx += 1
    cap.release()
    if remaining:
        raise RuntimeError(f"{path}: frames not readable on second pass: {sorted(remaining)}")


def _ref_paths(refs: Path, subject_id: str) -> list[Path]:
    found = []
    for p in refs.glob("*.png"):
        m = REF_NAME_RE.match(p.name)
        if m and m["subject"] == subject_id:
            found.append((int(m["index"]), p))
    return [p for _, p in sorted(found)]


def _write_log(log: dict[str, dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(log.values()).to_csv(tmp, index=False)
    tmp.replace(path)


def extract_refs(
    manifest: pd.DataFrame,
    root: Path,
    out: Path,
    model: Path | None = None,
    sample_every_s: float = SAMPLE_EVERY_S,
    min_gap_s: float = MIN_GAP_S,
    max_width: int = 640,
    overwrite: bool = False,
) -> dict[str, dict]:
    """Pull candidate frames per subject. Resumable through the log in `out`."""
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / REFS_LOG_NAME
    log: dict[str, dict] = {}
    if log_path.exists() and not overwrite:
        previous = pd.read_csv(log_path, dtype=str, keep_default_na=False)
        log = {r["subject_id"]: r for r in previous.to_dict("records")}

    subjects = sorted(manifest["subject_id"].unique())
    for i, subject in enumerate(subjects, 1):
        if subject in log:
            continue
        rows = manifest[manifest["subject_id"] == subject]
        alert = rows[rows["class"] == "alert"]
        entry = {
            "subject_id": subject,
            "clip_ids": ";".join(sorted(rows["video_id"])),
            "alert_video": "",
            "status": "ok",
            "n_candidates": 0,
            "n_scanned": 0,
            "n_detected": 0,
            "n_frontal": 0,
            "median_brightness": "",
            "frames": "",
            "yaws": "",
            "brightnesses": "",
        }
        for stale in _ref_paths(out, subject):
            stale.unlink()

        if len(alert) != 1:
            entry["status"] = "no_alert_video" if alert.empty else "multiple_alert_videos"
        else:
            entry["alert_video"] = alert.iloc[0]["video_id"]
            video = root / alert.iloc[0]["relpath"]
            if not video.exists():
                entry["status"] = "video_missing"
            else:
                try:
                    # A fresh tracker per subject, so no state crosses between people.
                    with FaceLandmarker(model_path=model) as fl:
                        scan, _ = scan_video(video, fl, sample_every_s, max_width)
                    picked = select_candidates(scan, min_gap_s=min_gap_s)
                    save_frames(video, {
                        int(r["frame"]): out / f"{subject}_{k}.png"
                        for k, (_, r) in enumerate(picked.iterrows())
                    })
                except Exception as exc:
                    entry["status"] = f"failed: {type(exc).__name__}: {exc}"
                else:
                    entry.update({
                        "n_candidates": len(picked),
                        "n_scanned": len(scan),
                        "n_detected": int(scan["detected"].sum()) if len(scan) else 0,
                        "n_frontal": int((scan["yaw"].abs() < MAX_ABS_YAW_DEG).sum()),
                        "median_brightness": (f"{scan['brightness'].median():.2f}"
                                              if len(scan) else ""),
                        "frames": ";".join(str(int(f)) for f in picked["frame"]),
                        "yaws": ";".join(f"{y:.2f}" for y in picked["yaw"]),
                        "brightnesses": ";".join(f"{b:.2f}" for b in picked["brightness"]),
                    })
                    if len(picked) < N_CANDIDATES:
                        entry["status"] = "too_few_frontal_frames"

        log[subject] = entry
        _write_log(log, log_path)
        print(f"[{i}/{len(subjects)}] {subject}  {entry['status']}  "
              f"{entry['n_candidates']} candidates")
    return log


# ======================================================================= ITA pass

def compute_ita(refs: Path, model: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ita table, refs log). Log subjects with no frames still get a row."""
    log_path = refs / REFS_LOG_NAME
    log = (pd.read_csv(log_path, dtype=str, keep_default_na=False) if log_path.exists()
           else pd.DataFrame(columns=["subject_id", "clip_ids", "status"]))

    subjects = set(log["subject_id"])
    for p in refs.glob("*.png"):
        m = REF_NAME_RE.match(p.name)
        if m:
            subjects.add(m["subject"])

    rows = []
    for subject in sorted(subjects):
        measures = []
        for ref in _ref_paths(refs, subject):
            frame = cv2.imread(str(ref), cv2.IMREAD_COLOR)
            if frame is None:
                measures.append(FrameMeasure(failure="unreadable_reference"))
                continue
            # A fresh landmarker per image: these are unrelated stills, so
            # tracking must not carry from one to the next.
            with FaceLandmarker(model_path=model) as fl:
                measures.append(measure_frame(frame, fl(frame, 0)))
        rows.append(summarise_subject(subject, measures))
    return pd.DataFrame(rows), log


def _second_annotator_sample(subjects: list[str], seed: int) -> set[str]:
    k = math.ceil(SECOND_ANNOTATOR_FRACTION * len(subjects))
    rng = np.random.default_rng(seed)
    return set(rng.choice(sorted(subjects), size=k, replace=False).tolist())


def build_sheets(ita: pd.DataFrame, log: pd.DataFrame, refs: Path, seed: int = 0
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the master sheet and the second annotator's blind sheet.

    Only measured or factual columns are filled. Each annotator's labels go in
    their own columns because the schema's single age_bracket, gender_presentation
    and glasses columns hold the adjudicated final labels.

    The blind sheet carries no ITA and no first-annotator labels, so the second
    annotator's labels stay independent for the kappa check and the skin-tone
    spot-check really is blind.
    """
    clips = dict(zip(log["subject_id"], log["clip_ids"])) if len(log) else {}
    subjects = ita["subject_id"].tolist()
    sample = _second_annotator_sample(subjects, seed) if subjects else set()

    refs_of = {s: ";".join(p.name for p in _ref_paths(refs, s)) for s in subjects}
    master, blind = [], []
    for r in ita.to_dict("records"):
        s = r["subject_id"]
        row = dict.fromkeys(SCHEMA_COLUMNS + ANNOTATOR_1_COLUMNS + ANNOTATOR_2_COLUMNS, "")
        row.update({
            "subject_id": s,
            "dataset_source": DATASET_SOURCE,
            "clip_ids": clips.get(s, ""),
            "skin_tone_ita_category": r["ita_category"],
            "ita_value_deg": "" if math.isnan(r["ita_median"]) else round(r["ita_median"], 2),
            "ita_reliable": r["ita_reliable"],
            "ita_spread_deg": "" if math.isnan(r["ita_spread"]) else round(r["ita_spread"], 2),
            "ita_unreliable_reason": r["ita_unreliable_reason"],
            "reference_frames": refs_of[s],
            "second_annotator_sample": s in sample,
        })
        master.append(row)
        if s in sample:
            blind.append({"subject_id": s, "reference_frames": refs_of[s], "annotator_2": "",
                          **dict.fromkeys(ANNOTATOR_2_COLUMNS, ""), "notes": ""})

    order = SCHEMA_COLUMNS + [
        "ita_reliable", "ita_spread_deg", "ita_unreliable_reason", "reference_frames",
        "second_annotator_sample",
    ] + ANNOTATOR_1_COLUMNS + ANNOTATOR_2_COLUMNS
    blind_order = ["subject_id", "reference_frames", "annotator_2"] + ANNOTATOR_2_COLUMNS + ["notes"]
    return pd.DataFrame(master, columns=order), pd.DataFrame(blind, columns=blind_order)


# ============================================================================ CLI

def _cmd_extract_refs(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}. Run src.manifest first.", file=sys.stderr)
        return 2
    if not args.root.exists():
        print(f"dataset root not found: {args.root}", file=sys.stderr)
        return 2
    manifest = pd.read_csv(args.manifest, dtype=str)
    log = extract_refs(manifest, args.root, args.out, model=args.model,
                       sample_every_s=args.sample_every_s, min_gap_s=args.min_gap_s,
                       max_width=args.max_width, overwrite=args.overwrite)
    problems = {s: e["status"] for s, e in log.items() if e["status"] != "ok"}
    print(f"\n{len(log)} subjects, {len(log) - len(problems)} with "
          f"{N_CANDIDATES} candidates. Log: {args.out / REFS_LOG_NAME}")
    for subject, status in sorted(problems.items()):
        print(f"  NEEDS ATTENTION {subject}: {status}")
    return 1 if problems else 0


def _cmd_ita(args: argparse.Namespace) -> int:
    if not args.refs.is_dir():
        print(f"reference folder not found: {args.refs}. Run extract-refs first.", file=sys.stderr)
        return 2
    sheet = args.sheet or args.out.with_name("subgroup_annotations_prefilled.csv")
    blind_path = sheet.with_name(sheet.stem + "_annotator2_blind.csv")
    # Checked before the slow pass. These files hold hand-entered labels once
    # annotation starts, and a rerun must not wipe them.
    existing = [p for p in (sheet, blind_path) if p.exists()]
    if existing and not args.overwrite_sheets:
        print(f"refusing to overwrite {[str(p) for p in existing]}: they may hold hand "
              f"labels. Move them or pass --overwrite-sheets.", file=sys.stderr)
        return 2

    ita, log = compute_ita(args.refs, model=args.model)
    if ita.empty:
        print(f"no reference frames or log in {args.refs}; nothing computed.", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ita.to_csv(args.out, index=False)
    master, blind = build_sheets(ita, log, args.refs, seed=args.sample_seed)
    master.to_csv(sheet, index=False)
    blind.to_csv(blind_path, index=False)

    reliable = int(ita["ita_reliable"].sum())
    print(f"{len(ita)} subjects, {reliable} reliable, {len(ita) - reliable} unreliable")
    print(ita["ita_category"].replace("", "(none)").value_counts().to_string())
    print(f"\nITA:             {args.out}\nAnnotation sheet: {sheet}\nBlind sheet "
          f"({len(blind)} subjects, seed {args.sample_seed}): {blind_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("extract-refs", help="save candidate frames from each ALERT video")
    a.add_argument("--manifest", required=True, type=Path)
    a.add_argument("--root", required=True, type=Path)
    a.add_argument("--out", type=Path, default=Path("data/refs"))
    a.add_argument("--model", type=Path, default=None, help="path to face_landmarker.task")
    a.add_argument("--sample-every-s", type=float, default=SAMPLE_EVERY_S)
    a.add_argument("--min-gap-s", type=float, default=MIN_GAP_S,
                   help="minimum time between a subject's candidates")
    a.add_argument("--max-width", type=int, default=640, help="inference width only; "
                   "saved frames stay at full resolution")
    a.add_argument("--overwrite", action="store_true", help="redo subjects already logged")
    a.set_defaults(func=_cmd_extract_refs)

    b = sub.add_parser("ita", help="compute ITA and write the annotation sheets")
    b.add_argument("--refs", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--model", type=Path, default=None, help="path to face_landmarker.task")
    b.add_argument("--sheet", type=Path, default=None,
                   help="default: subgroup_annotations_prefilled.csv beside --out")
    b.add_argument("--sample-seed", type=int, default=0,
                   help="seed for the second annotator's 20%% sample")
    b.add_argument("--overwrite-sheets", action="store_true")
    b.set_defaults(func=_cmd_ita)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
