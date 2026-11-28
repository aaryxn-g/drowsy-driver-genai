"""Build the pilot manifests in the Phase 1 manifest shape, so phase1 extract runs unchanged.

`synthetic_manifest.csv`: one row per generated clip. The Phase 1 columns come first, in
Phase 1 order, filled by phase1 `manifest.probe_columns` on the generated file; provenance
columns follow. Semantics:
  * subject_id  = identity_id (the synthetic person on screen);
  * fold        = the DRIVING subject's fold. The clip carries that real subject's motion, so
                  it must never be trained on while that subject is in the test fold;
  * class       = the driving clip's class;
  * relpath     relative to phase2_generator/pilot/.

`driving_manifest.csv`: the eight 512x512 driving crops in the same shape, as a control
(real faces, same crop and re-encode, same extractor).

Runs in the `drowsy` env (Phase 1 code, cv2 5.0):  python -m gen.pilot_manifest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PILOT = HERE / "pilot"
P1 = HERE.parent / "phase1_baseline"
sys.path.insert(0, str(P1))
from src import manifest as p1manifest  # noqa: E402  unchanged Phase 1 code

PHASE1_COLUMNS = ["video_id", "subject_id", "fold", "class", "part", "relpath", "size_bytes",
                  "rotation_deg", "rotation_method", "stored_width", "stored_height",
                  "effective_width", "effective_height", "fps", "duration_s", "bitrate_kbps",
                  "probe_error"]
PROVENANCE_COLUMNS = ["is_synthetic", "provenance_id", "identity_id", "identity_seed",
                      "identity_sha1", "driving_clip_id", "driving_video_id",
                      "driving_subject_id", "driving_fold", "driving_class", "driving_sha1",
                      "generator", "generator_commit", "generator_flags_hash"]
REQUIRED_PROVENANCE = ("provenance_id", "identity_id", "driving_subject_id",
                       "driving_video_id", "driving_fold")


class ProvenanceError(ValueError):
    pass


def synthetic_row(prov: dict, relpath: str, probe: dict) -> dict:
    """One manifest row from a provenance record and the probed file."""
    missing = [k for k in REQUIRED_PROVENANCE if prov.get(k) in (None, "")]
    if missing:
        raise ProvenanceError(f"{prov.get('provenance_id', '?')}: missing {missing}")
    if prov.get("is_synthetic") is not True:
        raise ProvenanceError(f"{prov['provenance_id']}: is_synthetic must be True")
    row = {
        "video_id": prov["provenance_id"], "subject_id": prov["identity_id"],
        "fold": str(prov["driving_fold"]), "class": prov["driving_class"], "part": "",
        "relpath": relpath, "size_bytes": probe.pop("size_bytes", ""), **probe,
    }
    row.update({k: prov.get(k, "") for k in PROVENANCE_COLUMNS})
    row["is_synthetic"] = True
    return row


def check_manifest(m: pd.DataFrame) -> None:
    """The invariants the pilot and Phase 4 rely on."""
    if list(m.columns[:len(PHASE1_COLUMNS)]) != PHASE1_COLUMNS:
        raise ProvenanceError("Phase 1 columns missing or out of order")
    if m.video_id.duplicated().any() or m.provenance_id.duplicated().any():
        raise ProvenanceError("duplicate video_id / provenance_id")
    if not (m.video_id == m.provenance_id).all():
        raise ProvenanceError("video_id must equal provenance_id")
    if not (m.fold.astype(str) == m.driving_fold.astype(str)).all():
        raise ProvenanceError("fold must be the driving subject's fold")
    if not m.is_synthetic.astype(str).str.lower().eq("true").all():
        raise ProvenanceError("every synthetic row must have is_synthetic True")
    if m.probe_error.fillna("").astype(str).str.len().gt(0).any():
        raise ProvenanceError(f"unreadable outputs: {m.loc[m.probe_error.fillna('') != '', 'video_id'].tolist()}")


def main() -> int:
    rows = []
    for jp in sorted((PILOT / "synthetic").glob("*.json")):
        prov = json.loads(jp.read_text())
        path = jp.with_suffix(".mp4")
        probe = p1manifest.probe_columns(path)
        probe["size_bytes"] = path.stat().st_size
        rows.append(synthetic_row(prov, f"synthetic/{path.name}", probe))
    m = pd.DataFrame(rows, columns=PHASE1_COLUMNS + PROVENANCE_COLUMNS)
    check_manifest(m)
    m.to_csv(PILOT / "synthetic_manifest.csv", index=False)

    d = pd.read_csv(PILOT / "driving" / "driving_clips.csv", dtype={"subject_id": str})
    drows = []
    for _, r in d.iterrows():
        path = PILOT / "driving" / r.file
        drows.append({"video_id": r.driving_clip_id, "subject_id": r.subject_id,
                      "fold": str(r.fold), "class": r["class"], "part": "",
                      "relpath": f"driving/{r.file}", "size_bytes": path.stat().st_size,
                      **p1manifest.probe_columns(path), "is_synthetic": False,
                      "provenance_id": f"uta-rldd-{r.video_id}-f{r.start_frame}"})
    pd.DataFrame(drows).to_csv(PILOT / "driving_manifest.csv", index=False)
    print(f"{len(m)} synthetic rows, {len(drows)} driving rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
