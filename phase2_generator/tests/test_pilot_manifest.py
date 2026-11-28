"""Pilot manifest and provenance: Phase 1 shape, driving fold, is_synthetic, start nudge."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen import pilot_driving, pilot_generate, pilot_manifest as pm


def prov(**kw) -> dict:
    p = {"provenance_id": pilot_generate.provenance_id("syn01", "drv_13_10_t30"),
         "is_synthetic": True, "identity_id": "syn01", "identity_seed": 2,
         "identity_sha1": "a" * 40, "driving_clip_id": "drv_13_10_t30",
         "driving_video_id": "13_10", "driving_subject_id": "13", "driving_fold": 2,
         "driving_class": "drowsy", "driving_sha1": "b" * 40, "generator": "LivePortrait",
         "generator_commit": "9b294b3", "generator_flags_hash": "x"}
    p.update(kw)
    return p


PROBE = {"size_bytes": 1, "rotation_deg": 0, "rotation_method": "none", "stored_width": 1024,
         "stored_height": 1024, "effective_width": 1024, "effective_height": 1024, "fps": 29.0,
         "duration_s": 62.0, "bitrate_kbps": 900.0, "probe_error": ""}


def manifest(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pm.PHASE1_COLUMNS + pm.PROVENANCE_COLUMNS)


def test_row_has_phase1_shape_and_driving_fold():
    r = pm.synthetic_row(prov(), "synthetic/x.mp4", dict(PROBE))
    m = manifest([r])
    pm.check_manifest(m)
    assert r["subject_id"] == "syn01" and r["fold"] == "2" and r["class"] == "drowsy"
    assert r["video_id"] == r["provenance_id"] == "p2pilot-lp-syn01-drv_13_10_t30"
    assert r["driving_subject_id"] == "13" and r["is_synthetic"] is True


def test_phase1_columns_match_the_real_manifest_header():
    header = list(pd.read_csv(pilot_driving.P1 / "data" / "manifest.csv", nrows=0).columns) \
        if (pilot_driving.P1 / "data" / "manifest.csv").exists() else pm.PHASE1_COLUMNS
    assert pm.PHASE1_COLUMNS == header


@pytest.mark.parametrize("field", pm.REQUIRED_PROVENANCE)
def test_missing_provenance_field_raises(field):
    with pytest.raises(pm.ProvenanceError, match=field):
        pm.synthetic_row(prov(**{field: ""}), "synthetic/x.mp4", dict(PROBE))


def test_is_synthetic_must_be_true():
    with pytest.raises(pm.ProvenanceError, match="is_synthetic"):
        pm.synthetic_row(prov(is_synthetic=False), "synthetic/x.mp4", dict(PROBE))


def test_fold_that_is_not_the_driving_fold_is_refused():
    m = manifest([pm.synthetic_row(prov(), "synthetic/x.mp4", dict(PROBE))])
    m.loc[0, "fold"] = "3"
    with pytest.raises(pm.ProvenanceError, match="driving subject's fold"):
        pm.check_manifest(m)


def test_duplicate_provenance_refused():
    r = pm.synthetic_row(prov(), "synthetic/x.mp4", dict(PROBE))
    with pytest.raises(pm.ProvenanceError, match="duplicate"):
        pm.check_manifest(manifest([r, dict(r)]))


def test_unreadable_output_refused():
    r = pm.synthetic_row(prov(), "synthetic/x.mp4", dict(PROBE, probe_error="RuntimeError: x"))
    with pytest.raises(pm.ProvenanceError, match="unreadable"):
        pm.check_manifest(manifest([r]))


# ------------------------------------------------------------- start nudge
def test_nudge_leaves_an_open_start_alone():
    o = np.ones(300)
    assert pilot_driving.nudge_start(o, 30.0, 100, 60) == (100, 1)


def test_nudge_moves_off_a_blink_to_the_nearest_clear_frame():
    o = np.ones(300)
    o[95:106] = 0.05                      # a blink covering the requested start
    i, tier = pilot_driving.nudge_start(o, 30.0, 100, 60)
    assert tier == 1
    # Must be open, with 0.1 s before and 0.4 s after clear of the blink.
    assert o[i] >= 0.85 and (o[i - 3:i + 13] >= 0.7).all()
    assert i in (81, 109)                 # nearest clear frames either side
    assert i == 109


def test_nudge_never_starts_on_an_untracked_frame():
    o = np.ones(300)
    o[100] = np.nan
    assert pilot_driving.nudge_start(o, 30.0, 100, 60)[0] != 100


def test_nudge_raises_rather_than_starting_closed():
    with pytest.raises(ValueError, match="no open-eye start in any tier"):
        pilot_driving.nudge_start(np.full(2000, 0.1), 30.0, 500, 60)


def test_generation_is_capped_at_32():
    assert pilot_generate.MAX_CLIPS == 32


def test_tier_two_used_only_when_tier_one_finds_nothing():
    o = np.full(1000, 0.78)               # never reaches 0.85, always above 0.75
    i, tier = pilot_driving.nudge_start(o, 30.0, 300, 60)
    assert (i, tier) == (300, 2)
    o[600] = 0.9                          # a tier-1 frame 10 s away beats tier 2 here
    o[597:613] = 0.9
    assert pilot_driving.nudge_start(o, 30.0, 300, 60) == (597, 1)  # hold 0.70 accepts the 0.78 around it
