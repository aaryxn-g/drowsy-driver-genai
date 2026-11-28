"""Subgroup audit: fixed collapse rules, unknown/unsure kept visible, gap exclusions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import audit
from src.detector import CLASSES

LABELS = CLASSES["3class"]


def annotations(n=12) -> pd.DataFrame:
    skin = ["Very light", "Light", "Intermediate", "Tan", "Brown", "Dark"] * (n // 6)
    skin[0] = None  # one subject without an ITA value
    return pd.DataFrame({
        "subject_id": [f"{i:02d}" for i in range(n)],
        "skin_tone_ita_category": skin,
        "age_bracket": (["18-25", "26-35", "36-50", "51+"] * n)[:n - 1] + ["unsure"],
        "gender_presentation": ["male_presenting"] * (n - 2) + ["female_presenting"] * 2,
        "glasses": ["none", "glasses"] * (n // 2),
    })


def votes_for(subjects, correct) -> pd.DataFrame:
    rows = []
    for s in subjects:
        for k, cls in enumerate(LABELS):
            pred = cls if correct(s) else LABELS[(k + 1) % 3]
            for seed in (0, 1):
                rows.append({"video_id": f"{s}_{k}", "subject_id": s, "class": cls,
                             "pred": pred, "seed": seed})
    return pd.DataFrame(rows)


def test_collapse_rules():
    g = audit.collapse_labels(annotations())
    assert g.skin_tone.tolist()[:6] == ["unknown", "Light", "Medium", "Medium", "Dark", "Dark"]
    assert set(g.age) == {"18-25", "26+", "unsure"}
    assert g.age.iloc[-1] == "unsure"


def test_unexpected_value_raises_rather_than_dropping():
    a = annotations()
    a.loc[3, "skin_tone_ita_category"] = "Olive"
    with pytest.raises(audit.AuditInputError, match="Olive"):
        audit.collapse_labels(a)


def test_blank_age_raises():
    a = annotations()
    a.loc[2, "age_bracket"] = ""
    with pytest.raises(audit.AuditInputError, match="blank"):
        audit.collapse_labels(a)


def test_audit_lists_small_and_unknown_groups_but_keeps_them_out_of_the_gap():
    a = annotations()
    groups = audit.collapse_labels(a)
    subjects = groups.subject_id.tolist()
    # Women (2 subjects) are all wrong, so they would dominate the gap if counted.
    women = set(a.subject_id[a.gender_presentation == "female_presenting"])
    votes = votes_for(subjects, lambda s: s not in women and int(s) % 3 != 0)
    videos = votes.drop_duplicates("video_id")[["video_id", "subject_id"]].assign(detection_rate=0.9)
    windows = pd.DataFrame({"subject_id": subjects, "kept": 10, "dropped": 0})
    windows.loc[windows.subject_id == "01", "dropped"] = 5
    t = audit.audit(votes, groups, videos, windows, LABELS, n_boot=200)

    gp = t[t.axis == "gender_presentation"]
    assert set(gp.group) == {"male_presenting", "female_presenting"}
    assert not gp.set_index("group").loc["female_presenting", "in_gap"]
    assert gp.gap_macro_f1.isna().all()  # one eligible group only

    skin = t[t.axis == "skin_tone"].set_index("group")
    assert skin.loc["unknown", "n_subjects"] == 1 and not skin.loc["unknown", "in_gap"]
    assert skin.loc["unknown", "n_videos"] == 3
    eligible = skin[skin.in_gap]
    assert np.isclose(skin.gap_macro_f1.iloc[0], eligible.macro_f1.max() - eligible.macro_f1.min())
    assert skin.loc["Light", "windows_dropped"] == 5
    assert np.isclose(skin.loc["Light", "frac_windows_dropped"], 5 / 35)  # 01, 06, 07
    for _, r in t.iterrows():
        assert r.macro_f1_ci_lo - 1e-9 <= r.macro_f1 <= r.macro_f1_ci_hi + 1e-9

    age = t[t.axis == "age"].set_index("group")
    assert not age.loc["unsure", "in_gap"]


def test_predictions_without_annotation_raise():
    groups = audit.collapse_labels(annotations())
    votes = votes_for(["99"], lambda s: True)
    videos = votes[["video_id", "subject_id"]].assign(detection_rate=1.0)
    windows = pd.DataFrame({"subject_id": ["99"], "kept": 1, "dropped": 0})
    with pytest.raises(audit.AuditInputError, match="99"):
        audit.audit(votes, groups, videos, windows, LABELS, n_boot=10)
