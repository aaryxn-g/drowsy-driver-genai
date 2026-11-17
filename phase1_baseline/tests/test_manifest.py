"""Manifest scanning: no real file may disappear without being reported.

Subject 46 records in .m4v, which the scanner used to filter out before
classification, so two of their three videos appeared in neither the manifest nor
the unclassified list. Subjects 32 and 49 have their drowsy recording split into
10_1 and 10_2.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import manifest as man

REPO = Path(__file__).resolve().parents[1]


def touch(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["video_id"]: r for r in rows}


class TestExtensions:
    def test_m4v_is_picked_up(self, tmp_path):
        for rel in ("Fold4_part1/46/0.m4v", "Fold4_part1/46/5.m4v",
                    "Fold4_part1/46/10.mov"):
            touch(tmp_path, rel)
        rows, problems, skipped = man.scan(tmp_path)
        assert {r["video_id"] for r in rows} == {"46_0", "46_5", "46_10"}
        assert {r["class"] for r in rows} == {"alert", "low_vigilant", "drowsy"}
        assert problems == [] and skipped == []

    @pytest.mark.parametrize("suffix", [".MOV", ".mov", ".Mp4", ".M4V", ".AVI"])
    def test_suffix_case_does_not_matter(self, tmp_path, suffix):
        touch(tmp_path, f"Fold1_part1/01/0{suffix}")
        rows, _, skipped = man.scan(tmp_path)
        assert len(rows) == 1 and skipped == []

    def test_skipped_extension_is_reported_not_dropped(self, tmp_path):
        touch(tmp_path, "Fold1_part1/01/0.mp4")
        touch(tmp_path, "Fold1_part1/01/5.wmv")
        touch(tmp_path, "Fold1_part1/01/notes")
        rows, problems, skipped = man.scan(tmp_path)
        assert len(rows) == 1 and problems == []
        assert {s["relpath"] for s in skipped} == {"Fold1_part1/01/5.wmv",
                                                   "Fold1_part1/01/notes"}
        assert all(s["reason"] for s in skipped)

    def test_every_file_lands_somewhere(self, tmp_path):
        for rel in ("Fold1_part1/01/0.mp4", "Fold1_part1/01/odd.mp4",
                    "Fold1_part1/01/x.txt", "loose/0.mp4"):
            touch(tmp_path, rel)
        rows, problems, skipped = man.scan(tmp_path)
        n_files = sum(1 for p in tmp_path.rglob("*") if p.is_file())
        assert len(rows) + len(problems) + len(skipped) == n_files

    def test_cli_writes_skipped_csv_and_prints_count(self, tmp_path):
        root = tmp_path / "data"
        touch(root, "Fold1_part1/01/0.mp4")
        touch(root, "Fold1_part1/01/readme.txt")
        out = tmp_path / "out" / "manifest.csv"
        res = subprocess.run(
            [sys.executable, "-m", "src.manifest", "--root", str(root), "--out", str(out)],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
        assert "skipped files     1" in res.stdout
        skipped = pd.read_csv(out.with_name("manifest_skipped.csv"))
        assert list(skipped["relpath"]) == ["Fold1_part1/01/readme.txt"]


class TestSplitRecordings:
    def test_parts_classify_with_their_number(self, tmp_path):
        for stem in ("0", "5", "10_1", "10_2"):
            touch(tmp_path, f"Fold3_part2/49/{stem}.mp4")
        rows, problems, _ = man.scan(tmp_path)
        assert problems == []
        rows = by_id(rows)
        assert (rows["49_10_1"]["class"], rows["49_10_1"]["part"]) == ("drowsy", "1")
        assert (rows["49_10_2"]["class"], rows["49_10_2"]["part"]) == ("drowsy", "2")

    def test_unsplit_video_has_blank_part(self, tmp_path):
        touch(tmp_path, "Fold1_part1/01/10.mp4")
        rows, _, _ = man.scan(tmp_path)
        assert (rows[0]["class"], rows[0]["part"]) == ("drowsy", "")

    def test_split_subject_is_not_reported_missing_a_class(self, tmp_path):
        for stem in ("0", "5", "10_1", "10_2"):
            touch(tmp_path, f"Fold3_part2/32/{stem}.mp4")
        rows, _, _ = man.scan(tmp_path)
        summary = man.summarise(rows)
        assert "videos            4" in summary
        assert "missing a class" not in summary
        assert "split recordings  2 parts" in summary

    @pytest.mark.parametrize("stem", ["mystery", "10_a", "7", "7_1", "_1", "10_1_x"])
    def test_unrecognised_stem_goes_to_unclassified(self, tmp_path, stem):
        touch(tmp_path, f"Fold1_part1/01/{stem}.mp4")
        rows, problems, _ = man.scan(tmp_path)
        assert rows == []
        assert [p["reason"] for p in problems] == ["unrecognised file stem"]

    @pytest.mark.parametrize("stem, expected", [
        ("10", ("10", "")), ("10_1", ("10", "1")), ("0_12", ("0", "12")),
        (" 5 ", ("5", "")),
    ])
    def test_split_stem(self, stem, expected):
        assert man.split_stem(stem) == expected


def test_same_stem_twice_is_refused_not_picked(tmp_path):
    touch(tmp_path, "Fold1_part1/01/0.mp4")
    touch(tmp_path, "Fold1_part1/01/0.MOV")
    rows, problems, _ = man.scan(tmp_path)
    assert rows == []
    assert [p["reason"] for p in problems] == ["duplicate video_id"] * 2


def test_one_subject_in_two_folds_is_refused(tmp_path):
    """Same video_id in two folds would leak an identity across the split."""
    touch(tmp_path, "Fold1_part1/01/0.mp4")
    touch(tmp_path, "Fold2_part1/01/0.mp4")
    rows, problems, _ = man.scan(tmp_path)
    assert rows == []
    assert {p["fold"] for p in problems} == {"1", "2"}
