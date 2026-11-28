# Phase 1.3 subgroup audit

Last updated 2026-11-28. Source for the Phase 1 deliverable report. Every number here points
at a results file; the rules behind each analysis were committed before it was run (decision
rows dated 2026-11-28 in `PROJECT_TRACKER.md`, commits `843f080` and `e013f8c`).

## Setup

- Predictions: 60 s windows, 3-class HGB, majority vote per video, leave-one-fold-out over the
  5 official UTA-RLDD folds, 5 seeds pooled (source batch `20261125T065741Z-04b301`). No
  retraining for the audit.
- Labels: annotator 1, `data/subgroup_annotations_prefilled.csv` (sha1:0254a281a1f5). The
  second annotator, Cohen's κ and the blind ITA spot-check are **still outstanding**.
- Collapse rules: skin tone Very light + Light → Light, Intermediate + Tan → Medium,
  Brown + Dark → Dark, no ITA value → `unknown`; age 18-25 vs 26+; gender presentation and
  glasses as annotated; `unsure` kept as its own row.
- 95% CIs: 2000 bootstrap resamples of subjects within each group. Groups under 3 subjects
  and `unknown`/`unsure` are listed but kept out of the max-min gap.

`python -m src.audit`, `python -m src.audit --skin-tone-reliable-only`,
`python -m src.audit --ita-covariates`.

## Results, all 60 subjects

Audit batch `20261128T133213Z-dc1b89` in `results/subgroup_audit.csv`.

| Axis | Group | Subjects | Videos | Macro F1 % [95% CI] | Detection % | Windows dropped |
|---|---|---|---|---|---|---|
| Skin tone | Light | 15 | 46 | 65.9 [52.6, 77.7] | 99.4 | 2 / 887 |
| | Medium | 17 | 52 | 54.9 [39.1, 68.7] | 98.7 | 28 / 1012 |
| | Dark | 26 | 78 | 62.7 [54.2, 69.8] | 99.7 | 0 / 1507 |
| | unknown* | 2 | 6 | 66.7 [33.3, 100] | 92.3 | 12 / 131 |
| Age | 18-25 | 15 | 46 | 68.7 [55.2, 80.0] | 97.9 | 35 / 895 |
| | 26+ | 43 | 130 | 59.1 [51.7, 65.8] | 99.5 | 7 / 2526 |
| | unsure* | 2 | 6 | 55.7 [33.3, 78.0] | 99.8 | 0 / 116 |
| Gender presentation | female | 9 | 27 | 64.8 [44.4, 83.8] | 98.7 | 8 / 520 |
| | male | 51 | 155 | 60.8 [53.9, 67.2] | 99.2 | 34 / 3017 |
| Glasses | glasses | 9 | 27 | 64.8 [46.7, 80.4] | 99.7 | 0 / 534 |
| | none | 48 | 146 | 63.1 [56.1, 69.6] | 98.9 | 42 / 2833 |
| | unsure* | 3 | 9 | 23.1 [4.2, 36.8] | 99.9 | 0 / 170 |

\* listed, not in the gap.

Max-min macro F1 gap: skin tone **11.1** [2.5, 30.5] (Light vs Medium), age 9.6 [0.5, 23.5],
gender presentation 3.9 [0.4, 24.3], glasses 1.7 [0.2, 21.7].

**No baseline gap is statistically established at n=60.** Every group CI overlaps every other
on its axis, and a bootstrapped max-min gap is biased upward, since sampling noise alone makes
the maximum larger and the minimum smaller. This is a property of the dataset (60 subjects,
9 women, 0 of African descent), not an analysis choice. It is why the Phase 4 endpoint is a
*paired* change in gap, where the subject-sampling noise is shared by both models and cancels.

### Tracking fairness

Detection rate is at least 97.9% in every gap-eligible group. Window drops for low coverage
are concentrated in individual subjects, not spread across groups. Subject 49 (Medium, 18-25)
lost 27 of 85 windows (32%), and subject 30 (18-25) lost 8 of 62. Medium's and 18-25's higher
drop rates are mostly those two people.

The two `unknown` skin-tone subjects, 26 and 30, are also the worst tracked (92.3% detection,
9.2% of windows dropped). ITA failed on them for `non_positive_b_star` and too few usable
reference frames, which means the same footage that defeats landmark tracking also defeats
skin measurement. Their skin tone is unmeasured exactly where tracking is weakest, so this
audit cannot rule out that tracking failure depends on skin tone.

## Sensitivity: skin tone on ITA-reliable subjects only

Audit batch `20261128T134204Z-be594a`. 32 of 60 subjects have `ita_reliable = True`.

| Group | Subjects | Videos | Macro F1 % [95% CI] | Detection % | Windows dropped |
|---|---|---|---|---|---|
| Light | 7 | 21 | 61.0 [41.2, 80.1] | 99.2 | 1 / 402 |
| Medium | 10 | 31 | 56.5 [37.8, 72.5] | 98.1 | 27 / 609 |
| Dark | 15 | 45 | 63.0 [51.6, 73.1] | 99.8 | 0 / 868 |

Gap 6.5 [2.6, 33.7], best Dark, worst Medium.

On reliable ITA, Medium is still the worst group. The 11.1-point headline, however, shrinks to
6.5 and the best group changes from Light to Dark, because Light loses about 5 points once its
8 unreliable-ITA subjects are removed. **So the size of the skin-tone gap, and which group comes
out on top, depend on ITA measurement quality.** Only "Medium is lowest" holds up in both
versions. Both versions are reported, and neither replaces the other.

## Is ITA measuring lighting?

`results/ita_covariates.csv`, batch `20261128T134206Z-20d012`, 58 subjects with an ITA value.
Covariates come from the subject's alert video, which is where the reference frames are taken.

| ITA against | Spearman ρ | p |
|---|---|---|
| L*, mean over reference frames | 0.93 | 2e-26 |
| bitrate_kbps | −0.05 | 0.71 |
| effective resolution, short side | 0.00 | 0.98 |
| detection_rate, alert video | −0.08 | 0.53 |
| detection_rate, all videos | −0.02 | 0.88 |
| container rotation flag | −0.01 | 0.92 |

ITA does not track video quality, tracking quality or rotation. It is effectively independent of
all of them, so the rotation bug still open in `subgroups.py` does not show up as an ITA bias.

The strong correlation with L* **cannot be used either way**. ITA is defined as
atan((L* − 50) / b*), so it rises with L* by construction. L* combines a subject's skin lightness
with the exposure of the shot, and nothing in this data separates the two. The underexposure
hypothesis is therefore **neither confirmed nor ruled out**. Twelve subjects below −30° in a
dataset with no subjects of African descent remains an unexplained result. Separating the two
would need an exposure reference that does not depend on skin, such as the L* of a fixed
background region in the same frames, or the blind human spot-check.

## What ITA measures

ITA here measures **apparent** skin tone: the colour of the skin as this camera recorded it
under this lighting. For a camera-based detector this is arguably the variable that matters,
because the model only ever sees apparent tone. It is not the same as a subject's actual skin
tone, and the report must not describe it as if it were. "Dark" in this audit means
"recorded dark", which may combine darker skin with a darker shot.

## Open items

- Annotator 2, κ on age, gender presentation and glasses, and the blind ITA spot-check.
- An exposure proxy that does not depend on skin (background L*), to separate lighting from
  tone.
- Std-dev and equalized-odds gap metrics.
- `subgroups.py` should read video through `video.OrientedVideo`. No ITA bias from rotation is
  visible, but the invariant still stands.
