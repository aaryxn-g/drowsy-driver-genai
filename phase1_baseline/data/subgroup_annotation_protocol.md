# Subgroup Annotation Protocol (Phase 1.1)

This is the methodology behind `subgroup_annotation_schema.csv`. Get this right — every
fairness number in the entire project (Phases 1, 4, 5, 6) is computed against these labels.

## Why not self-declared race?
Video subjects in NTHU-DDD/UTA-RLDD have no self-declared demographic labels attached, and
inferring race from video is both technically unreliable and ethically fraught. Instead,
use validated, measurable proxies:

## 1. Skin tone — Individual Typology Angle (ITA)
ITA is a standard, reproducible skin-tone measure used in dermatology and increasingly in
fairness-in-vision research (it's what underlies datasets like Diversity in Faces / Fitzpatrick
tone benchmarks).

**Method:**
1. From a well-lit reference frame per subject, sample a patch of forehead/cheek skin
   (avoid shadows, makeup, screen glare).
2. Convert to CIELAB color space (L*, a*, b*).
3. Compute: `ITA = atan((L* - 50) / b*) * (180 / pi)`
4. Bucket into categories (commonly used bands):
   - Very light: ITA > 55
   - Light: 41 < ITA <= 55
   - Intermediate: 28 < ITA <= 41
   - Tan: 10 < ITA <= 28
   - Brown: -30 < ITA <= 10
   - Dark: ITA <= -30
5. **Caveat to log explicitly in your report:** ITA is sensitive to lighting/camera color
   calibration, which varies across NTHU-DDD (IR-augmented) vs UTA-RLDD (webcam, visible
   light). Compute ITA only from matched-lighting reference frames per subject where
   possible, and flag/exclude frames where lighting makes ITA unreliable (log an
   `ita_reliable` boolean per subject, not just the value).

## 2. Age bracket
Use apparent-age visual estimation into broad brackets (e.g., 18-25 / 26-40 / 41-60 / 60+),
not exact age — exact age is unreliable from video and unnecessary for subgroup-gap analysis.

## 3. Gender presentation
Label as apparent gender presentation (not gender identity, which cannot be inferred from
video) — use categories consistent with prior fairness literature (e.g., "presents
masculine" / "presents feminine") purely as a visual-presentation proxy, and say so
explicitly in the report's limitations section.

## 4. Glasses / occlusion
Binary + type: none / prescription glasses / sunglasses / other occlusion (mask, hand,
phone). This one directly matters for Phase 5's stress test.

## Inter-annotator agreement
- Have a second person (classmate, TA, or you-on-a-different-day with a randomized reveal)
  independently label a **20% random sample** of subjects.
- Compute Cohen's κ per attribute (skin-tone-category, age-bracket, gender-presentation,
  glasses). Report all four κ values in the Phase 1 deliverable.
- If κ < 0.6 on any attribute, tighten the labeling instructions (add example anchor images
  per bucket) and re-annotate the disagreement cases before proceeding — don't compute
  fairness metrics on unreliable subgroup labels.

## Output
Fill `subgroup_annotation_schema.csv` with one row per subject, both annotators' labels, an
`agreement_flag` (match/mismatch per attribute), and adjudicated final labels used downstream.
