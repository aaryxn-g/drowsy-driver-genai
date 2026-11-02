# Claude Code Prompts Library

Paste-ready prompts for `claude` run inside `C:\Aaryan\College_Stuff\Gen AI Project`.

Claude Code reads `CLAUDE.md` automatically, so it already knows the repo layout, the
logging conventions and the dataset decisions. The prompts below only add what it cannot
infer.

**Last rewritten 2026-11-02.** Phase 1.1 is now written and tested, so the old "build the
ingestion pipeline" prompt is gone. Start at 1.1-A.

House rules that apply to every prompt here, worth repeating to Claude Code when it drifts:

- Never fabricate results. If the data is not there, say so and stop.
- Every run appends to a results CSV with a `run_id`. Never overwrite one.
- Set and log seeds for numpy, torch and random.
- Any claim of the form "we chose X because" must point at a file or chart.

---

## 1.1-A Verify the pipeline on one real video

Run this the moment the first fold finishes downloading. It is the only check the test
suite cannot do for you.

```
Read phase1_baseline/README.md and phase1_baseline/src/selfcheck.py.

I have just downloaded the first fold of UTA-RLDD to <PATH>. Do this in order:

1. Create the conda env from phase1_baseline/requirements.txt (python 3.11) and run
   `python -m pytest tests -q` from inside phase1_baseline. I expect 120 passed. If any
   fail, stop and show me the failure before doing anything else.
2. Run `python -m src.manifest --root "<PATH>" --out data/manifest.csv` and show me the
   summary it prints plus the first 5 rows. Tell me whether the discovered structure
   matches what manifest.py expects, and show me data/manifest_unclassified.csv if it
   was created.
3. Pick one video and run `python -m src.selfcheck --video "<one video path>"`.
4. Open data/selfcheck_ear.png and describe what you see: does each shaded band line up
   with a visible dip in the EAR trace above it? Is the detection rate above 90%? Is the
   blink rate in the 10 to 30 per minute range?

Do not run the full extraction yet. Report back first.
```

## 1.1-B Time the extraction before committing to it

```
Using the manifest at data/manifest.csv, run the extractor over just 3 videos with
`--limit 3` and report the wall-clock seconds per video from data/frames/_extraction_log.csv.

Then work out, and show me the arithmetic for:
- projected hours for all 180 videos on this machine, single process
- the same if I run one process per fold in parallel
- how much disk the parquet output will take in total

Tell me whether this is an overnight job or a multi-day job. Do not start the full run.
```

## 1.1-C Fold-by-fold extraction so storage stays bounded

This is the one to use if disk space is tight. See `phase0_scoping/data_budget.md`.

```
Write phase1_baseline/src/run_fold.py, a helper that processes one fold end to end so I
never need all 111 GB of UTA-RLDD on disk at once.

For a given fold number it should:
1. Check the fold's videos are present under the dataset root; exit with a clear message
   if not.
2. Run the manifest scan restricted to that fold.
3. Run the extraction for that fold's videos only.
4. Print exactly which video files are now safe to delete, and their total size. DO NOT
   delete anything itself. I will delete manually after I have checked the output.
5. Verify every expected parquet file exists and report any video that failed.

Add --dry-run. Add tests in tests/test_run_fold.py covering: fold not present, partial
extraction, and the happy path, using the StubLandmarker pattern already in
tests/test_pipeline.py. Run the full suite when done.
```

## 1.1-D Parallel extraction across cores

Only after 1.1-B says the single-process run is too slow.

```
MediaPipe does not parallelise inside one process, so add phase1_baseline/src/run_parallel.py
that shards the manifest across N worker processes using multiprocessing, one FaceLandmarker
per worker.

Constraints:
- Shard by subject, never by frame, so a subject's videos stay on one worker.
- Each worker writes its own parquet files and appends to its own log; merge the logs at
  the end into data/frames/_extraction_log.csv.
- Must stay resumable: a worker skips videos whose parquet already exists.
- Default N to os.cpu_count() - 1, capped at 6, because MediaPipe is memory-hungry.

Add a test that two workers over four videos produce the same parquet content as one
worker over the same four. Run the full suite.
```

## 1.1-E The subgroup annotation tool

This is on the critical path and does not need the full dataset, only one reference frame
per subject. Build it while the download runs.

```
Read phase1_baseline/data/subgroup_annotation_protocol.md and
phase1_baseline/data/subgroup_annotation_schema.csv.

Build phase1_baseline/src/subgroups.py with two entry points.

A) `python -m src.subgroups extract-refs --manifest data/manifest.csv --root <ROOT>`
   For each subject, pull a few candidate reference frames from their ALERT video:
   - only frames where the face was detected and |yaw| < 10 degrees, so lighting on the
     cheek is comparable across subjects
   - prefer frames near the median overall frame brightness for that video, to avoid
     over- and under-exposed frames
   - save 3 candidates per subject to data/refs/<subject_id>_{0,1,2}.png

B) `python -m src.subgroups ita --refs data/refs --out data/ita.csv`
   Compute Individual Typology Angle per the protocol:
   - sample forehead and cheek patches using the MediaPipe landmark indices, not fixed
     pixel boxes
   - convert sRGB to CIELAB properly (D65, with the sRGB gamma decode; do not use a naive
     linear conversion)
   - ITA = atan((L* - 50) / b*) * 180 / pi, bucketed into the protocol's six bands
   - report per subject: ITA per candidate frame, the median, the spread across the three
     candidates, and an `ita_reliable` boolean that is False when the spread exceeds 10
     degrees or more than 2% of patch pixels are clipped (any channel at 0 or 255),
     with ITA taken as the median over the non-clipped pixels

Write tests with synthetic patches of known L*a*b* values where the expected ITA is
computed by hand, including one clipped patch that must come back unreliable.

Do not guess any subject's age, gender presentation or glasses. Those are human
judgements; this tool only prepares the frames and computes ITA. Emit a pre-filled CSV
for me and a second annotator to complete by hand.
```

## 1.1-F Class and subgroup balance table

```
Once data/video_features.csv and the adjudicated subgroup labels exist, write
phase1_baseline/src/balance_report.py producing phase1_baseline/results/balance_table.md.

It must show, as counts and percentages:
- videos and subjects per class
- subjects per ITA band, age bracket, gender presentation, glasses
- the cross-tabulation of class against each subgroup attribute
- mean face-detection rate per subgroup, with a warning line for any subgroup whose mean
  detection rate is more than 5 points below the overall mean

That last one matters: if tracking fails more often for one subgroup, that is a separate
finding from the detector being inaccurate, and it has to surface before any accuracy
number is interpreted. Flag any subgroup with fewer than 5 subjects as too small for a
reliable gap estimate, and say so in the table rather than in a footnote.
```

---

## 1.2 Architecture benchmark

```
Read phase1_baseline/README.md, src/blinks.py and data/video_features.csv.

Build the Phase 1.2 benchmark harness in phase1_baseline/src/benchmark.py. Compare these
on the 3-class task (alert / low_vigilant / drowsy):

  A. Logistic regression on the per-video features. This is the floor.
  B. Gradient boosting on the same features.
  C. A small temporal model (TCN or BiLSTM) over the per-frame sequences from
     data/frames/, windowed to 60 s with 50% overlap.
  D. Majority-class and always-drowsy baselines. These are not optional. Every
     number needs a trivial comparison beside it.

Non-negotiable evaluation rules:
- Use the five OFFICIAL UTA-RLDD folds. Never a random split. A subject appears in exactly
  one fold.
- Report mean and standard deviation across folds, never a single split.
- Minimum 3 seeds per configuration.
- Log architecture, seed, fold, overall macro F1, per-class F1 and inference latency per
  frame to phase1_baseline/results/architecture_benchmark.csv, appended with a run_id.

Then write results/architecture_choice.md justifying the selection from the CSV. The
published baseline to beat is 65.2% video accuracy (Ghoddoosian et al., CVPRW 2019) on
these same folds. If we come in under that, say so plainly rather than finding a framing
that sounds better.
```

## 1.3 Subgroup fairness audit

```
Using the selected Phase 1.2 model and the adjudicated subgroup labels, write
phase1_baseline/src/fairness_audit.py producing results/subgroup_audit.md.

Compute per subgroup (ITA band, age bracket, gender presentation, glasses):
- F1, precision, recall
- false-positive rate, which is the alert-driver-wrongly-warned rate
- bootstrap 95% confidence intervals, 1000 resamples, resampling SUBJECTS not videos,
  because videos from one subject are not independent

Then the gap metrics: max-minus-min subgroup F1, standard deviation across subgroups, and
the equalized-odds gap.

Two things I want handled honestly rather than smoothed over:
1. Several subgroups have very few subjects. For any subgroup with fewer than 5 subjects,
   report the interval but label the point estimate unreliable, and keep it out of the
   headline gap number with a note saying why.
2. Report face-detection rate per subgroup alongside accuracy. A model that was never
   given a face is failing differently from one that saw it and got it wrong.

End the report with the honest one-paragraph summary of what the audit does and does not
establish given 60 subjects.
```

---

## 2.1 GAN vs diffusion reenactment pilot

```
Read phase0_scoping/dataset_decision.md and the Phase 2 section of PROJECT_TRACKER.md.

Set up phase2_generator/ to run a 50-clip pilot comparing:
  A. FOMM-style GAN reenactment, pretrained checkpoint, inference only
  B. A video diffusion reenactment, pretrained checkpoint, inference only

Both driven by cue trajectories from phase1_baseline, not by copying source faces.

For each generated clip measure:
- identity preservation: ArcFace cosine similarity between the source identity image and
  sampled generated frames
- cue fidelity: re-run the Phase 1 extractor on the GENERATED clip and compute the error
  between recovered PERCLOS, MAR and head pitch and the template that drove it. This is
  the metric that actually matters and the one most papers skip.
- visual quality: FID against real UTA-RLDD frames, and FVD if feasible
- wall-clock seconds per clip

Write phase2_generator/pilot/pilot_results.csv and a comparison note that picks one
pipeline with reasons tied to those numbers. If cue fidelity is poor for both, say so:
that is a real finding and it changes the project.
```

## 2.2 Cue template library

```
Build phase2_generator/src/cue_templates.py that mines data/closure_events.csv and
data/frames/ for reusable drowsiness cue trajectories.

Extract, per severity level (alert / low_vigilant / drowsy):
- eye openness trajectories around closure events, time-normalised
- MAR trajectories around yawns
- head pitch trajectories around nods

Cluster them so we keep a representative library rather than thousands of near-duplicates.
Store each template with provenance: source subject, source video, start time, class.

Keep the provenance. Phase 6 needs to show no real subject can be recovered from a
generated clip, and that argument is impossible without knowing which real segment drove
which synthetic one.
```

---

## 3 Synthetic dataset generation

```
Using the generator chosen in 2.1 and the templates from 2.2, generate two batches into
phase3_synthetic_dataset/:

  raw/       distribution-matched to UTA-RLDD's existing demographics
  balanced/  deliberately oversampling the groups UTA-RLDD lacks

Every sample carries: is_synthetic=True, provenance_id, source_template_id,
source_identity_id, generator, seed, severity.

Then an automated QC pass that rejects a clip when:
- ArcFace similarity to its source identity falls below threshold (identity drifted)
- cue fidelity error exceeds threshold (it does not actually look drowsy)
- the Phase 1 landmarker cannot track it (useless downstream)

Report the filter rate per subgroup. If generation quality is worse for the groups we are
trying to help, the whole approach is in trouble and we need to know early, not in Phase 4.
```

## 4 Retraining sweep, the centrepiece

```
Build phase4_retraining_sweep/src/sweep.py.

Grid: ratios 100:0, 90:10, 75:25, 50:50, 25:75, 0:100 real:synthetic
    x 3 seeds
    x 2 tracks (untargeted synthetic, targeted synthetic)
= 36 runs.

Hard rules:
- The test set is REAL held-out subjects at every single ratio. Synthetic data never
  appears in test. Assert this in code, do not just intend it.
- Subject-level separation: assert no subject_id appears in both train and test. Fail loudly.
- Log every run to results/sweep_results.csv with run_id, ratio, seed, track, overall F1,
  per-subgroup F1, gap metrics.

Produce two charts: overall F1 against ratio, and subgroup gap against ratio, both with
error bars across seeds.

Then results/operating_point.md selecting a ratio. If no ratio improves the gap, write
that conclusion. A negative result here is publishable and pretending otherwise would make
the whole project worthless.
```

## 5 Stress testing

```
Build phase5_stress_test/ evaluating the Phase 4 operating-point model against:
- night / infrared conditions (NTHU-DDD if the licence arrives, otherwise DROZY or NITYMED)
- glasses and occlusion, using the UTA-RLDD glasses subset and any NTHU glasses scenarios
- cross-dataset generalisation: train on one dataset family, test on another

Report subgroup gaps under each condition, not just overall accuracy. The expectation is
that gaps widen under stress. Confirm or refute it with numbers.
```

## 6 Responsible-AI layer

```
Build phase6_responsible_ai/:

6.2 privacy: a membership-inference check. Can an attacker tell whether a specific FFHQ
    identity was in the generator's training set, given only generated outputs? Implement
    a concrete attack and report its AUC. An AUC near 0.5 is the result we want.
6.3 false alarms: per-subgroup false-positive rate at the chosen operating point,
    expressed as alerts per hour of alert driving, which is the unit a driver would feel.
6.5 datasheet: complete a Datasheet for Datasets for the released synthetic set. The
    licence is forced to CC BY-NC-SA 4.0 by FFHQ. State that explicitly.

For 6.4 limitations, be specific and quantitative. "60 subjects, 9 of them women, no
subjects of African descent, so the skin-tone gap is estimated from bands with as few as
N subjects" is a real limitation. "Results may not generalise" is filler.
```

## 7 Consolidation

```
Run the ablations: generator choice, targeted against untargeted, architecture choice.
Then a reproducibility pass: fixed seeds, frozen splits, one clean end-to-end rerun from
raw video to final numbers, verifying the headline figures reproduce.

Then assemble the final report. Every number in it must trace to a results CSV or chart in
the repo. Produce a table mapping each claim in the report to its evidence file. Anything
that cannot be traced gets cut.
```

---

## Using these well

**Give Claude Code one prompt at a time.** These are each a session's work. Pasting three
at once gets you a shallow attempt at all three.

**Make it show you the numbers before it writes the conclusion.** The failure mode on a
project like this is a confident write-up built on a bug upstream.

**When it says a run succeeded, ask to see the CSV.** Especially for anything in Phase 4.

**Push back when a number looks too good.** 97% on this task means a leak, almost always a
subject appearing in both train and test. That specific bug is what the whole evaluation
protocol exists to prevent, and it is easy to reintroduce by accident.
