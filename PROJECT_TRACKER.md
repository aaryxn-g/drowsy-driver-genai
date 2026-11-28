# Project Tracker — Synthetic Drowsy-Driver Data Generation & Fatigue Detection

Owner: Aaryan (BITS Pilani Dubai) · Started: 2026-09-21 · Last updated: 2026-11-28

Legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

This file is the single source of truth for status. Update it every session (you or Claude Code)
so nothing gets lost between sessions. Each phase folder has its own README with finer detail.

---

## Phase 0 — Scoping, literature, dataset access, resource audit
- [~] 0.1 Annotated bibliography (drowsiness detection SOTA / face reenactment / synthetic-fairness) — `phase0_scoping/annotated_bibliography.md`
- [x] 0.2 Final dataset list verified live (27 Sep): UTA-RLDD, NTHU-DDD (official), YawDD, FFHQ, FairFace — `phase0_scoping/dataset_decision.md`
- [x] 0.2 Licences reviewed and logged (FFHQ CC BY-NC-SA 4.0 → synthetic release must match; FairFace CC BY 4.0; NTHU no-redistribution)
- [x] 0.2 **UTA-RLDD downloaded** (direct, no form; 111.3 GB across 10 zips)
- [ ] 0.2 NTHU-DDD licence signed by HoD/lab director and emailed
- [ ] 0.2 YawDD downloaded (IEEE DataPort, free account)
- [ ] 0.2 FFHQ metadata + thumbnails downloaded; FairFace images/labels/model downloaded
- [ ] 0.3 Compute plan decided (local GPU / Colab Pro / university cluster / cloud credits) — `phase0_scoping/compute_plan.md`
- [ ] 0.4 IRB/ethics sign-off requested from department
- [ ] 0.4 Dataset-release plan drafted (datasheet skeleton)
- [ ] Phase 0 exit doc complete — `phase0_scoping/scoping_document.md`

## Phase 1 — Baseline fatigue detector + subgroup audit
- [x] 1.1 Manifest scanner: discovers subjects/folds/classes, refuses to guess unknown files — `phase1_baseline/src/manifest.py`
- [x] 1.1 Landmark extraction wired up (MediaPipe FaceMesh, both solutions and tasks API) — `phase1_baseline/src/landmarks.py`
- [x] 1.1 Per-frame geometry: EAR, MAR, head pose via solvePnP, yaw correction — `phase1_baseline/src/geometry.py`
- [x] 1.1 Blink/PERCLOS layer: per-subject baselines, closure events, rolling PERCLOS — `phase1_baseline/src/blinks.py`
- [x] 1.1 Two-pass CLI (`extract` → `aggregate`), resumable, NaN-safe — `phase1_baseline/src/`
- [x] 1.1 Test suite: 146 tests, geometry vs synthetic projections, ITA vs hand-computed values, pipeline end to end with a stub landmarker
- [x] 1.1 Subgroup annotation tool: reference frames, ITA, pre-filled and blind sheets (no real data run yet) — `phase1_baseline/src/subgroups.py`
- [x] 1.1 Head-pose axis fix: pose model now matches OpenCV's Y-down camera frame, positive pitch = nodding down — `phase1_baseline/src/config.py`
- [x] 1.1 Framing confound surfaced: face position per frame, centred pose, drift features and `pitch_confounded` — `phase1_baseline/src/aggregate.py`
- [x] 1.1 Server env: MediaPipe loads on the tasks backend after `apt-get install libegl1 libgles2`
- [x] 1.1 (2026-11-04) Container rotation verified per video and applied before downscale; rotation, resolution, fps, duration and bitrate recorded in manifest and log; `low_resolution` flag added (flagged, kept) — `phase1_baseline/src/video.py`
- [x] 1.1 (2026-11-04) `--shard i/n` with per-shard logs and `--merge-logs` — `phase1_baseline/src/extract.py`
- [x] 1.1 (2026-11-17) **Bug, blocking:** one landmarker was shared across videos, so MediaPipe's VIDEO mode saw timestamps fall from ~601000 ms back to 0 and raised `Input timestamp must be monotonically increasing` on every video after a shard's first (01_0 ran, 01_10 failed). Now one instance per video, closed after it; the stub landmarker enforces the same contract, plus a real-MediaPipe two-video test that runs where the model is cached — `phase1_baseline/src/extract.py:extract_videos`, `tests/test_extraction_run.py`
- [x] 1.1 (2026-11-17) **Bug:** the manifest dropped real files silently. `.m4v` was missing, so two of subject 46's three videos appeared nowhere; suffixes are now compared lowercased. Every non-video file goes to `data/manifest_skipped.csv` with a count in the summary. `10_1`/`10_2` (subjects 32, 49) classify as drowsy with a new `part` column, kept as separate videos, not concatenated. A `video_id` seen twice is refused to the unclassified list — `phase1_baseline/src/manifest.py`, `tests/test_manifest.py`
- [x] 1.1 (2026-11-17) `scripts/run_extraction.sh`: caps ffmpeg/OpenMP/BLAS at one thread per shard and staggers N shards (default 8). Uncapped, 16 shards exhaust the container's pids.max of 2048 on a 64-core box and die with `Could not open codec h264, error: -11`
- [ ] 1.1 `subgroups.py` reads video through plain `cv2.VideoCapture`: move it to `video.OrientedVideo` before `extract-refs` runs on a build that ignores rotation
- [ ] 1.1 **Self-check run on one real video** (`python -m src.selfcheck --video ...`) ← first thing to do once data lands
- [x] 1.1 Full extraction run over all 182 videos (180 recordings, two of them split in two). **Gate 3 passed:** mean PERCLOS80 rises 0.040 / 0.126 / 0.266 across alert / low vigilant / drowsy — `phase1_baseline/data/video_features.csv`
- [ ] 1.1 Demographic annotation pass (skin tone/ITA, age bracket, gender presentation, glasses) + 2nd annotator κ check on age, gender presentation and glasses + blind ITA spot-check
- [ ] 1.1 Class/subgroup balance table generated
- [ ] 1.2 Architecture benchmark: CNN-hybrid vs YOLOv8-based vs ViT/Swin vs LSTM-temporal — `phase1_baseline/results/architecture_benchmark.csv`
- [ ] 1.2 Baseline architecture selected + justification written
- [x] 1.3 (2026-11-25) Baseline detector on per-video features: HGB + logistic reference, leave-one-fold-out over the 5 official folds with subject disjointness asserted, 5 seeds, subject-level bootstrap CIs. 3-class HGB 53.5 ± 1.5% accuracy [95% CI 48.0, 59.0], below the published 65.2% and the 57.8% human figure; binary alert-vs-drowsy 81.1 ± 0.6% [74.0, 87.9]. Per-seed rows in `results/detector_runs.csv` (gitignored, on the server) — `phase1_baseline/src/detector.py`
- [x] 1.3 (2026-11-25) Window-level features (`aggregate --window-s`, alert-video baseline asserted per row, windows under 70% coverage dropped and counted per subject) and window-level detector with majority-vote video accuracy. Sweep 60/30/15/10 s: 3-class HGB video accuracy 62.9 / 61.1 / 57.4 / 58.4%, window macro F1 53.6 / 50.9 / 48.3 / 47.0%. See decision row — `phase1_baseline/results/detector_window_runs.csv`
- [ ] 1.3 Per-subgroup F1 + bootstrap CIs computed (hook is `detector.subgroup_report`; waits on age_bracket, 5 of 60 filled)
- [ ] 1.3 Gap metrics computed (max-min F1 gap, std dev, equalized-odds gap)
- [ ] 1.3 Detection-rate-by-subgroup analysis (tracking failure is itself a fairness finding)
- [ ] Phase 1 deliverable report done

## Phase 2 — Build the generator
- [ ] 2.1 GAN reenactment pipeline stood up (pretrained, e.g. FOMM-style)
- [ ] 2.1 Diffusion reenactment pipeline stood up (pretrained, e.g. FRVD-style / open I2V backbone)
- [ ] 2.1 50-clip pilot comparison run (identity preservation, motion fidelity, FID/FVD)
- [ ] 2.1 Generator pipeline selected + justification written
- [ ] 2.2 Cue-parameter template library extracted from real data (PERCLOS/yawn/head-pose trajectories × severity level)
- [ ] 2.3 Diversity injection plan implemented (identity sampling weighted to underrepresented subgroups, lighting, angle, occlusion)
- [ ] 2.4 Generator quality validation report (identity/motion-fidelity/FID + failure-mode log)

## Phase 3 — Synthetic dataset construction
- [ ] Untargeted (distribution-matched) synthetic batch generated
- [ ] Targeted (subgroup-balanced) synthetic batch generated
- [ ] Provenance metadata schema defined and applied to every sample
- [ ] Automated QC pass run, filter rate reported
- [ ] Dataset card draft started

## Phase 4 — Retraining sweep
- [ ] Ratio sweep grid defined (100:0, 90:10, 75:25, 50:50, 25:75, 0:100) × 3 seeds × 2 tracks
- [ ] Held-out real-only test set frozen (no subject overlap with train)
- [ ] Sweep executed, results logged to `phase4_retraining_sweep/results/`
- [ ] Accuracy-vs-ratio and gap-vs-ratio charts produced
- [ ] Operating point selected + written justification

## Phase 5 — Stress-testing
- [ ] Night/IR-style eval set assembled
- [ ] Glasses/occlusion eval set assembled
- [ ] Cross-dataset generalization test (train on one dataset family, test on other)
- [ ] Subgroup-gap metrics reported under each stress condition

## Phase 6 — Responsible-AI layer
- [ ] 6.1 Fairness narrative consolidated across Phases 1/4/5
- [ ] 6.2 Identity-leakage / membership-inference check on generator
- [ ] 6.3 False-alarm rate per subgroup reported
- [ ] 6.4 Limitations section drafted
- [ ] 6.5 Full datasheet-for-datasets completed

## Phase 7 — Consolidation, ablations, writeup
- [ ] Ablations run (generator choice, targeted vs untargeted, architecture choice)
- [ ] Reproducibility pass (fixed seeds, frozen splits, one clean end-to-end re-run)
- [ ] Final report written
- [ ] Deliverables packaged (model, dataset, datasheet, charts, code)

---

## Decision Log
(Every "why did we pick X" answer goes here, one line each, with a pointer to the evidence file/chart.)

| Date | Decision | Evidence | Chosen |
|---|---|---|---|
| 2026-09-27 | UTA-RLDD is the primary real dataset (training, eval, fairness audit) | Colour video needed for ITA skin tone; 60 subjects; real (not acted) drowsiness; 5 official folds; verified imbalance 51M/9F, no African-descent subjects | UTA-RLDD |
| 2026-09-27 | NTHU-DDD via official licence only, not Kaggle mirrors | Mirrors are frames only (no subject IDs/video), licence "Unknown", and breach NTHU's no-redistribution term; cited mirror now 404 | Official NTHU-DDD |
| 2026-09-27 | NTHU-DDD used for cue templates + night-IR/glasses stress test, NOT skin-tone analysis | NTHU footage is active-IR (grayscale), so ITA can't be computed | — |
| 2026-09-27 | FFHQ (identity pool) + FairFace (demographic classifier + secondary pool); CelebA dropped | CelebA skews young/light/Western; FairFace built for balance and ships a pretrained classifier | FFHQ + FairFace |
| 2026-09-27 | DMD dropped | IR/depth removed from public release (RGB only); 25 TB; NoDerivatives | — |
| 2026-10-29 | **Eye-openness baseline is per subject, taken from their ALERT video and reused for all three of their videos** | A within-video percentile would rescale drowsiness away: a drowsy clip may never contain a fully open eye, so the pipeline would run clean and produce worthless numbers. Subjects with no usable baseline are reported and skipped, never scored on a guess. See `src/aggregate.py:build_baselines` and `tests/test_pipeline.py::test_baseline_is_shared_across_a_subjects_videos` | Alert-video baseline |
| 2026-10-29 | Undetected frames written as NaN, excluded from PERCLOS numerator and denominator, never dropped and never counted as closed | Dropping compresses the timeline and corrupts every measured duration; counting as closed makes a tracking failure look like a drowsy driver. PERCLOS also returns a per-window `coverage` so thin windows can be discarded. See `tests/test_blinks.py::test_missing_frames_do_not_count_as_closed` | NaN + coverage |
| 2026-10-29 | A closure event must reach <0.30 openness to count, on top of threshold hysteresis | Hysteresis alone let a signal sagging to 0.44 register as one 20-second microsleep. Depth gate added. See `config.EVENT_MIN_DEPTH` and `tests/test_blinks.py::test_hysteresis_rejects_threshold_noise` | Depth gate |
| 2026-10-29 | `--stride` kept at 1; no temporal downsampling | A 0.2 s blink at 15 fps is 3 frames, too few to measure duration or reopening speed, which is the earliest drowsiness indicator | Full frame rate |
| 2026-10-29 | Face-detection rate logged per video and treated as a fairness metric in its own right | If tracking fails more often on darker skin or glasses, that is a distinct failure mode from the detector being wrong, and it must not be hidden inside an accuracy number | Logged per video |
| 2026-11-02 | Pose model switched to OpenCV's Y-down camera frame; positive pitch = nodding down | The Y-up model put a frontal pitch near the ±180 wrap, which corrupts `pitch_std` and `pitch_range`. See `tests/test_geometry.py::TestHeadPoseConvention`, which starts from image coordinates because the roundtrip tests cancel the error | Y-down model |
| 2026-11-02 | ITA clipping: unreliable above `clipped_fraction` 0.02, with the median over non-clipped pixels | "Any channel at 0 or 255" tripped on 0.03% of pixels at a 0.37° spread in a smoke test. 0.02 is not yet checked on UTA-RLDD. See `data/subgroup_annotation_protocol.md` | 2% threshold |
| 2026-11-02 | Cohen's κ on age, gender presentation and glasses only; ITA gets a blind human spot-check | ITA is computed, so κ against it is not κ. The spot-check is reported as ITA-to-human agreement, never as inter-annotator reliability. See `data/subgroup_annotation_protocol.md` | κ on 3 attributes + spot-check |
| 2026-11-02 | Absolute pitch is not used; raw and centred pitch are both kept until real data decides. **Phase 6 limitation** | Absolute pitch from an uncalibrated camera is not meaningful. Only change within a video is, and even that is confounded by framing drift unless checked. Not fully solvable without camera calibration. Neither pitch is drift-proof, and `pitch_confounded` also fires on large real nods. See `tests/test_framing.py` and `phase1_baseline/README.md` | Both, undecided |
| 2026-11-17 | One landmarker instance per video, not a timestamp offset carried across videos | An offset would satisfy MediaPipe's monotonic check but carry tracking state from one subject's video into the next; reloading the cached model costs a second or two per video. See `tests/test_extraction_run.py` | Fresh instance per video |
| 2026-11-17 | Split recordings (`10_1`, `10_2`) kept as separate videos with a `part` column, not concatenated | Subject 49's halves are 270x480 and 162x288; joining them needs a re-encode that degrades the better half. Per-video features and subject-level folds both handle four videos per subject. See `tests/test_manifest.py::TestSplitRecordings` | Separate parts |
| 2026-11-04 | Rotation is verified per video rather than left to cv2, and stored size is read with AUTO off | On opencv-python 5.0.0, AUTO defaults on and `CAP_PROP_FRAME_WIDTH` reports the rotated size, so predicting from it with AUTO on swaps twice and rotates again; other builds may ignore AUTO. Checked on MP4s with patched display matrices, no UTA-RLDD file was available | `phase1_baseline/src/video.py` |
| 2026-11-25 | **Phase 2 clip length: 60 s.** Rule fixed before the sweep: shortest window whose 3-class HGB window macro F1 and majority-vote video accuracy (means over 5 seeds) are both within 2 points of the longest window | 60 s: F1 53.6, video acc 62.9 [95% CI 57.0, 68.3]. 30 s: 50.9 (−2.7, misses), 61.1 (−1.8). 15 s: 48.3, 57.4. 10 s: 47.0, 58.4. Window F1 falls monotonically with shorter windows, so nothing shorter than 60 s passes; the sweep stops at 60 s, so longer clips might do better still. 30 s is the near miss if generator limits force a shorter clip, at a 2.7-point window-F1 cost. Subject-level bootstrap CIs overlap across all four lengths — `phase1_baseline/results/detector_window_runs.csv` batch `20261125T065741Z-04b301` | 60 s |
| 2026-11-28 | **Subgroup collapse rules for the Phase 1.3 audit, fixed and committed before the audit was first run** (this row and `src/audit.py:COLLAPSE` land in a commit that precedes any `results/subgroup_audit.csv` row). Skin tone: Very light + Light → Light; Intermediate + Tan → Medium; Brown + Dark → Dark; no ITA value → `unknown`, listed, never dropped. Age: 18-25 vs 26+ (26-35, 36-50, 51+); `unsure` its own row. Gender presentation and glasses as annotated, `unsure` its own row. Groups under 3 subjects and `unknown`/`unsure` are listed with their size but kept out of the max-min gap | Collapsing is for statistical power: the six ITA bands hold 7 to 14 subjects each. Sizes in subjects (annotator 1, `data/subgroup_annotations_prefilled.csv`): skin tone Light 15, Medium 17, Dark 26, unknown 2 (subjects 26, 30); age 18-25 15, 26+ 43, unsure 2; gender presentation male 51, female 9; glasses none 48, glasses 9, unsure 3 | Fixed bands |

## Blockers / Open Questions
- NTHU-DDD needs the licence **signed by department head or lab director**. Unanswered since the first email. DROZY and NITYMED are the fallbacks for the night-IR stress test; nothing in Phases 1 or 2 depends on NTHU.
- Compute plan not yet fixed — determines whether Phase 2 uses a GAN pipeline only or can afford diffusion reenactment training too. Phase 1.1 extraction is CPU-only and runs overnight on a laptop.
- Demographic annotation (1.1) needs a second annotator for the κ check. Who?
