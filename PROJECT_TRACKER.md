# Project Tracker — Synthetic Drowsy-Driver Data Generation & Fatigue Detection

Owner: Aaryan (BITS Pilani Dubai) · Started: 2026-09-21 · Last updated: 2026-10-29

Legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

This file is the single source of truth for status. Update it every session (you or Claude Code)
so nothing gets lost between sessions. Each phase folder has its own README with finer detail.

---

## Phase 0 — Scoping, literature, dataset access, resource audit
- [~] 0.1 Annotated bibliography (drowsiness detection SOTA / face reenactment / synthetic-fairness) — `phase0_scoping/annotated_bibliography.md`
- [x] 0.2 Final dataset list verified live (27 Sep): UTA-RLDD, NTHU-DDD (official), YawDD, FFHQ, FairFace — `phase0_scoping/dataset_decision.md`
- [x] 0.2 Licences reviewed and logged (FFHQ CC BY-NC-SA 4.0 → synthetic release must match; FairFace CC BY 4.0; NTHU no-redistribution)
- [ ] 0.2 **UTA-RLDD downloaded** (direct, no form; 111.3 GB across 10 zips) ← blocks Phase 1.1 execution
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
- [x] 1.1 Test suite: 120 tests, geometry vs synthetic projections, ITA vs hand-computed values, pipeline end to end with a stub landmarker
- [x] 1.1 Subgroup annotation tool: reference frames, ITA, pre-filled and blind sheets (no real data run yet) — `phase1_baseline/src/subgroups.py`
- [x] 1.1 Head-pose axis fix: pose model now matches OpenCV's Y-down camera frame, positive pitch = nodding down — `phase1_baseline/src/config.py`
- [x] 1.1 Server env: MediaPipe loads on the tasks backend after `apt-get install libegl1 libgles2`
- [ ] 1.1 **Self-check run on one real video** (`python -m src.selfcheck --video ...`) ← first thing to do once data lands
- [ ] 1.1 Full extraction run over all 180 videos
- [ ] 1.1 Demographic annotation pass (skin tone/ITA, age bracket, gender presentation, glasses) + 2nd annotator κ check on age, gender presentation and glasses + blind ITA spot-check
- [ ] 1.1 Class/subgroup balance table generated
- [ ] 1.2 Architecture benchmark: CNN-hybrid vs YOLOv8-based vs ViT/Swin vs LSTM-temporal — `phase1_baseline/results/architecture_benchmark.csv`
- [ ] 1.2 Baseline architecture selected + justification written
- [ ] 1.3 Per-subgroup F1 + bootstrap CIs computed
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

## Blockers / Open Questions
- **UTA-RLDD not yet downloaded.** 111.3 GB across 10 zips, direct from Google Drive, no form needed. Phase 1.1 code is written and tested but cannot run until this lands. Download one fold first and run the self-check on a single video before pulling the rest.
- NTHU-DDD needs the licence **signed by department head or lab director**. Unanswered since the first email. DROZY and NITYMED are the fallbacks for the night-IR stress test; nothing in Phases 1 or 2 depends on NTHU.
- Compute plan not yet fixed — determines whether Phase 2 uses a GAN pipeline only or can afford diffusion reenactment training too. Phase 1.1 extraction is CPU-only and runs overnight on a laptop.
- Demographic annotation (1.1) needs a second annotator for the κ check. Who?
