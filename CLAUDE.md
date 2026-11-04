# CLAUDE.md — project memory for Claude Code sessions

Read automatically at the start of a session in this repo. Keep it current: it is how a
fresh session picks up context instantly.

Last updated 2026-11-04.

## What this project is

Use conditional face-video generation (GAN or diffusion reenactment) to synthesise
drowsiness cues (eye closure, yawn, head nod) on demographically diverse synthetic
identities, in order to close subgroup accuracy gaps in a real-time fatigue detector
trained on small, imbalanced real data.

The headline result is **not** higher accuracy. It is a smaller gap between the best and
worst demographic subgroup, with the overall figure held, reported honestly either way.

- Phase-by-phase plan: `docs/full_project_plan.md`
- Live status: `PROJECT_TRACKER.md` — check first, update after every finished item
- Prompts for this repo: `docs/CLAUDE_CODE_PROMPTS.md`
- Prompts for the GPU server: `docs/SERVER_PROMPTS.md`

## Where things run

A faculty GPU server on the campus network, over SSH and VS Code Remote-SSH. See
`phase0_scoping/compute_plan.md`. Everything heavy runs there, including the dataset
download and the CPU-bound feature extraction. The laptop is for writing and git.

Shared machine, no known scheduler: run `nvidia-smi` before any GPU job, pin with
`CUDA_VISIBLE_DEVICES`, use tmux for anything long.

## Repo layout

- `phase0_scoping/` — literature, dataset decisions, compute plan, data budget
- `phase1_baseline/` — baseline detector: feature pipeline, architecture benchmark, subgroup audit
- `phase2_generator/` — GAN/diffusion reenactment pipeline and pilot comparison
- `phase3_synthetic_dataset/` — generated dataset (raw and balanced) and QC
- `phase4_retraining_sweep/` — real+synthetic ratio sweep, the centrepiece result
- `phase5_stress_test/` — night/IR, occlusion, cross-dataset evaluation
- `phase6_responsible_ai/` — fairness, privacy, false alarms, datasheet
- `phase7_writeup/` — ablations, final report

## Phase 1.1 is built. Read it before touching it.

`phase1_baseline/src/` holds a working, tested two-pass feature pipeline. Full description
in `phase1_baseline/README.md`.

| Module | Role |
|---|---|
| `manifest.py` | scans UTA-RLDD, discovers subject/fold/class, refuses to guess |
| `video.py` | opens videos the right way up, verifies rotation, records quality covariates |
| `geometry.py` | per-frame EAR, MAR, head pose (solvePnP), yaw correction |
| `blinks.py` | openness normalisation, closure events, rolling PERCLOS |
| `landmarks.py` | MediaPipe wrapper, supports both solutions and tasks APIs |
| `extract.py` | pass 1, video to per-frame parquet, resumable |
| `aggregate.py` | pass 2, per-frame to per-video features |
| `selfcheck.py` | validates the landmarker on one real video |
| `subgroups.py` | reference frames, ITA and annotation sheets; never guesses human labels |

`python -m pytest tests -q` from `phase1_baseline/` should pass clean. If it does not, fix
that before anything else.

## Invariants that must not be broken

These are load-bearing. Breaking one invalidates results silently rather than loudly.

**Subject-level splits, always.** Use the five official UTA-RLDD folds. A subject appears in
exactly one fold. Never a random frame or video split. Identity leakage is why the
literature reports 99% and honest evaluation reports 65%. Assert it in code, do not merely
intend it.

**The eye-openness baseline is per subject, from their alert video**, reused across all
three of their videos. A within-video percentile would rescale drowsiness away, because a
drowsy video may never contain a fully open eye. The pipeline would run clean and the
numbers would be worthless. See `aggregate.py:build_baselines`.

**Undetected frames are NaN.** Never dropped (compresses the timeline, corrupts durations),
never counted as closed (makes a tracking failure look like a drowsy driver). PERCLOS
excludes them from numerator and denominator and reports `coverage`.

**Never drop subjects to save time or space.** 60 subjects, 9 women, 0 of African descent.
Subsetting subjects destroys the ability to measure the gap. Cut clip length or clip count
instead. See `phase0_scoping/data_budget.md`.

**Test on real data only.** Synthetic samples never appear in a test set, at any mix ratio.

**Face-detection rate is a fairness metric**, logged per video. If tracking fails more for
one subgroup, that is a distinct finding from the detector being inaccurate, and it must
not hide inside an accuracy number.

**Rotation metadata must be honoured, and never assumed to be.** UTA-RLDD files store sideways
pixels plus a container flag, and whether `cv2.VideoCapture` acts on it depends on the build.
A sideways face inverts EAR without raising anything, and if the file is an alert video it
corrupts that subject's baseline and so all three of their videos. Rotate before any downscale.
Read video through `video.OrientedVideo`, which checks the result per video. `subgroups.py`
does not do this yet; see the tracker. `rotation_deg` uses OpenCV's sign, the opposite of ffprobe's.

## Conventions

- Every trained run logs architecture, mix ratio, seed, subgroup metadata version, overall
  F1, per-subgroup F1 and latency to a CSV in that phase's `results/`. Append with a
  `run_id`; never overwrite.
- Seeds set and logged for numpy, torch and python random. Minimum 3 per Phase 4 cell.
- Real and synthetic samples always carry `is_synthetic` and `provenance_id`.
- Subgroup labels come from `phase1_baseline/data/subgroup_annotations.csv`. Never re-derive
  ad hoc.
- Any "we chose X because" claim in the report must point at a results file or chart.

## Datasets (verified 27 Sep 2026, see `phase0_scoping/dataset_decision.md`)

UTA-RLDD is primary: 60 subjects, 180 videos, roughly 30 hours, colour, real (not acted)
drowsiness, five official folds. The published baseline on those folds is **65.2% video
accuracy** (Ghoddoosian et al., CVPRW 2019); human observers scored 57.8%.

Also: YawDD (yawn cues, real in-car colour), NTHU-DDD (night IR and glasses stress test,
official licence only, currently unanswered), FFHQ (identity pool, CC BY-NC-SA 4.0, which
forces the synthetic release licence), FairFace (demographic labelling).

CelebA and DMD are not used. Never use NTHU-DDD Kaggle mirrors: frames only, no subject
IDs, unknown licence, and they breach NTHU's no-redistribution term.

## Current status

Phase 1.1 code complete and tested. **Blocked on the UTA-RLDD download.** Nothing has been
run against real data yet, so there are no results in this repo and none should be invented.

If data is not present under the configured dataset root, say so and stop rather than
producing placeholder numbers.

## Environment

Python 3.11, not 3.13 (missing wheels). Install from `phase1_baseline/requirements.txt`.
MediaPipe 1.2 needs `libEGL.so.1` and `libGLESv2.so.2`, which the tests do not catch because
they stub the landmarker. Install them with `sudo apt-get install -y libegl1 libgles2`, not
conda, since libgl1 also comes from apt. Run `src.selfcheck` to confirm a real load.

Check `torch.cuda.is_available()` before assuming GPU; a CPU-only torch on a GPU server is a
common and quiet mistake.
