# Phase 1.1 — feature extraction pipeline

Turns raw driver video into the per-frame and per-video drowsiness features that
everything later in the project is built on: EAR, MAR, head pose, blink events
and PERCLOS.

Nothing here trains a model. This is the measurement layer.

---

## Install

From inside `phase1_baseline/`:

```bash
conda create -n drowsy python=3.11 -y
conda activate drowsy
pip install -r requirements.txt
```

Python 3.11 rather than 3.13, because several of the pinned packages do not yet
ship 3.13 wheels on Windows.

## Check it works before committing to a long run

```bash
python -m pytest tests -q
```

54 tests. They cover the geometry against hand-computed values and against
synthetically projected faces at known head angles, the blink and PERCLOS logic
against series with known ground truth, and both pipeline passes end to end with
a stub landmarker. They do not need any dataset.

Then point the self-check at one real video:

```bash
python -m src.selfcheck --video "D:/UTA-RLDD/Fold1_part1/01/0.mp4"
```

This is the one thing the test suite cannot do, because MediaPipe needs a model
file and a real face. It reports the face-detection rate, the EAR range and the
blink rate, and saves `data/selfcheck_ear.png`. **Open the plot.** Every shaded
band should line up with a blink you can see. If the bands are in the wrong
places, stop and fix that before running anything else.

---

## Run

### 1. Build the manifest

```bash
python -m src.manifest --root "D:/UTA-RLDD" --out data/manifest.csv
```

Walks the extracted folds and records subject, fold and class for every video.
Anything it cannot classify goes to `data/manifest_unclassified.csv` rather than
being guessed at. Read that file if it appears.

Expected output for a complete UTA-RLDD: 180 videos, 60 subjects, 5 folds,
60 videos per class.

### 2. Extract per-frame features

```bash
python -m src.extract --manifest data/manifest.csv --root "D:/UTA-RLDD" --out data/frames
```

One Parquet file per video. This is the slow pass: roughly 30 hours of video, at
CPU landmarking speed, is an overnight job. It is resumable, so a video already
written is skipped and an interruption costs you one video.

Useful flags:

| Flag | Why |
|---|---|
| `--limit 5` | first five videos only, to time the job before launching it |
| `--max-frames 900` | first 30 s of each video, for a fast dry run |
| `--max-width 640` | inference resolution (default 640; lower is faster) |
| `--overwrite` | redo videos already extracted |
| `--model <path>` | use a local `face_landmarker.task` instead of downloading |

`--stride` exists but leave it at 1. Processing every second frame halves the
effective frame rate, and a 0.2 s blink at 15 fps is three frames, which is not
enough to measure its duration or its reopening speed.

### 3. Aggregate into per-video features

```bash
python -m src.aggregate --manifest data/manifest.csv --frames data/frames --out data/video_features.csv
```

Produces `video_features.csv` (one row per video), `closure_events.csv` (one row
per blink or closure) and a `.meta.json` recording what was skipped and why.

It prints a class summary at the end. **PERCLOS should rise from alert to
low_vigilant to drowsy.** If it does not, something upstream is wrong and no
amount of modelling will fix it.

---

## The one design decision worth knowing

Eye openness is meaningless in absolute terms. Eye shape varies so much between
people that a fixed EAR threshold would mark narrow-eyed subjects as permanently
drowsy. So EAR is rescaled per subject against their own fully-open eye.

That baseline is taken from the subject's **alert** video and reused for all
three of their videos.

The obvious alternative, taking a percentile within each video separately, is
wrong in a way that is easy to miss: a drowsy recording may never contain a
fully open eye, so a within-video baseline silently rescales the drowsiness away
and makes every subject look equally alert. The pipeline would run clean and the
numbers would be worthless.

If a subject has no usable alert baseline they are reported and skipped, not
scored on a guess.

---

## Outputs

```
data/
  manifest.csv                 subject, fold, class, path per video
  manifest_unclassified.csv    anything the scanner refused to label
  frames/<video_id>.parquet    per-frame EAR, MAR, yaw, pitch, roll, detected
  frames/_extraction_log.csv   fps, detection rate, runtime per video
  video_features.csv           per-video features, the Phase 1.2 model input
  closure_events.csv           every blink and closure with duration and speeds
  video_features.meta.json     what was skipped and why
```

### Per-video features

PERCLOS (mean and max, P80 and P70), blink count and rate, mean blink duration,
mean closing and reopening speed, long-closure and microsleep counts, total
closure time, MAR mean and 95th percentile, yawn frame fraction, head-pitch
standard deviation and range, plus `detection_rate` and a `low_quality` flag.

---

## Two things to watch

**Detection rate is a fairness signal, not a preprocessing detail.** It is
recorded per video for exactly this reason. If MediaPipe fails more often on
darker-skinned subjects or on people wearing glasses, that is a finding for
Phase 1.3, and it is a different failure mode from the detector being wrong. A
model that is never given a face cannot be fair. Check
`_extraction_log.csv` grouped by subgroup before concluding anything about
accuracy gaps.

**Frames where no face was found are written as NaN, never dropped.** Dropping
them would compress the timeline and corrupt every duration measured afterwards.
PERCLOS excludes them from both numerator and denominator and reports a
`coverage` figure, so a tracking failure can never masquerade as a closed eye.

---

## Next

Phase 1.2 consumes `video_features.csv`. Before that, Phase 1.1 still needs the
demographic annotation pass (`data/subgroup_annotation_protocol.md`), which is
what makes the subgroup audit in 1.3 possible.
