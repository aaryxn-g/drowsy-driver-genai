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

The suite should pass clean. It covers the geometry against hand-computed values and against
synthetically projected faces at known head angles, the blink and PERCLOS logic
against series with known ground truth, and both pipeline passes end to end with
a stub landmarker. They do not need any dataset.

Then point the self-check at one real video:

```bash
python -m src.selfcheck --video "D:/UTA-RLDD/Fold1_part1/01/0.mp4"
```

This is the one thing the test suite cannot do, because MediaPipe needs a model
file and a real face. It reports the face-detection rate, the EAR range and the
blink rate, prints `rotation_deg` and the effective resolution, and saves
`data/selfcheck_ear.png`. **Open the plot.** Every shaded
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

It also opens each video and records its orientation and quality covariates; see
"Rotation and resolution" below. A video it cannot read stays in the manifest with
`probe_error` set.

Expected output for a complete UTA-RLDD: 182 videos, 60 subjects, 5 folds, no
subject missing a class. 182 rather than 180 because subjects 32 and 49 each have
their drowsy recording split into `10_1` and `10_2`. Both parts classify as drowsy,
carry `part` 1 and 2 (blank for unsplit videos), and stay separate videos: subject
49's halves are 270x480 and 162x288, so joining them would mean a re-encode that
degrades the better half.

Every file under the root ends up in exactly one place: the manifest, `manifest_unclassified.csv`
(unknown stem, no fold, or a `video_id` seen twice), or `manifest_skipped.csv` (not a
video extension). Extensions are compared lowercased and include `.m4v`, which subject
46 uses. Check the skipped file holds no recordings.

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
| `--shard 3/16` | process only rows where `row_index % 16 == 2`; see below (`--limit` applies first) |
| `--model <path>` | use a local `face_landmarker.task` instead of downloading |

`--stride` exists but leave it at 1. Processing every second frame halves the
effective frame rate, and a 0.2 s blink at 15 fps is three frames, which is not
enough to measure its duration or its reopening speed.

To split the job across processes, use the launcher, inside tmux:

```bash
ROOT=~/data/UTA-RLDD PYTHON=~/miniconda3/envs/drowsy/bin/python scripts/run_extraction.sh 8
```

It caps ffmpeg, OpenMP and BLAS at one thread per shard, starts N shards (default 8)
two seconds apart, waits for them, greps their output for `FAILED` and `MISSING`, and
runs `--merge-logs`. Do not start the shards with a bare loop. ffmpeg's h264 decoder
starts one thread per core, the server has 64 cores and the container's `pids.max` is
2048, so 16 uncapped shards run out of processes and every one dies with
`Could not open codec h264, error: -11` before reading a frame. To merge by hand:

```bash
python -m src.extract --merge-logs --out data/frames --manifest data/manifest.csv
```

Each video gets its own landmarker instance. MediaPipe's VIDEO mode requires strictly
increasing timestamps for an instance's whole life, and every video's clock starts at
0, so a shared instance runs one video and then raises
`Input timestamp must be monotonically increasing` on every one after it.

Every shard writes Parquet into the same directory, which is safe because `video_id`
makes the filenames disjoint. Each keeps its own `_extraction_log_shard<i>of<n>.csv`,
because concurrent appends to one CSV interleave and corrupt rows. `--merge-logs`
writes `_extraction_log.csv` with one row per video, the newest if a video was redone,
and leaves the shard logs in place. With `--manifest` it also lists videos that have no
log row and exits non-zero. Failures are printed rather than logged, so check
`logs/shard*.out` for `FAILED` and `MISSING`. `--shard 1/1`, or no flag, behaves as an
ordinary single run.

The machine is shared: check the load first, and time one shard before choosing `n`,
since MediaPipe and OpenCV each start their own threads.

### 3. Aggregate into per-video features

```bash
python -m src.aggregate --manifest data/manifest.csv --frames data/frames --out data/video_features.csv
```

Produces `video_features.csv` (one row per video), `closure_events.csv` (one row
per blink or closure) and a `.meta.json` recording what was skipped and why.

It prints a class summary at the end. **PERCLOS should rise from alert to
low_vigilant to drowsy.** If it does not, something upstream is wrong and no
amount of modelling will fix it.

### 3b. Window-level features (for Phase 2)

```bash
python -m src.aggregate --manifest data/manifest.csv --window-s 60 30 15 10
```

Cuts each video into fixed-length windows (stride half the window unless
`--stride-s` is given with a single `--window-s`) and computes the same feature
set per window, via the same `summarise_frames` as the per-video path. Writes
`data/window_features_w{W}_s{S}.csv` plus a `.meta.json` per length. The
per-video file is left untouched.

- The eye baseline is still the subject's whole alert video, and
  `assert_subject_baselines` checks every row against it before writing.
- Windows with eye openness on fewer than 70% of frames
  (`config.MIN_WINDOW_COVERAGE`) are dropped. The meta file counts the drops in
  total and per subject, since uneven tracking loss is a fairness signal.
- Each window is summarised on its own, the way a generated clip will be, so a
  blink at a boundary is truncated. At 60 s or less the rolling PERCLOS window
  covers the whole slice, so `perclos80_max` equals `perclos80_mean`.

## Detector (Phase 1.3)

```bash
python -m src.detector                                    # per video
python -m src.detector --windows data/window_features_w*.csv   # per window, sweep
```

Leave-one-fold-out over the five official folds, with subject disjointness
raised on. The window path reports window-level macro F1 and video accuracy from
a majority vote over each video's windows. Results are appended to
`results/detector_runs.csv` and `results/detector_window_runs.csv`.

---

## Subgroup annotation

Run this once the dataset is on disk. It needs only each subject's alert video.

```bash
python -m src.subgroups extract-refs --manifest data/manifest.csv --root "D:/UTA-RLDD"
python -m src.subgroups ita --refs data/refs --out data/ita.csv
```

`extract-refs` saves up to three full-resolution PNGs per subject to
`data/refs/<subject_id>_{0,1,2}.png`. The frames are frontal (|yaw| < 10 degrees),
near the video's median brightness and at least 15 s apart, so the spread check
compares separate moments. It can resume, and `data/refs/_refs_log.csv` records
what was chosen and why any subject fell short.

`ita` writes:

| File | Contents |
|---|---|
| `data/ita.csv` | ITA per candidate, median, spread, band, `ita_reliable` and the reason if not |
| `data/subgroup_annotations_prefilled.csv` | every subject, in schema column order; only ITA and clip ids filled |
| `data/subgroup_annotations_prefilled_annotator2_blind.csv` | seeded 20% sample for the second annotator, with no ITA and no first-annotator labels; includes the skin-tone spot-check |

ITA is the median over the patch pixels that are not clipped (no channel at 0 or
255). `ita_reliable` is False if the spread exceeds 10 degrees, more than 2% of a
candidate's patch pixels are clipped, there are fewer than three usable candidates,
or b* <= 0. `clipped_fraction_*` is kept per candidate so the threshold can be tuned
on real data.
Subjects with no usable frame still get a row. `ita` refuses to overwrite existing
sheets, because they may hold hand labels.

In the prefilled sheet, `age_bracket`, `gender_presentation` and `glasses` hold the
adjudicated final labels. Each annotator's own labels go in the `_a1` and `_a2`
columns, which feed Cohen's kappa. Skin tone is computed, so it gets no kappa:
`skin_tone_spot_check` is a blind human check that validates the ITA band, and it
is reported as ITA-to-human agreement. Save the adjudicated result as `data/subgroup_annotations.csv`, the file
everything downstream reads.

---

## Rotation and resolution

Phone videos store sideways pixels plus a rotation flag in the container. Whether
`cv2.VideoCapture` acts on that flag depends on the build, and getting it wrong is
silent: EAR is vertical over horizontal, so a turned face makes a blink push EAR up.
`src/video.py` therefore requests rotation, checks the first decoded frame against a
prediction made from the stored size, and rotates by hand if the request was ignored.
A video whose frames match no prediction fails loudly rather than going through.

Frames are rotated before the `--max-width` downscale, since rotating swaps the
dimensions and so decides whether the limit applies. The limit caps width only, so a
portrait video keeps more pixels than the same footage stored as landscape:
`inference_width` and `inference_height` record what the landmarker actually saw.

| Column | In | Meaning |
|---|---|---|
| `rotation_deg` | manifest, log, features | clockwise turn to display, 0/90/180/270. OpenCV's sign: ffprobe's -90 is 90 here, ffprobe's +90 is 270 |
| `rotation_method` | manifest, log, features | `none`, `auto` (decoder rotated) or `manual` (we rotated) |
| `stored_width`, `stored_height` | manifest, log, features | as stored in the file |
| `effective_width`, `effective_height` | manifest, log, features | after rotation, before downscaling |
| `inference_width`, `inference_height` | log | after downscaling |
| `fps`, `duration_s` | manifest, log | from the container; features has its own, from the frames read |
| `bitrate_kbps` | manifest, log, features | from the container, else file size over duration |

`low_resolution` is True when the short side of the effective size is below
`MIN_RESOLUTION_SHORT_SIDE` (360 px, a starting value not yet checked on UTA-RLDD). It
sits beside `low_quality` in `video_features.csv`, and the aggregate run prints and
records the affected videos and subjects. **It is a flag, not a filter.** Low-resolution
subjects are kept, because dropping subjects destroys the ability to measure the gap.
Blank means the manifest had no resolution for that video, so rebuild it. Resolution is
a covariate to regress out in Phase 1.3, or a landmark-precision effect can be misread
as a subgroup accuracy gap.

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
  frames/<video_id>.parquet    per-frame EAR, MAR, raw and centred pose, face position, detected
  frames/_extraction_log.csv   rotation, resolution, fps, bitrate, detection rate, runtime per video
  frames/_extraction_log_shard<i>of<n>.csv   the same, per shard, before --merge-logs
  video_features.csv           per-video features, the Phase 1.2 model input
  closure_events.csv           every blink and closure with duration and speeds
  video_features.meta.json     what was skipped and why
```

### Per-video features

PERCLOS (mean and max, P80 and P70), blink count and rate, mean blink duration,
mean closing and reopening speed, long-closure and microsleep counts, total
closure time, MAR mean and 95th percentile, yawn frame fraction, head-pitch
standard deviation and range, plus `detection_rate` and the `low_quality` and
`low_resolution` flags, with the rotation and resolution columns described above.

Framing columns: `pitch_centered_std`, `pitch_centered_range`, `nose_y_drift`,
`nose_x_drift`, `face_scale_drift`, `pitch_nose_y_corr` and `pitch_confounded`. See
"Head pitch and framing" below.

---

## Head pitch and framing

Positive pitch means nodding down.

**Absolute pitch from an uncalibrated camera is not meaningful.** Only change within
a video is, and even that is confounded by framing drift unless checked. A subject
slumping down the frame over ten minutes can look like a head nod, and because
slumping goes with drowsiness, the feature could look predictive while measuring the
wrong thing. This is a limitation for the Phase 6 write-up. It cannot be fully solved
without camera calibration.

Both pitches are kept, and the choice between them waits for real data:

| Column | Holds up when | Goes wrong when |
|---|---|---|
| `pitch` | the head moves through space without rotating | the face stays aimed at the camera while moving in the frame |
| `pitch_centered` | the image shifts but the face looks the same (a crop, or a face kept aimed at the camera) | the head moves through space without rotating, the plain slump |

So `pitch_centered` is not a slump-proof version of `pitch`.
`tests/test_framing.py` pins both cases.

`pitch_confounded` is True when `abs(pitch_nose_y_corr) > 0.5` and `nose_y_drift > 0.1`.
It means the apparent nodding *may* be movement in the frame, not that it is. A real
nod pivots at the neck and moves the nose down the frame as well, so large real nods
trip the flag too, and more flagged drowsy videos than alert ones can mean either
thing. Both limits are starting values, not yet checked on UTA-RLDD. The aggregate run
prints the flagged fraction per class.

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
