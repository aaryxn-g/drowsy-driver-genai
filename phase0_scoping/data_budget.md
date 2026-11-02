# Data budget and subsetting policy

Written 2026-11-02, in answer to "if the data sizes are extremely big, can we use portions
instead of the full set and note it as a limitation?"

Short version: yes, but **which axis you cut decides whether the project still works.**
Cutting along the wrong axis destroys the result you are trying to produce, and it does so
silently.

---

## The rule

### Never drop subjects

Every headline number in this project is a gap between demographic subgroups. UTA-RLDD
gives you 60 subjects to measure that gap with, and the groups you care about are already
tiny:

| Group | Subjects |
|---|---|
| Women | 9 of 60 |
| African descent | 0 of 60 |
| Glasses | 21 of 180 videos |
| Age 40+ | a handful; mean age is 25 |

Take a random 50% of subjects and you are left with roughly 4 or 5 women. A subgroup F1
estimated from 4 people has a confidence interval wide enough to swallow any effect you
might find. The project does not become less precise, it becomes unable to answer its own
question, and the failure shows up as "no significant difference" rather than as an error.

So: **all 60 subjects, all 180 videos, always.** This is not negotiable, and it is also
cheap, as the next section shows.

### Cut time and clips instead

These are safe to reduce, in rough order of safety:

1. **Seconds per video.** Each UTA-RLDD video is one constant state for about 10 minutes.
2. **Synthetic clips generated** in Phases 2 and 3.
3. **Identities downloaded** from FFHQ.

### Whatever you cut, prove the composition survived

Before and after any reduction, print the subject count per subgroup and confirm it is
unchanged. Put that table in the report. A limitation you can quantify is a limitation; one
you assert is a hole.

---

## Storage is not actually your problem

The 111 GB figure is the download size, not the working-set size. You never need it all at
once.

| Item | Size |
|---|---|
| UTA-RLDD, all 10 zips | 111.3 GB |
| One fold (2 parts), zipped | about 22 GB |
| One fold, extracted | about 22 GB |
| **Peak working set, one fold at a time** | **about 25 GB** |
| Per-frame features, one video | 1 to 2 MB |
| **Per-frame features, all 180 videos** | **about 300 MB** |

The workflow, which `src/run_fold.py` exists to support (prompt 1.1-C):

1. Download fold N
2. Extract features for fold N
3. Verify the parquet files are there and the detection rates look sane
4. Delete fold N's video
5. Repeat

You end up with roughly 300 MB of feature tables that drive the whole of Phase 1, and you
never need more than about 25 GB free at any moment.

**One thing to keep before deleting:** Phase 2.2 mines cue templates, and depending on the
generator you pick it may want real driving frames, not just the trajectories. Keep a
curated set of short segments, say 10 to 30 seconds around interesting closures and yawns,
spread across subjects and classes. A few GB, and it saves re-downloading later.

### FFHQ: 3 GB, not 89 GB

Do not download the full 89 GB image set. The pipeline is:

1. Download thumbnails at 128px plus the JSON metadata: **1.95 GB**
2. Run the FairFace classifier over the thumbnails to get estimated demographics for all 70,000
3. Select only the identities that fill UTA-RLDD's gaps, perhaps 500 to 1000
4. Download **only those** at 1024px, roughly 1 MB each

Total around 3 GB. Log the FairFace labels as *estimated* attributes in the provenance
metadata; they come from a classifier, not from self-report, and the dataset card has to
say so.

---

## Compute is your actual constraint

Feature extraction is CPU-bound and single-threaded per process.

- 30 hours of video at 30 fps is about 3.2 million frames
- MediaPipe at 640px on a laptop CPU runs somewhere in the 25 to 50 fps range

That puts the full extraction somewhere between **18 and 36 hours single-process**. Do not
take my range on faith: prompt 1.1-B times three real videos on your actual machine and
extrapolates. Measure before you plan around it.

Two ways to shrink it:

**Run folds in parallel.** MediaPipe does not parallelise inside a process, but you can run
one process per fold. Four workers takes it to roughly 5 to 9 hours. Prompt 1.1-D.

**Halve the seconds per video.** Each video is a single constant state, so the second half
tells you much the same as the first. Five minutes still gives you roughly 300 one-second
PERCLOS windows and 75 to 100 blinks per video, which is plenty for stable statistics.

If you do cut, take the **middle** five minutes rather than the first five. The opening
seconds of a self-recorded video contain the subject settling into frame and the webcam
still adjusting exposure, and those frames are exactly where landmark tracking is worst.

**My recommendation:** run the full 10 minutes. It is an overnight job either way, and
having the full-length features means Phase 2 and Phase 5 never need re-extraction. Cut to
five minutes only if 1.1-B comes back above about 30 hours.

---

## Sizing the synthetic data

This is where compute genuinely bites, because it is GPU time.

Real UTA-RLDD is 180 videos across 60 subjects. For the synthetic set to shift subgroup
representation enough to measure, it has to be comparable in scale, not a token sample.

| Batch | Suggested scale | Why |
|---|---|---|
| Pilot (2.1) | 50 clips, about 10 s each | Enough to compare two generators on cue fidelity |
| Untargeted (3) | 150 to 200 clips, 30 s | Matches the real distribution, the control arm |
| Targeted (3) | 300 to 400 clips, 30 s | Needs to over-represent the missing groups to move the gap |

If GPU time forces a cut, cut **clip length before clip count**. Forty subjects' worth of
15-second clips beats twenty subjects' worth of 30-second clips, because identity diversity
is the entire point of generating them. A 15-second clip still holds several blinks and a
PERCLOS window.

---

## What goes in the limitations section

Write these as numbers, not as hedges:

- "Features were extracted from the middle N minutes of each 10-minute recording, giving M
  PERCLOS windows per video" if you subset time
- "The synthetic set comprises N clips across K identities, against 180 real videos across
  60 subjects"
- "Skin-tone subgroups contain as few as N subjects; gaps for those bands are reported with
  confidence intervals and excluded from the headline figure"
- "FFHQ demographic attributes are classifier estimates, not self-reported"

"Results may not generalise" is not a limitation. It is a sentence that could appear in any
paper ever written.
