# Claude Code prompts for the GPU server

Run these in a Claude Code session started inside a **VS Code Remote-SSH terminal**, so it
operates on the server's filesystem and can see the GPU. Plain Claude Code on your laptop
will do the wrong thing for all of these.

Use with `docs/CLAUDE_CODE_PROMPTS.md`, which covers the project work itself. This file is
only about the server.

---

## S-0 Survey the machine, first thing

```
I have SSH access to a shared faculty GPU server and you are running on it now. Before any
project work, survey the machine and fill in the table in
phase0_scoping/compute_plan.md under "Before running anything".

Run and report:
  nvidia-smi, plus the csv query for index/name/memory.total/memory.used
  nproc, free -g, df -h ~ , df -h /
  quota -s if it exists
  which sbatch squeue sinfo
  python3 --version, which conda, conda --version

Then tell me plainly:
- how many GPUs there are, which are currently free, and how much VRAM each has
- whether other users' processes are running right now
- how much disk I can safely use
- whether there is a job scheduler I should be submitting to rather than running directly
- whether the server can reach the internet: test with a small file download, not by
  assuming

Write the answers into phase0_scoping/compute_plan.md. Do not install anything yet.
```

## S-1 Set up the environment

```
Create the project environment on this server following phase0_scoping/compute_plan.md.

1. conda env named drowsy, python 3.11
2. Install phase1_baseline/requirements.txt
3. Install the torch build matching THIS server's CUDA version. Check nvidia-smi for the
   version first and pick the matching wheel index. Do not install the default pip torch
   without checking, it is often CPU-only.
4. Verify with: python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
   If cuda.is_available() is False, stop and tell me, do not proceed.
5. cd phase1_baseline && python -m pytest tests -q

Report the torch version, the CUDA version it was built against, and the test result.
```

## S-2 Get the dataset onto the server

```
I need UTA-RLDD on this server. It is 111.3 GB across 10 zip files on Google Drive:
https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl

Before downloading anything:
1. Confirm there is enough free disk for one fold at a time, about 25 GB working set.
   Check any quota. Read phase0_scoping/data_budget.md for why we do this fold by fold.
2. Confirm the server can actually reach Google Drive. Test with one small file first.

Then download ONE fold only, verify it extracts, and stop. I want to check the structure
matches what src/manifest.py expects before pulling the other nine zips.

Use a tool that resumes on interruption, and run it inside tmux so an SSH drop does not
kill it. Tell me the tmux session name.
```

## S-3 Run extraction with the server's cores

```
Read phase0_scoping/data_budget.md and phase1_baseline/README.md.

This server has <N> cores. Time the extraction on 3 videos first (--limit 3) and project
the full 180-video runtime, single process and parallel.

If single-process is over about 6 hours, implement the parallel runner from prompt 1.1-D in
docs/CLAUDE_CODE_PROMPTS.md, sharding by subject across cores. Cap workers at nproc-2 so I
am not monopolising a shared machine, and tell me what you capped it to.

Run the real job inside tmux. Before starting, show me:
- the projected runtime
- the projected disk use for the parquet output
- how many workers you will use and how much RAM that needs

Then start it and tell me how to check progress without attaching to the session.
```

## S-4 Confirm the GPU actually works end to end

Do this before Phase 2, not during it.

```
Write a short throwaway script that confirms this server can train something on the GPU:
a few hundred steps of a small CNN on random tensors, on one pinned GPU via
CUDA_VISIBLE_DEVICES.

Report: which GPU, peak VRAM used, steps per second, and whether anything else was running
on that GPU at the time.

This is a smoke test, so put it in a scratch directory, not in the repo.
```

## S-5 Long training runs

```
I am about to start a long training job on a shared GPU server with no scheduler.

Before launching:
1. nvidia-smi to confirm which GPU is free, and pin to it with CUDA_VISIBLE_DEVICES
2. Start it in tmux
3. Make sure it checkpoints often enough that losing the machine costs me under an hour
4. Make sure it appends to the results CSV with a run_id as it goes, not only at the end.
   A run that dies at hour 9 should still leave usable partial results.

Tell me the tmux session name, the GPU index, and how to check progress from a fresh
terminal.
```

---

## Habits worth keeping

**`nvidia-smi` before every GPU job.** Somebody else may be mid-run. There is no scheduler
stopping you from taking their memory out from under them.

**`tmux` for anything over about 10 minutes.** SSH drops.

**Check `df -h` before large downloads.** On a shared account, filling the disk is
everyone's problem, not just yours.

**Delete fold video after extracting its features.** The feature tables are about 300 MB in
total. The video is 111 GB. There is no reason to keep both.
