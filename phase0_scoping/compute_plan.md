# Compute Plan

Surveyed 2026-11-02 on the real machine. Supersedes the earlier estimates.

## What we have

A faculty GPU server, reached over SSH. Each user is dropped into an isolated GPU
environment by a login wrapper, and `sudo` plus `apt` are available inside it.

| | |
|---|---|
| Host | `172.16.100.14` (campus network) |
| Account | `Shivang` (capital S), faculty-issued |
| GPUs | **2 x Tesla V100-PCIE-32GB**, 32 GB VRAM each, 64 GB total |
| CPU | **64 cores** |
| Disk | `/home/Shivang` on a 4.4 TB volume, **904 GB free** (79% used) |
| Network | git, GitHub and pypi all reachable |
| Scheduler | none seen; confirm with `which sbatch squeue` |

This is substantially more capable than the project was planned around.

### Known limitation: scp and sftp do not work

The login wrapper runs `sudo` to place you in the isolated environment. A non-interactive
connection has no terminal to prompt on, so it fails with:

```
sudo: a terminal is required to read the password
lost connection
```

Nothing in the user's own dotfiles causes this (`grep sudo ~/.bashrc ~/.profile` is empty),
so it is configured at system level and is not something to work around locally.

**Consequence: move code with git, not scp.** git is installed and GitHub is reachable, so
clone onto the server and pull to update. This is better practice anyway.

## What the real specs change

### Phase 1.1 extraction: minutes, not an overnight job

64 cores against roughly 3.2 million frames. MediaPipe runs single-threaded per process at
somewhere around 25 to 50 fps, so with 48 workers the full 180-video extraction should land
in the **30 to 60 minute** range rather than the 18 to 36 hours a laptop would take.

Cap workers at about 48 rather than all 64: it is a shared machine, and each MediaPipe
process holds a few hundred MB. Confirm free RAM with `free -g` before choosing, and if RAM
is tight, fewer workers.

### Storage: no fold-by-fold streaming needed

904 GB free against a 111 GB dataset. Download the whole of UTA-RLDD at once. The
fold-by-fold workflow in `data_budget.md` was written for a laptop and is no longer
necessary, though it remains the fallback if the shared volume fills up.

Keep an eye on it regardless. The volume is already 79% full and other people use it.

### Phase 2: both generators are now viable

The earlier plan assumed Colab-class compute and defaulted to pretrained GAN inference,
with diffusion as a token pilot. With 32 GB of VRAM per card that assumption is void:
diffusion reenactment fine-tuning is genuinely feasible, so the 2.1 comparison can be a
real experiment rather than a formality.

Two practical notes on V100 specifically:

**No bf16.** Volta has fp16 tensor cores but bfloat16 needs Ampere or newer. Use
`torch.cuda.amp` with fp16, and watch for overflow in loss scaling. Code copied from a
modern repo often assumes `bf16` and will fail or silently fall back to fp32.

**Two cards, two options.** Either run two experiments side by side, one per GPU, which
suits the Phase 2.1 comparison and the Phase 4 sweep, or use both for one job via DDP.
Parallel experiments are usually the better use here, since the sweep is 36 independent
runs.

### Phase 4: the sweep is comfortable

36 runs across six ratios, three seeds and two tracks. Two V100s running independent jobs
makes this routine rather than the bottleneck it would have been on notebook sessions.

## Environment

```bash
conda create -n drowsy python=3.11 -y
conda activate drowsy
pip install -r phase1_baseline/requirements.txt
```

Install torch matching the server's CUDA, not the default wheel. `scripts/server_setup.sh`
detects the version and picks the right index automatically. Verify:

```python
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
```

Expect `True` and `2`.

## Shared-machine habits

No scheduler was found, so nothing stops you taking a GPU somebody else is mid-run on.

```bash
nvidia-smi                              # who is on which card, before you start
CUDA_VISIBLE_DEVICES=0 python ...       # pin to one card
tmux new -s extract                     # survive the SSH drop
df -h ~                                 # the volume is shared and already 79% full
```

## Still to confirm

- [ ] Free RAM (`free -g`), which caps the extraction worker count
- [ ] Whether a scheduler exists after all (`which sbatch squeue sinfo`)
- [ ] Whether the account is shared with other students
- [ ] Whether there is an etiquette expectation about daytime GPU use
