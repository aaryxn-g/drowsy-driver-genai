# Server scripts

Run these on the GPU server, from the repo root, in a VS Code Remote-SSH terminal.

```bash
bash scripts/server_survey.sh    # read-only. writes server_report.txt
bash scripts/server_setup.sh     # creates the conda env. writes setup_report.txt
bash scripts/run_extraction.sh 8 # Phase 1.1 pass 1 as 8 shards; run inside tmux
```

Send both report files back.

## If you see `bad interpreter: ^M`

The files picked up Windows line endings on the way over. Fix once:

```bash
sed -i 's/\r$//' scripts/*.sh
```

## What each one does

**server_survey.sh** installs nothing and changes nothing. It records GPUs and who is
using them, CPU cores, RAM, disk and quota, whether a job scheduler exists, which tools
are present, and whether the server can actually reach pypi, GitHub and Google Drive.
That last check decides whether the dataset can be downloaded on the server or has to
come via your laptop.

**server_setup.sh** creates a conda env called `drowsy` with Python 3.11, installs a
torch build matching the server's CUDA version rather than the default CPU-only wheel,
installs the project requirements, and runs the test suite, which should pass clean. Everything stays inside
the conda env and this repo.

Override the env name if `drowsy` is taken on a shared account:

```bash
ENV_NAME=aaryan_drowsy bash scripts/server_setup.sh
```

**run_extraction.sh** runs `src.extract` as N shards (default 8) from `phase1_baseline/`,
with ffmpeg, OpenMP and BLAS capped at one thread each, then merges the shard logs. Set
`ROOT` to the dataset root (default `~/data/UTA-RLDD`) and `PYTHON` to the env's python.
Extra arguments pass through to `src.extract`, e.g. `--max-frames 900` for a dry run.
The thread caps are what keep 8 to 16 shards under the container's 2048-process limit;
the script's header says why.
