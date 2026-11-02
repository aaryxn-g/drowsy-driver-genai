# Server scripts

Run these on the GPU server, from the repo root, in a VS Code Remote-SSH terminal.

```bash
bash scripts/server_survey.sh    # read-only. writes server_report.txt
bash scripts/server_setup.sh     # creates the conda env. writes setup_report.txt
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
installs the project requirements, and runs the 54-test suite. Everything stays inside
the conda env and this repo.

Override the env name if `drowsy` is taken on a shared account:

```bash
ENV_NAME=aaryan_drowsy bash scripts/server_setup.sh
```
