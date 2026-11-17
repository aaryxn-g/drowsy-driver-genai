#!/usr/bin/env bash
# Create the project environment on the GPU server and verify it.
#
#   bash scripts/server_setup.sh
#
# Run scripts/server_survey.sh first. Writes setup_report.txt.
#
# Creates a conda env named `drowsy` with python 3.11, installs the project
# requirements, installs a torch build matching this server's CUDA, and runs
# the test suite. Nothing outside the conda env and this repo is touched.

set -uo pipefail
ENV_NAME="${ENV_NAME:-drowsy}"
REPORT="setup_report.txt"
: > "$REPORT"
say() { echo "$@" | tee -a "$REPORT"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

say "=== environment setup ==="
say "repo : $REPO_ROOT"
say "env  : $ENV_NAME"
say ""

# ------------------------------------------------------------------ conda
if ! command -v conda >/dev/null 2>&1; then
  say "ERROR: conda not found."
  say "Either load it (try: module avail conda) or install miniconda to \$HOME."
  say "Do not install system-wide on a shared machine."
  exit 1
fi
eval "$(conda shell.bash hook)" 2>/dev/null || {
  say "ERROR: could not initialise conda for this shell."; exit 1; }

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "env '$ENV_NAME' already exists, reusing it"
else
  say "creating env '$ENV_NAME' (python 3.11)"
  conda create -n "$ENV_NAME" python=3.11 -y >>"$REPORT" 2>&1 || {
    say "ERROR: env creation failed, see $REPORT"; exit 1; }
fi
conda activate "$ENV_NAME" || { say "ERROR: could not activate $ENV_NAME"; exit 1; }
say "python: $(python --version 2>&1)  at $(command -v python)"

# --------------------------------------------------- torch matching CUDA
say ""
say "=== torch ==="
CUDA_VER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
fi

if [ -z "$CUDA_VER" ]; then
  say "No CUDA version detected. Installing the default (likely CPU-only) torch."
  say "If this server does have a GPU, stop and investigate before training."
  TORCH_INDEX=""
else
  say "driver reports CUDA $CUDA_VER"
  MAJ=${CUDA_VER%%.*}; MIN=${CUDA_VER##*.}
  if   [ "$MAJ" -ge 13 ];                             then TAG="cu128"
  elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ];         then TAG="cu126"
  elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 4 ];         then TAG="cu124"
  elif [ "$MAJ" -eq 12 ];                             then TAG="cu121"
  elif [ "$MAJ" -eq 11 ] && [ "$MIN" -ge 8 ];         then TAG="cu118"
  else TAG=""; fi
  if [ -n "$TAG" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/$TAG"
    say "selecting torch build: $TAG"
  else
    TORCH_INDEX=""
    say "CUDA $CUDA_VER is older than any current torch wheel; using the default."
  fi
fi

if [ -n "$TORCH_INDEX" ]; then
  pip install --quiet torch torchvision --index-url "$TORCH_INDEX" >>"$REPORT" 2>&1 \
    || say "WARNING: torch install from $TORCH_INDEX failed, see $REPORT"
else
  pip install --quiet torch torchvision >>"$REPORT" 2>&1 \
    || say "WARNING: torch install failed, see $REPORT"
fi

# ----------------------------------------------------------- requirements
say ""
say "=== project requirements ==="
pip install --quiet -r phase1_baseline/requirements.txt >>"$REPORT" 2>&1 \
  || say "WARNING: some requirements failed, see $REPORT"

# ---------------------------------------------------------------- verify
say ""
say "=== verification ==="
python - <<'PY' 2>&1 | tee -a "$REPORT"
import importlib
for mod in ("numpy", "pandas", "cv2", "mediapipe", "pyarrow", "scipy", "sklearn"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok      {mod:<12} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  MISSING {mod:<12} {type(e).__name__}: {e}")

try:
    import torch
    print(f"  ok      torch        {torch.__version__}")
    print(f"          cuda built   {torch.version.cuda}")
    print(f"          cuda usable  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"          gpu {i}        {p.name}, {p.total_memory/1e9:.1f} GB")
    else:
        print("          >>> torch cannot see a GPU. If this server has one,")
        print("          >>> the wrong wheel was installed. Do not train yet.")
except Exception as e:
    print(f"  MISSING torch        {type(e).__name__}: {e}")
PY

# ------------------------------------------------------------------ tests
say ""
say "=== test suite (should pass clean) ==="
cd phase1_baseline || { say "ERROR: phase1_baseline not found"; exit 1; }
python -m pytest tests -q 2>&1 | tail -20 | tee -a "../$REPORT"
cd "$REPO_ROOT" || exit 1

say ""
say "================================================================"
say " Written to $REPORT. Send that file back."
say ""
say " Activate with:  conda activate $ENV_NAME"
say "================================================================"
