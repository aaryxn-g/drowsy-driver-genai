#!/usr/bin/env bash
# Survey the GPU server. Read-only: installs nothing, changes nothing.
#
#   bash scripts/server_survey.sh
#
# Writes server_report.txt in the current directory. Send me that file.

set -uo pipefail
REPORT="server_report.txt"
: > "$REPORT"

say() { echo "$@" | tee -a "$REPORT"; }
run() {
  local label="$1"; shift
  say ""
  say "--- $label"
  local out
  if out=$("$@" 2>&1); then
    if [ -n "$out" ]; then printf '%s\n' "$out" | sed 's/^/    /' | tee -a "$REPORT"; fi
  else
    say "    (command failed or not available)"
  fi
}

say "================================================================"
say " GPU server survey"
say " host    : $(hostname 2>/dev/null || echo unknown)"
say " user    : $(whoami 2>/dev/null || echo unknown)"
say " date    : $(date -u 2>/dev/null) UTC"
say "================================================================"

# ----------------------------------------------------------------- GPUs
say ""
say "### GPUs"
if command -v nvidia-smi >/dev/null 2>&1; then
  run "nvidia-smi header (driver and CUDA version)" bash -c "nvidia-smi | head -12"
  run "per-GPU memory" nvidia-smi \
      --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv
  run "processes currently on the GPUs" nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  say ""
  if [ "$busy" -gt 0 ]; then
    say "    NOTE: $busy process(es) are using the GPUs right now."
    say "    Someone else may be mid-run. Check before claiming a GPU."
  else
    say "    No GPU processes running at this moment."
  fi
else
  say "    nvidia-smi NOT FOUND. Either this is a login node with no GPU,"
  say "    or the drivers are not on PATH. This matters: check before planning."
fi

# ------------------------------------------------------------------ CPU
say ""
say "### CPU and memory"
say "    cores: $(nproc 2>/dev/null || echo unknown)"
run "model" bash -c "lscpu 2>/dev/null | grep -E 'Model name|Socket|Thread|Core' || true"
run "memory" free -g
run "current load (1/5/15 min averages)" uptime

# ----------------------------------------------------------------- disk
say ""
say "### Disk"
run "home" df -h "$HOME"
run "root" df -h /
run "quota, if one is set" quota -s
say ""
say "    Large scratch or data volumes, if any:"
df -h 2>/dev/null | awk 'NR==1 || /scratch|data|work|mnt|storage/' | tee -a "$REPORT"

# ------------------------------------------------------------ scheduler
say ""
say "### Job scheduler"
found_sched=0
for cmd in sbatch squeue sinfo qsub bsub; do
  if command -v "$cmd" >/dev/null 2>&1; then
    say "    found: $cmd"
    found_sched=1
  fi
done
if [ "$found_sched" -eq 0 ]; then
  say "    None found. This is a free-for-all machine:"
  say "    run nvidia-smi before every GPU job and pin with CUDA_VISIBLE_DEVICES."
else
  run "queue state" bash -c "squeue 2>/dev/null | head -20 || true"
fi

# -------------------------------------------------------------- toolchain
say ""
say "### Toolchain"
for cmd in python3 conda mamba git tmux screen wget curl unzip aria2c rsync; do
  if command -v "$cmd" >/dev/null 2>&1; then
    say "    $(printf '%-8s' "$cmd") $(command -v "$cmd")"
  else
    say "    $(printf '%-8s' "$cmd") MISSING"
  fi
done
run "python version" python3 --version
run "conda version" conda --version

# -------------------------------------------------------------- network
say ""
say "### Internet access"
say "    (needed to pip install, and to download the dataset)"
probe() {
  local name="$1" url="$2"
  if curl -sS -I --max-time 15 "$url" >/dev/null 2>&1; then
    say "    OK      $name"
  else
    say "    BLOCKED $name"
  fi
}
if command -v curl >/dev/null 2>&1; then
  probe "pypi.org (pip installs)"            "https://pypi.org/simple/"
  probe "github.com (code and models)"       "https://github.com/"
  probe "drive.google.com (UTA-RLDD)"        "https://drive.google.com/"
  probe "storage.googleapis.com (MediaPipe model)" "https://storage.googleapis.com/"
else
  say "    curl missing, cannot test"
fi
say ""
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy; do
  [ -n "${!v:-}" ] && say "    proxy env: $v=${!v}"
done

say ""
say "================================================================"
say " Written to $REPORT"
say " Send that file back. Nothing was installed or changed."
say "================================================================"
