#!/usr/bin/env bash
# Launch the Phase 1.1 extraction (pass 1) as N parallel shards, then merge the logs.
#
#   scripts/run_extraction.sh [N] [extra src.extract args...]
#   ROOT=/path/to/UTA-RLDD PYTHON=~/miniconda3/envs/drowsy/bin/python scripts/run_extraction.sh 8
#
# Run it inside tmux: it waits for every shard, which takes a while. Resumable, since
# a video whose parquet exists is skipped, so re-running after a crash only does
# what is left. Per-shard output goes to phase1_baseline/logs/shard<i>of<N>.out.
#
# Why the thread caps below are not optional:
# ffmpeg's h264 decoder starts one thread per core by default. This box has 64
# cores, and the container's pids.max is 2048 (threads count against it). 16 shards
# with uncapped decoders, plus OpenMP/BLAS pools of their own, exhaust that limit,
# and every shard dies with "Could not open codec h264, error: -11" (EAGAIN from
# thread creation) before reading a single frame. One decoder thread per shard is
# enough, because the landmarker, not decoding, is the bottleneck; parallelism comes
# from the shards.
set -euo pipefail

N="${1:-8}"
shift || true
ROOT="${ROOT:-$HOME/data/UTA-RLDD}"
PYTHON="${PYTHON:-python}"
STAGGER_S="${STAGGER_S:-2}"

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
  echo "N must be a positive integer, got '$N'" >&2
  exit 2
fi

export OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;1"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$(dirname "$0")/../phase1_baseline"
[ -d "$ROOT" ] || { echo "dataset root not found: $ROOT (set ROOT=...)" >&2; exit 1; }
[ -f data/manifest.csv ] || { echo "no data/manifest.csv; run: $PYTHON -m src.manifest --root $ROOT" >&2; exit 1; }
mkdir -p logs data/frames

echo "pids.max: $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo unknown)   cores: $(nproc)"
echo "starting $N shards over $(($(wc -l < data/manifest.csv) - 1)) manifest rows, root $ROOT"

pids=()
for i in $(seq 1 "$N"); do
  "$PYTHON" -m src.extract --manifest data/manifest.csv --root "$ROOT" --out data/frames \
      --shard "$i/$N" "$@" > "logs/shard${i}of${N}.out" 2>&1 &
  pids+=("$!")
  echo "  shard $i/$N  pid $!  -> logs/shard${i}of${N}.out"
  # Staggered so the shards do not all open the model and their first video at once.
  if [ "$i" -lt "$N" ]; then sleep "$STAGGER_S"; fi
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "shard $((i + 1))/$N exited non-zero; see logs/shard$((i + 1))of${N}.out"
    failed=$((failed + 1))
  fi
done

echo
grep -h "FAILED\|MISSING" logs/shard*of"${N}".out || echo "no FAILED or MISSING lines"
echo
"$PYTHON" -m src.extract --merge-logs --out data/frames --manifest data/manifest.csv || failed=$((failed + 1))
exit $(( failed > 0 ))
