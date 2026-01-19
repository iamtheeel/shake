#!/usr/bin/env bash

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/.local_triton_cache}"
mkdir -p "$TRITON_CACHE_DIR" || true

# --- pick the best physical GPU (most free MiB). require at least MIN_FREE MiB if set.
MIN_FREE_MIB=22000
#MIN_FREE_MIB=44000
MIN_FREE_MIB="${MIN_FREE_MIB:-0}"   # e.g., export MIN_FREE_MIB=60000 to require >= ~60 GiB free

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Are you on a GPU node?" >&2
  exit 2
fi

mapfile -t LINES < <(nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
                                --format=csv,noheader,nounits)

BEST_ID=""
BEST_FREE=-1
for line in "${LINES[@]}"; do
  IFS=',' read -r IDX TOT USED UTIL <<<"$line"
  IDX="${IDX//[[:space:]]/}"; TOT="${TOT//[[:space:]]/}"; USED="${USED//[[:space:]]/}"; UTIL="${UTIL//[[:space:]]/}"
  FREE=$((TOT-USED))
  if (( FREE >= MIN_FREE_MIB )) && (( FREE > BEST_FREE )); then
    BEST_FREE=$FREE
    BEST_ID=$IDX
  fi
done

if [[ -z "${BEST_ID}" ]]; then
  echo "No GPU meets MIN_FREE_MIB=${MIN_FREE_MIB} MiB. Aborting." >&2
  exit 3
fi

# Map logical cuda:0 to the chosen physical GPU
export CUDA_VISIBLE_DEVICES="$BEST_ID"

echo "=== GPU picker ==="
echo "Host: $(hostname)"
echo "Chosen physical GPU: $BEST_ID  (free=${BEST_FREE} MiB)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$BEST_ID" || true
echo "==================="

# IMPORTANT: use --include ONLY (no --num_gpus / --num_nodes), or DS will ignore CUDA_VISIBLE_DEVICES.
deepspeed --include "localhost:${BEST_ID}" main.py "$@"
