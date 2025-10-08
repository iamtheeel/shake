#!/usr/bin/env bash
# run_main.sh — pick a free GPU, then launch your job
# Submit exactly as you do now:
#   sbatch -p gpucluster run_main.sh [your args...]

set -euo pipefail

# --- Your existing settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.local_triton_cache}"

# --- If Slurm already assigned a specific GPU, honor it and skip probing.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CHOSEN_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
else
  # Probe with nvidia-smi and pick the GPU with the most free MiB
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU node?" >&2
    exit 2
  fi
  # Query: index, total, used (MiB). Compute free and pick max.
  read -r CHOSEN_GPU _ < <(
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,""); free=$2-$3; print $1, free}' \
    | sort -k2,2nr \
    | head -n1
  )
  export CUDA_VISIBLE_DEVICES="$CHOSEN_GPU"
fi

echo "=== GPU picker ==="
echo "Host: $(hostname)"
echo "Chosen physical GPU: ${CHOSEN_GPU}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
command -v nvidia-smi >/dev/null && nvidia-smi -i "${CHOSEN_GPU}" || true
echo "==================="

# Launch DeepSpeed on logical device 0 (which maps to the chosen physical GPU)
deepspeed --num_gpus 1 main.py "$@"
