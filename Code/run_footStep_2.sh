#!/usr/bin/env bash
set -euo pipefail

# Your existing settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.local_triton_cache}"

# Pick the GPU with the most free memory (physical ID)
read -r GPUID _ < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
                    | awk -F',' '{gsub(/ /,""); free=$2-$3; print $1, free}' \
                    | sort -k2,2nr | head -n1)

# Map your job's logical cuda:0 to that physical GPU
export CUDA_VISIBLE_DEVICES="$GPUID"

echo "=== GPU picker ==="
echo "Host: $(hostname)"
echo "Chosen physical GPU: $GPUID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$GPUID" || true
echo "==================="

# IMPORTANT: Pin DeepSpeed to the same physical GPU explicitly
deepspeed --include "localhost:$GPUID" --num_gpus 1 main.py "$@"
