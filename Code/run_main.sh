#!/bin/bash
# 🛠 Helps reduce fragmentation by allowing larger memory segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$HOME/.local_triton_cache


deepspeed --num_gpus 1 main.py $@
