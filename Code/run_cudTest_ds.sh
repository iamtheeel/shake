#!/bin/bash

export TRITON_CACHE_DIR=$HOME/.local_triton_cache
deepspeed --num_gpus 1 testCuda.py $@
