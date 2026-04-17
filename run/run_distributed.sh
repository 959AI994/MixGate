#!/bin/bash

# Distributed training launcher
# Usage: bash run_distributed.sh [NUM_GPUS] [extra args...]

# Default: 8 GPUs
NUM_GPUS=${1:-8}

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use 8 GPUs
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Start distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_staged.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --num_workers 8 \
    ${@:2}

