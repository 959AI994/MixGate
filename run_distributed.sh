#!/bin/bash

# 分布式训练启动脚本
# 使用方法: bash run_distributed.sh [GPU数量] [其他参数]

# 默认使用8个GPU
NUM_GPUS=${1:-8}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用8张GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_staged.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --num_workers 8 \
    ${@:2}

