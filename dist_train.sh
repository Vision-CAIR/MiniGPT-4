#!/bin/bash
set -x

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS=${GPUS:-${ARNOLD_WORKER_GPU}}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9909}
# ARNOLD_WORKER_0_PORT
# ARNOLD_WORKER_0_ADDR

# settings for torch log
export BYTED_TORCH_FX=O1
export BYTED_TORCH_BYTECCL=O1
export TOKENIZERS_PARALLELISM=false
export HADOOP_ROOT_LOGGER=error,console

# settings for DDP multi-node for lab.pytorch image >= 1.13
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SHM_DISABLE=0

# start training
CONFIG=$1
torchrun --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --nproc_per_node=$GPUS \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         train.py \
         --cfg-path \
         $CONFIG \
         ${@:2}
