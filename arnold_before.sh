#!/bin/bash
# This script is for 1) Install dependancies; 2) Align internal cluster with standard practice

# Pip install 
# export http_proxy=10.20.47.147:3128  https_proxy=10.20.47.147:3128 no_proxy=code.byted.org
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install byted-dataloader -i "https://bytedpypi.byted.org/simple"
pip3 install mmmengine==0.7.3
pip3 install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# unset http_proxy && unset https_proxy && unset no_proxy

# # ----------------------------------------------------------------------------------------
# # setup environment variables
# # disable TF verbose logging
# TF_CPP_MIN_LOG_LEVEL=2
# # fix known issues for pytorch-1.5.1 accroding to 
# # https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
# MKL_THREADING_LAYER=GNU
# # set NCCL envs for disributed communication
# NCCL_IB_GID_INDEX=3
# NCCL_IB_DISABLE=0
# NCCL_DEBUG=INFO
# ARNOLD_FRAMEWORK=pytorch
# # get distributed training parameters 
# METIS_WORKER_0_HOST=${METIS_WORKER_0_HOST:-"127.0.0.1"}
# NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# ARNOLD_WORKER_GPU=${ARNOLD_WORKER_GPU:-$NV_GPUS}
# ARNOLD_WORKER_NUM=${ARNOLD_WORKER_NUM:-1}
# ARNOLD_ID=${ARNOLD_ID:-0}
# ARNOLD_PORT=${METIS_WORKER_0_PORT:-3343}


# export NNODES=$ARNOLD_WORKER_NUM
# export NODE_RANK=$ARNOLD_ID
# export MASTER_ADDR=$METIS_WORKER_0_HOST
# export MASTER_PORT=$ARNOLD_PORT
# export GPUS=$ARNOLD_WORKER_GPU
