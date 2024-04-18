#!/bin/bash

# start: lizrun start -c "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/6_vicuna7b_qformer_moe_ft_llava_mix_blip2_5expert_top1_3loss.sh" -n 1 -j blip2-vi-qfmoe-mixto-5e1-1128 -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-nccl -p default
# if -g 4 , adjust nproc_per_node=4
# PATH_ORI=${0%/*}
# PROJECT_PATH=$(echo ${PATH_ORI} | sed -r 's/\/{2,}/\//')
# echo "========"
# echo $PROJECT_PATH
PROJECT_PATH=/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE
cd ${PROJECT_PATH}
# pip install -e .

### RDMA Config ###
# export NCCL_IB_HCA=^mlx5_0,mlx5_1,^mlx5_2,mlx5_3,^mlx5_4,mlx5_5,^mlx5_6,mlx5_7,^mlx5_8
export NCCL_IB_GID_INDEX=3
### RDMA Config ###

pip install -r requirements.txt

# DNS to IP
# sleep 30 # waiting for system init
MASTER_IP=""
if [ "${RANK}" == "0" ];then
  while [[ "$MASTER_IP" == "" ]]
  do
    MASTER_IP=`ping ${MASTER_ADDR} -c 3 | sed '1{s/[^(]*(//;s/).*//;q}'`
    # MASTER_IP=127.0.0.1
    sleep 1
  done
else
  ## Convert DNS to IP for torch
  MASTER_IP=`getent hosts ${MASTER_ADDR} | awk '{print $1}'` # Ethernet
fi

# training cofiguration
CONFIG_FILE=/tmp/blip2_config_${RANK}.yaml
# WORLD_SIZE=`expr ${WORLD_SIZE} \* 8`
DIST_URL="env://${MASTER_IP}:${MASTER_PORT}"
# 配置生成
cat <<EOT > ${CONFIG_FILE}
model:
  arch: blip2_vicuna_instruct
  model_type: vicuna7b_pretrain
  load_pretrained: True
  load_finetuned: False
  vit_model: eva_clip_g
  pretrained: "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2_vicuna7b/blip2_pretrained_vicuna7b.pth"
  # finetuned: ""
  q_former_model: "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2_vicuna7b/blip2_pretrained_vicuna7b.pth"

  # vit encoder
  image_size: 224
  drop_path_rate: 0
  use_grad_checkpoint: False
  vit_precision: "fp16"

  # Q-Former
  num_query_token: 32
  qformer_text_input: True

  # vicuna7b
  llm_model: "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/vicuna-7b-v1.1"
  prompt: ""
  max_txt_len: 256
  max_output_txt_len: 256

  # freeze
  freeze_vit: True
  freeze_llm: True
  freeze_qformer: False
  freeze_t5_proj: False

  # moe
  use_moeqformer: True
  moebert_expert_num: 5
  moebert_route_method: "gate-sentence"
  moebert_load_balance: 0.00
  moe_topk: 1
  use_balance_loss: False

datasets:
  llava_mix: # train: 1385835
    type: mix_coco_gqa
    batch_size: 16
    vis_processor:
      train:
        name: "blip2_image_train"
        image_size: 224
      eval:
        name: "blip2_image_eval"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"
      eval:
        name: "blip_caption"
  
  gqa:
    type: balanced_sft_raw_eval
    batch_size: 16
    vis_processor:
      eval:
        name: "blip2_image_eval"
        image_size: 224
    text_processor:
      eval:
        name: "blip_caption"

  ok_vqa:
    type: ok_vqa_eval
    batch_size: 16
    vis_processor:
      eval:
        name: "blip2_image_eval"
        image_size: 224
    text_processor:
      eval:
        name: "blip_caption"
  
  coco_vqa:    # 658104
    type: vqa_v2_eval
    batch_size: 16
    vis_processor:
      eval:
        name: "blip2_image_eval"
        image_size: 224
    text_processor:
      eval:
        name: "blip_caption"

run:
  task: instruction_tuning
  # optimizer
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 2e-5
  min_lr: 1e-6
  warmup_lr: 1e-6
  log_freq: 5
  save_freq: 1500

  weight_decay: 0.05
  max_epoch: 3
  num_workers: 4
  warmup_steps: 600

  seed: 42
  output_dir: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe/mix_coco_gqa_together_raw_QformerMoE_train_qf_train_qt_linear_gate_loss1_5ex_top1_textinqf_epo3_1128/"
  
  amp: True
  resume_ckpt_path: null

  evaluate: False 
  train_splits: ["train"]
  valid_splits: ["val"]

  device: "cuda"
  world_size: ${WORLD_SIZE}
  dist_url: ${DIST_URL}
  distributed: True
EOT


torchrun --nnodes=${WORLD_SIZE}  --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP} \
  train.py \
  --cfg-path ${CONFIG_FILE}
