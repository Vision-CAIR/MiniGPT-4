#!/bin/bash

# start: lizrun start -c "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/8_vicuna7b_qformer_moe_route_post_ft_mix_aokvqa_cap_blip2_3expert_3beam_2loss.sh" -n 1 -j r-post-lora-mix5-3e3b-15ep-0309 -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-nccl -p basev
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
  lora_r: 64 # freeze_llm = False
  lora_alpha: 16
  prompt: ""
  max_txt_len: 256
  max_output_txt_len: 256

  # freeze
  freeze_vit: True
  freeze_llm: False
  freeze_qformer: False
  freeze_t5_proj: False

  # moe
  use_moeqformer: True
  use_route_moe: True
  moebert_route_method: "post-route"
  moebert_load_balance: 0.05
  moebert_expert_num: 3
  moebert_num_beams: 3
  moe_weight_type: 'ffn_prob'
  use_balance_loss: False
  ln_position: "out"

datasets:
  gqa: # train: 943000, 12578, 12578)
    type: balanced_sft_raw
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
    sample_ratio: 10
  
  ok_vqa: # train, valid (9009, 5046)
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
    sample_ratio: 1
  
  coco_vqa:    # 658104
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
    sample_ratio: 9

  coco_caption: # 414113 train
    batch_size: 32
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
    sample_ratio: 7

  aok_vqa: # train: 17056, val: 1145
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
    sample_ratio: 2

run:
  task: instruction_tuning
  # optimizer
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 5e-5
  min_lr: 1e-6
  warmup_lr: 1e-6
  log_freq: 5
  save_freq: 1500

  weight_decay: 0.05
  max_epoch: 15
  num_workers: 4
  warmup_steps: 600
  iters_per_epoch: 5000

  seed: 42
  output_dir: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_route/lora_mix_coco_gqa_ao_cap_raw_QformerMoE_Post_Route_linear_gate_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_15epo_0309/"
  
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
