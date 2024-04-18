#!/bin/bash

# start: lizrun start -c "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/3_prompt_moe_post_coco_inference_st_blip2_prompt.sh" -n 1 -j blip2-post-moe-gqa-5ex-1027  -t nvidia-a800-sxm4-80gb -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-nccl -p default
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
  arch: blip2_t5_instruct_pro_moe
  model_type: flant5xxl
  load_pretrained: True
  load_finetuned: False
  vit_model: eva_clip_g
  pretrained: "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2-flant5-xxl/blip2_pretrained_flant5xxl.pth"

  # vit encoder
  image_size: 224
  drop_path_rate: 0
  use_grad_checkpoint: False
  vit_precision: "fp16"

  # Q-Former
  num_query_token: 32
  qformer_text_input: False

  # T5
  t5_model: "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/google-flan-t5-xxl"
  prompt: ""
  max_txt_len: 256
  max_output_txt_len: 256

  # freeze
  freeze_vit: True
  freeze_llm: True
  freeze_qformer: False
  freeze_t5_proj: False

  # moe
  moe_position: "pre" # post (position to insert PromptMoE Part)
  embed_extract: "blip2_pretrain" # t5, random (way to extract embeddings of task instruction if moe_position is pre)
  repeat_to_init_qt_candidates: True
  num_qt_candidates: 1
  moe_topk: 1
  eval_gate_save: False
  train_gate_save: False
  gate_save_path: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/prompt_moe/gqa_943k_raw_postMoE_train_qf_train_qt_linear_gate_5ex_top2_3loss_textinqf_epo3_1027/"

datasets:
  gqa: # train: 943000, 12578, 12578)
    type: balanced_sft_raw
    # batch_size: 4
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
    sample_ratio: 50
  
  ok_vqa: # train, valid (9009, 5046)
    # batch_size: 6
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
    sample_ratio: 8  
  
  coco_vqa:    # 658104
    # batch_size: 6
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
    sample_ratio: 15

run:
  task: instruction_tuning
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 2e-5
  min_lr: 1e-6
  warmup_lr: 1e-6
  log_freq: 5
  save_freq: 1500

  weight_decay: 0.05
  max_epoch: 5
  num_workers: 4
  warmup_steps: 600

  max_len: 20
  min_len: 1
  num_beams: 5
  inference_method: "generate"
  prompt: "Question: {} Short answer:"

  seed: 42
  output_dir: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/prompt_moe/mix_coco_okvqa_gqa_raw_inference_textnoinqf_prompt_1102/"
  
  amp: True
  resume_ckpt_path: null

  evaluate: True 
  train_splits: ["train"]
  valid_splits: ["val"]
  test_splits: ["val"]

  device: "cuda"
  world_size: ${WORLD_SIZE}
  dist_url: ${DIST_URL}
  distributed: True
EOT


torchrun --nnodes=${WORLD_SIZE}  --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP} \
  train.py \
  --cfg-path ${CONFIG_FILE}