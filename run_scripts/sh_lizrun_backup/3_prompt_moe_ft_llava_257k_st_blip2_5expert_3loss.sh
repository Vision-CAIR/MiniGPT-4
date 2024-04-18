#!/bin/bash

# start: lizrun start -c "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/3_prompt_moe_ft_llava_257k_st_blip2_5expert_3loss.sh" -n 1 -j blip2-moe-1012  -t nvidia-a800-sxm4-80gb -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-nccl -p default
# PATH_ORI=${0%/*}
# PROJECT_PATH=$(echo ${PATH_ORI} | sed -r 's/\/{2,}/\//')
# echo "========"
# echo $PROJECT_PATH
PROJECT_PATH=/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS
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
  qformer_text_input: True

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
  repeat_to_init_qt_candidates: True
  num_qt_candidates: 5
  eval_gate_save: True
  train_gate_save: True
  gate_save_path: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/prompt_moe/llava_st_257k_raw_train_qf_train_qt_linear_gate_textt5_5ex_2loss_textinqf_epo3_1012/"

datasets:
  llava150k_en_sft:
    type: prompt_moe
    vis_processor:
      train:
        name: "blip_image_train"
        image_size: 224
      eval:
        name: "blip_image_eval"
        image_size: 224
    text_processor:
      train:
        name: "blip_question"
      eval:
        name: "blip_question"
    build_info:
      images:
        storage: "/mnt/pfs-guan-ssai/nlu/dingyifeng/data/COCO"
    

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
  batch_size_train: 16
  batch_size_eval: 32
  num_workers: 4
  warmup_steps: 600

  seed: 42
  output_dir: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/prompt_moe/llava_st_257k_raw_train_qf_train_qt_linear_gate_textt5_5ex_2loss_textinqf_epo3_1012/"
  
  amp: True
  resume_ckpt_path: null

  evaluate: False 
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