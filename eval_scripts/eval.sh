#!/bin/bash --login
cfg_path=eval_configs/minigpt4_llama2_eval.yaml
CKPT=YOUR_CKPT_PATH
NAME=EXP_NAME
IMG_PATH=YOUR_IMG_PATH
EVAL_FILE_PATH=YOUR_EVAL_FILE_PATH

torchrun --nproc_per_node 1 eval_ref.py --name ${NAME} \
 --cfg-path ${cfg_path} \
 --ckpt ${CKPT} --dataset refcoco,refcoco+,refcocog --lora_r 64 --lora_alpha 16 \
 --batch_size 64 --max_new_tokens 20 --resample --img_path ${IMG_PATH} --eval_file_path ${EVAL_FILE_PATH}
 
torchrun --nproc_per_node 1 eval_vqa.py --name ${NAME} \
 --cfg-path ${cfg_path} \
 --ckpt ${CKPT} --split val,test --dataset okvqa,vizwiz,aokvqa,iconqa,gqa,vsr,hm --lora_r 64 --lora_alpha 16 \
 --batch_size 32 --max_new_tokens 20 --resample

torchrun --master-port ${PORT} --nproc_per_node 1 run_textvqa_eval.py --name ${NAME} --ckpt_path ${CKPT} --lora_r 64 --lora_alpha 16 --eval_file_path ${EVAL_FILE_PATH}
torchrun --master-port ${PORT} --nproc_per_node 1 run_sciencevqa_eval.py --name ${NAME} --ckpt_path ${CKPT} --lora_r 64 --lora_alpha 16
