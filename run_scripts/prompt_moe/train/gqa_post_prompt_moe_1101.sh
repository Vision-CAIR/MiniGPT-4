GPUS_PER_NODE=1
export MASTER_PORT=8520
export CUDA_VISIBLE_DEVICES=4
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} train.py --cfg-path minigpt4/projects/prompt_moe/train/gqa_sft_post_prompt_moe.yaml