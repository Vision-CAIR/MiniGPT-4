GPUS_PER_NODE=2
export MASTER_PORT=8520
export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} train.py --cfg-path minigpt4/projects/prompt_moe/train/mix_llava_665k_prompt_moe_post_blip2.yaml