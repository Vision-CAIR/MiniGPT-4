GPUS_PER_NODE=2
WORKER_CNT=1
MASTER_PORT=25647
export CUDA_VISIBLE_DEVICES=5,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} evaluate.py --cfg-path minigpt4/projects/prompt_moe/eval/gqa_llava_prompt_flant5xxl_eval_1012.yaml