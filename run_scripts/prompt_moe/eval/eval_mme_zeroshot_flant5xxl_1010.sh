GPUS_PER_NODE=1
WORKER_CNT=1
MASTER_PORT=25646
export CUDA_VISIBLE_DEVICES=4
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} evaluate_mme.py --cfg-path minigpt4/projects/prompt_moe/eval/mme_llava_prompt_flant5xxl_eval_1007.yaml