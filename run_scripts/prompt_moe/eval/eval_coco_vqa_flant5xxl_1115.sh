GPUS_PER_NODE=4
MASTER_PORT=25647
export CUDA_VISIBLE_DEVICES=0,2,6,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} evaluate.py --cfg-path minigpt4/projects/prompt_moe/eval/vqa_coco_flant5xxl_eval_1115.yaml