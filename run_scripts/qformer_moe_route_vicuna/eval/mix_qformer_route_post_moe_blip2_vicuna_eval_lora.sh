# GPUS_PER_NODE=8
# export MASTER_PORT=8524
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_route_vicuna/eval/mix_vqa_coco_vicuna_eval_lora.yaml

export CUDA_VISIBLE_DEVICES=3
python evaluate.py --cfg-path minigpt4/projects/qformer_moe_route_vicuna/eval/mix_vqa_coco_vicuna_eval_lora.yaml
