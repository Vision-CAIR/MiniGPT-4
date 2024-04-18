GPUS_PER_NODE=4
export MASTER_PORT=8546
export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_route_vicuna/eval/mix_vqa_coco_vicuna_eval.yaml
GPUS_PER_NODE=4
export MASTER_PORT=8545
export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_route_vicuna/eval/mix_vqa_coco_vicuna_eval_copy.yaml

# export CUDA_VISIBLE_DEVICES=3
# python evaluate.py --cfg-path minigpt4/projects/qformer_moe_route_vicuna/eval/mix_vqa_coco_vicuna_eval.yaml
