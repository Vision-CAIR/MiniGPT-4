GPUS_PER_NODE=2
export MASTER_PORT=8546
export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_cls_route_uni_caption_eval1.yaml
