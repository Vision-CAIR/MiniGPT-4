GPUS_PER_NODE=4
export MASTER_PORT=8546
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix5_cls_route_uni_caption_eval.yaml

