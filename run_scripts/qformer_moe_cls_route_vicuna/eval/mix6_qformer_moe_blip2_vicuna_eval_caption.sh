GPUS_PER_NODE=6
export MASTER_PORT=8546
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/4_mix6_post_route_uni_caption_eval.yaml


# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/4_mix6_post_caption_eval.yaml

# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_cls_route_uni_caption_eval.yaml

# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/4_mix6_cls_route_3ex_caption_eval.yaml
# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/4_mix6_cls_route_3ex_uni_caption_eval.yaml
# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/4_mix6_cls_route_8ex_uni_caption_eval.yaml

# visualization expert seperate
GPUS_PER_NODE=6
export MASTER_PORT=8546
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_cls_route_eval.yaml

# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/4_mix6_cls_route_4ex_caption_eval.yaml

# GPUS_PER_NODE=6
# export MASTER_PORT=8546
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/4_mix6_cls_caption_eval.yaml

# GPUS_PER_NODE=6
# export MASTER_PORT=8546
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/4_mix6_pre_caption_eval.yaml
