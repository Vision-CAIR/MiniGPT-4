GPUS_PER_NODE=2
export MASTER_PORT=8527
export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py  --cfg-path minigpt4/projects/qformer_moe_post_vicuna/eval/mix_qformer_moe_post_vicuna7b_evaluation2.yaml