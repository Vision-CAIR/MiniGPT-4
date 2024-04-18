GPUS_PER_NODE=2
export MASTER_PORT=8524
export CUDA_VISIBLE_DEVICES=2,5
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_post_vicuna/eval/vqav2_okvqa_gqa_evaluation.yaml