GPUS_PER_NODE=2
export MASTER_PORT=8530
export CUDA_VISIBLE_DEVICES=3,4
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} train.py --cfg-path minigpt4/projects/qformer_moe/train/gqa_qformer_moe_blip2_short_answer_224.yaml