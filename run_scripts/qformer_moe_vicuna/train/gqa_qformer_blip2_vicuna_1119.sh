GPUS_PER_NODE=4
export MASTER_PORT=8526
export CUDA_VISIBLE_DEVICES=0,2,6,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} train.py --cfg-path minigpt4/projects/qformer_moe/train/gqa_raw_qformer_blip2_vicuna7b.yaml