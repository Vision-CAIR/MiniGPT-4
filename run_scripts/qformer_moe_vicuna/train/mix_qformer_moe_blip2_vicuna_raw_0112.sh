GPUS_PER_NODE=8
export MASTER_PORT=8526
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} train.py --cfg-path minigpt4/projects/qformer_moe_vicuna/train/mix_qformer_moe_blip2_vicuna7b_data_raw_0112.yaml
