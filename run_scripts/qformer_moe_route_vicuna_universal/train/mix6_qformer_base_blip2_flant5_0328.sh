GPUS_PER_NODE=3
export MASTER_PORT=8570
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} train.py --cfg-path minigpt4/projects/qformer_moe_route_universal_vicuna/train/mix6_base_flant5xxl_qformer_blip2.yaml

