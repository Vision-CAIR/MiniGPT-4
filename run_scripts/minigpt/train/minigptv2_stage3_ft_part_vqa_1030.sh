GPUS_PER_NODE=1
export MASTER_PORT=8517
export CUDA_VISIBLE_DEVICES=3
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} train.py --cfg-path minigpt4/projects/minigpt/train/minigptv2_finetune_vqa.yaml