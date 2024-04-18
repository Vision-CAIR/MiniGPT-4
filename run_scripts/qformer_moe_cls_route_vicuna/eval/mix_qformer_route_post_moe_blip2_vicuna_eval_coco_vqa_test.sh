GPUS_PER_NODE=2
export MASTER_PORT=8526
export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix_vqa_coco_vicuna_eval_coco_vqa_test.yaml

