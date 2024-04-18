python minigpt4/eval_scripts/eval_vqa_cap.py \
    --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_vicuna_caption_eval_base.yaml\
    --device 3 \
    --dataset ['nocap','vsr']















# export CUDA_VISIBLE_DEVICES=$device
# export MASTER_PORT=8390
# python minigpt4/eval_scripts/eval_vqa.py \
#     --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_vicuna_cls_route_uni_caption_eval_moe.yaml\
#     --device 4

# export MASTER_PORT=8390
# python minigpt4/eval_scripts/eval_vqa.py \
#     --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_flant5_cls_route_uni_caption_eval_base.yaml\
#     --device 5

# GPUS_PER_NODE=6
# export MASTER_PORT=8546
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_flant5_cls_route_uni_caption_eval_base.yaml


# export MASTER_PORT=8390
# python minigpt4/eval_scripts/eval_vqa.py \
#     --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_flant5_cls_route_uni_caption_eval_moe.yaml\
#     --device 6

# GPUS_PER_NODE=6
# export MASTER_PORT=8546
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} evaluate.py --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_flant5_cls_route_uni_caption_eval_moe.yaml

# # /mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/eval/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_4ex4b_001_lr5e5_top6layer_textinqf_8epo_0404_flant5_ckpt5/20240408232/result/test_vqa_result_vqav2.json