# python minigpt4/eval_scripts/eval_vqa.py \
#     --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_flant5_cls_route_uni_caption_eval_moe.yaml\
#     --device 6

python minigpt4/eval_scripts/eval_vqa_cap.py \
    --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_flant5_cls_route_uni_caption_eval_moe.yaml\
    --device 6 \
    --dataset ['nocap']