# python minigpt4/eval_scripts/eval_vqa.py \
#     --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_vicuna_cls_route_uni_caption_eval_moe.yaml\
#     --device 4

python minigpt4/eval_scripts/eval_vqa_cap.py \
    --cfg-path minigpt4/projects/qformer_moe_cls_vicuna/eval/mix6_vicuna_cls_route_uni_caption_eval_moe.yaml\
    --device 4 \
    --dataset 'nocap,vsr,hm,vizwiz'