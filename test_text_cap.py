from minigpt4.common.caption_tools.caption_utils import coco_caption_eval, textcaps_caption_eval
result_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_12epo_0317/20240317165/result/val_vqa_result_text_cap.json"
annotaion_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/TextCap/TextCaps_0.1_val.json"
eval = textcaps_caption_eval(annotaion_file, result_file)
print(eval.eval.item())