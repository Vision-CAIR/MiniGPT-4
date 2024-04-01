import json
import os
import pandas as pd
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_route/mix_coco_gqa_ao_cocap_tcap_raw_QformerMoE_Post_Route_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_15epo_0314/20240314230/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_route/mix_coco_gqa_ao_cocap_tcap_raw_Qformer_base_lr5e5_10epo_0316/20240316192"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0317/20240317165/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_12epo_0317/20240317165/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0318/20240318145/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_tcap_raw_Qformer_base_lr5e5_10epo_0318/20240318113"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_4ex4b_2loss_005_top6layer_textinqf_10epo_0319/20240319110"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_4ex4b_2loss_005_top6layer_textinqf_10epo_0319/20240319110/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr6e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0319/20240319105"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_tcap_raw_Qformer_base_lr5e5_10epo_0320_instruct/20240320230/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr3e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0326/20240326134"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/base/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_lr5e5_top6layer_textinqf_8epo_0328/20240328100/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_8epo_0328/20240328164"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_base_lr5e5_textinqf_8epo_0329/20240329110"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Route_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_8epo_0330/20240329232/"

# mix5
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_raw_QformerMoE_CLS_Gate_Query_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330211/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_raw_QformerMoE_CLS_Query_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330210/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_raw_QformerMoE_CLS_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330211/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_raw_QformerMoE_CLS_Cross_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330183/"

# mix6 3ex3b uni 005
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Gate_Query_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330211/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Query_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330182/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330173/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330173/"

# mix6 3ex3b uni 001
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_UNI_CLS_Gate_Route_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_10epo_0330/20240330221/"

# mi6 non-universal
# cls-cross-route-mix6-001-3ex3b-0330-wanghanzi-master-0
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_10epo_0330/20240330221/"
# cls-cross-route-mix6-001-4ex4b-wanghanzi-master-0
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331213/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_query_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331214/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Route_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331214/"

# mix6 4ex4b uni 001
path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Base_textinqf_10epo_0331/20240331132/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0330/20240330222/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Query_Universal_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331144/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Route_Universal_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331143/"

# cls cross
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr4e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331215/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr6e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331215/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr5e5_6ex6b_2loss_001_top6layer_textinqf_10epo_0331/20240331213/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr5e5_8ex8b_2loss_001_top6layer_textinqf_10epo_0331/20240331213/"

# 8ex8b uni
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Route_Universal_lnout_lr5e5_8ex8b_2loss_001_top6layer_textinqf_10epo_0331/20240331150/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Query_Universal_lnout_lr5e5_8ex8b_2loss_001_top6layer_textinqf_10epo_0331/20240331213/"

# post
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_10epo_0331/20240331131/"
modes = ['gqa','okvqa','vqav2','aokvqa']
file_name = dict()
data = dict()
for mode in modes:
    file_name[mode] = os.path.join(path, f"evaluate_{mode}.txt")
    tmp = list()
    with open(file_name[mode], "r") as f:
        for line in f: 
            tmp.append(json.loads(line))
    accs = [dd['agg_metrics'] for dd in tmp]
    data[mode] = accs
print(data)
df = pd.DataFrame(data)
# calculate average score
df['avg'] = df.mean(axis=1)
name = path.split('/')[-3]
df.to_csv(f"/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/0401_results_{name}.csv",index=False)
print("save to /mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/0401_results_{}.csv".format(name))
print(df)
modes = ['coco_cap','text_cap']
for mode in modes:
    file_name[mode] = os.path.join(path, f"evaluate_{mode}.txt")
    tmp = list()
    with open(file_name[mode], "r") as f:
        for line in f: 
            tmp.append(json.loads(line)["val"])
    df1 = pd.DataFrame(tmp)
    df1.to_csv("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/caption_result/results_{}_{}.csv".format(mode,name),index=False)
    print("\n",df1)

    

