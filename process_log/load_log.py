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
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_raw_QformerMoE_CLS_Cross_Route_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_10epo_0403/20240403183/"

# mix6 3ex3b uni 005
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Gate_Query_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330211/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Query_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330182/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330173/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Route_Universal_lnout_lr5e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0330/20240330173/"

# mix6 3ex3b uni 001
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_UNI_CLS_Gate_Route_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_10epo_0330/20240330221/"

# mi6 non-universal
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_10epo_0330/20240330221/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331213/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_query_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331214/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Route_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331214/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_raw_QformerMoE_CLS_Cross_Route_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_10epo_0403/20240403183/"

# mix6 4ex4b uni 001
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Base_textinqf_10epo_0331/20240331132/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Query_Universal_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331144/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Route_Universal_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331143/"

# # cls cross
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0330/20240330222/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr4e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331215/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr6e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0331/20240331215/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr5e5_6ex6b_2loss_001_top6layer_textinqf_10epo_0331/20240331213/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Universal_lnout_lr5e5_8ex8b_2loss_001_top6layer_textinqf_10epo_0331/20240331213/"
path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_CROSS_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_sead40_0409/20240410101/"
# # 8ex8b uni
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Route_Universal_lnout_lr5e5_8ex8b_2loss_001_top6layer_textinqf_10epo_0331/20240331150/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Query_Universal_lnout_lr5e5_8ex8b_2loss_001_top6layer_textinqf_10epo_0331/20240331213/"

# # post
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_10epo_0331/20240331131/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_raw_QformerMoE_CLS_Cross_Uni_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_8epo_0403/20240403190/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_Route_Universal_lnout_lr5e5_3ex3b_2loss_001_top6layer_textinqf_8epo_0403/20240403191/"

## greedy search
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_lnout_lr5e5_4extop1_2loss_001_top6layer_textinqf_10epo_0404/20240404154/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Pre_lnout_lr5e5_4extop1_2loss_001_top6layer_textinqf_10epo_0404/20240404154/"
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_lnout_lr5e5_4extop1_2loss_001_top6layer_textinqf_10epo_0404/20240404154/"


# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_raw_QformerMoE_Post_Route_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0404/20240404155/"

# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0408/20240408223/"

# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_mode_cls/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_PRE_lnout_lr5e5_4ex4b_2loss_001_top6layer_textinqf_10epo_0408/20240408223/"
## flant5
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/uni_route//mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_CLS_Cross_4ex4b_001_lr5e5_top6layer_textinqf_8epo_0404/20240404201/"
###########
## save vqa acc to csv
###########
modes = ['gqa','okvqa','vqav2','aokvqa']
# modes = ['gqa','okvqa','vqav2']
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
df['name'] = name
df.to_csv(f"/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/vqa_result/0404_results_{name}.csv",index=False)
print("save to /mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/vqa_result/0404_results_{}.csv".format(name))
print(df)

###########
## statistic all results
###########
# # load data from csv whose prefix is 0401_results
# import os
# import pandas as pd
# import numpy as np
# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/"
# files = os.listdir(path)
# files = [f for f in files if f.startswith("0403_results")]
# df = pd.DataFrame()
# for f in files:
#     tmp = pd.read_csv(os.path.join(path,f))
#     # expand tmp.shape to (10,)
#     if tmp.shape[0] < 10:
#         tmp = pd.concat([tmp,pd.DataFrame(np.zeros((10-tmp.shape[0],tmp.shape[1])),columns=tmp.columns)])
#     df = pd.concat([df,tmp])
# print(df)
# print(df.describe())
# df.to_csv("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/0403_all.csv")


###########
## save caption acc to csv
###########
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

    

