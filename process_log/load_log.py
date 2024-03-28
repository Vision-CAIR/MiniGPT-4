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
path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_uni_route/mix_coco_gqa_ao_cocap_textcap_raw_QformerMoE_Post_Route_Universal_lnout_lr3e5_3ex3b_2loss_005_top6layer_textinqf_10epo_0326/20240326134"
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
df.to_csv("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/results.csv",index=False)
print(df)
modes = ['coco_cap','text_cap']
for mode in modes:
    file_name[mode] = os.path.join(path, f"evaluate_{mode}.txt")
    tmp = list()
    with open(file_name[mode], "r") as f:
        for line in f: 
            tmp.append(json.loads(line)["val"])
    df1 = pd.DataFrame(tmp)
    df1.to_csv("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/process_log/results_{}.csv".format(mode),index=False)
    print("\n",df1)

    

