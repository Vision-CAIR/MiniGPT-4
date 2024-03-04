import os
import json
import pandas as pd
from tqdm import tqdm

from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict

class COCO_Annotation:
    def __init__(self, annotation_file):
        self.coco_cn_file = annotation_file
        self.imgToAnns = self.build_imgToAnns()
    
    def build_imgToAnns(self):
        imgToAnns = defaultdict(list)
        with open(self.coco_cn_file, "r", encoding="UTF-8") as fin:
            for line in fin:
                line = line.strip()
                temp = eval(line)
                annotations = temp['annotations']
                for ann in annotations:
                    image_id = str(ann['image_id']).zfill(6)
                    imgToAnns[image_id].append({'image_id':image_id,'caption':ann['caption'],'image': ann['image_id']})
        return imgToAnns
    
    def getImgIds(self):
        return self.imgToAnns.keys()  

class COCO_Result:
    def __init__(self,result_file):
        self.coco_cn_file = result_file
        self.imgToAnns = self.build_imgToAnns()
    
    def build_imgToAnns(self):
        imgToAnns = dict()
        data = json.load(open(self.coco_cn_file, "r"))
        for d in data:
            tmp = {
                'image_id':d['question_id'][-6:],
                'caption':d['answer']
            }
            imgToAnns[d['question_id'][-6:]] = [tmp]
        return imgToAnns
    
def coco_caption_eval(results_file, split_name):
    files = {
        "val":"/mnt/pfs-guan-ssai/nlu/wanghanzi/data/COCO_Cap/coco_karpathy_val_gt.json",
        "test":"/mnt/pfs-guan-ssai/nlu/wanghanzi/data/COCO_Cap/coco_karpathy_test_gt.json"
    }

    # create coco object and coco_result object
    annotation_file = files[split_name]
    coco = COCO_Annotation(annotation_file)
    coco_result = COCO_Result(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def main():
    result_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/vicuna7b/qformer_moe_post/mix_coco_gqa_cap_raw_QformerMoE_Post_linear_gate_lnout_lr5e5_3ex_top1_2loss_005_top6layer_textinqf_6epo_0302/20240302231/result/val_vqa_result_coco_cap.json"
    split_name = "val"
    coco_val = coco_caption_eval(result_file, split_name)

    agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
    
    # log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}
    # with open(
    #     os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    # ) as f:
    #     f.write(json.dumps(log_stats) + "\n")

    coco_res = {k: v for k, v in coco_val.eval.items()}
    coco_res["agg_metrics"] = agg_metrics

    print(coco_res)


main()