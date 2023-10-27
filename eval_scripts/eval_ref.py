import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_LLama2

from minigpt4.datasets.datasets.coco_caption import RefCOCOEvalData

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--split", type=list_of_str, default='test', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
parser.add_argument("--img_path", type=str)
parser.add_argument("--eval_file_path", type=str)
args = parser.parse_args()

print(args.ckpt)
print(args.name)

eval_dict = {'refcoco': args.split, 
            'refcoco+': args.split,
            'refcocog': args.split}

model, vis_processor = init_model(args)
model.eval()
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

model.eval()
img_path=f'{args.img_path}/COCO/cocoapi/data/2017/images/jpeg/train'
    
for dataset in args.dataset:
    for split in eval_dict[dataset]:
        with open(f'{args.eval_file_path}/{dataset}/{dataset}_{split}.json', 'r') as f:
            refcoco = json.load(f)

        data = RefCOCOEvalData(refcoco, vis_processor, img_path)
        eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

        minigpt4_predict = defaultdict(list)

        resamples = []

        for images, questions, img_ids in tqdm(eval_dataloader):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)
            for answer, img_id, question in zip(answers, img_ids, questions):
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                if re.match(pattern, answer):
                    minigpt4_predict[img_id].append(answer)
                else:
                    resamples.append({'img_id': img_id, 'sents': [question.replace('[refer] where is','').replace('?','').strip()]})
        
        if args.resample:
            for i in range(20):
                data = RefCOCOEvalData(resamples, vis_processor, img_path)
                resamples = []
                eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
                for images, questions, img_ids in tqdm(eval_dataloader):
                    texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                    answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)
                    for answer, img_id, question in zip(answers, img_ids, questions):
                        answer = answer.replace("<unk>","").replace(" ","").strip()
                        pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                        if re.match(pattern, answer) or i == 4:
                            minigpt4_predict[img_id].append(answer)
                        else:
                            resamples.append({'img_id': img_id, 'sents': [question.replace('[refer] where is','').replace('?','').strip()]})
                            
                if len(resamples) == 0:
                    break

        with open(f'results/{args.name}_{dataset}_{split}.json','w') as f:
            json.dump(minigpt4_predict, f)

        count=0
        total=len(refcoco)
        res=args.res
        refcoco_dict = defaultdict()
        for item in refcoco:
            refcoco_dict[item['img_id']] = item
        for img_id in refcoco_dict:
            item = refcoco_dict[img_id]
            bbox = item['bbox']
            outputs = minigpt4_predict[img_id]
            for output in outputs:
                try:
                    integers = re.findall(r'\d+', output)
                    pred_bbox = [int(num) for num in integers]
                    height = item['height']
                    width = item['width']
                    pred_bbox[0] = pred_bbox[0] / res * width
                    pred_bbox[1] = pred_bbox[1] / res * height
                    pred_bbox[2] = pred_bbox[2] / res * width
                    pred_bbox[3] = pred_bbox[3] / res * height

                    gt_bbox = [0,0,0,0]
                    gt_bbox[0] = bbox[0]
                    gt_bbox[1] = bbox[1]
                    gt_bbox[2] = bbox[0] + bbox[2]
                    gt_bbox[3] = bbox[1] + bbox[3]

                    iou_score = computeIoU(pred_bbox, gt_bbox)
                    if iou_score > 0.5:
                        count+=1
                except:
                    continue
        
        print(f'{dataset} {split}:', count / total * 100, flush=True)
