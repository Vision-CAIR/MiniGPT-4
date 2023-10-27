import argparse
import os
import random
import requests
from io import BytesIO

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, StoppingCriteriaList, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--cfg-path", required=False, default='textvqa_eval.yaml', help="path to configuration file.")
parser.add_argument("--ckpt_path", required=False, help="path to configuration file.")
parser.add_argument("--lora_r", type=int, default=64, help="path to configuration file.")
parser.add_argument("--lora_alpha", type=int, default=16, help="path to configuration file.")
parser.add_argument("--name", type=str)
parser.add_argument("--img_path", type=str)
parser.add_argument("--eval_file_path", type=str)
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)


print('Initializing Chat')
args = parser.parse_args()
cfg = Config(args)

ckpt_list = [args.ckpt_path]

print('evaluating config:', args.cfg_path)
print('evaluating checkpoint:', args.ckpt_path)



import json
import numpy as np

f = open(f'{args.eval_file_path}/textocr/TextOCR_0.1_val.json', 'r')
textcap_dataset_val = json.load(f)

all_img_ids = list(textcap_dataset_val['imgs'].keys())
ann_ids = list(textcap_dataset_val['anns'].keys())


for ckpt in ckpt_list:
    cfg.model_cfg.ckpt = ckpt
    cfg.model_cfg.lora_r=args.lora_r
    cfg.model_cfg.lora_alpha=args.lora_alpha

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')
    model.eval()
    print('Initialization Finished')

    vis_processor_cfg = cfg.datasets_cfg.coco_vqa.vis_processor.eval
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    text_processor_cfg = cfg.datasets_cfg.coco_vqa.text_processor.eval
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
    task = tasks.setup_task(cfg)

    from collections import Counter

    def find_most_frequent_item(items):
        counter = Counter(items)
        most_common = counter.most_common(1)
        if most_common:
            return most_common[0][0]
        else:
            return None
    from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

    class textVQAEvalDataset(VQADataset):
        def __init__(self, vis_processor, text_processor, vis_root=None, ann_paths=None, use_ocr=False):
    #         super().__init__(vis_processor, text_processor, vis_root, ann_paths)

            from datasets import load_dataset
            self.annotation = load_dataset("textvqa", split="validation",cache_dir=f"{args.eval_file_path}/textvqa_cache")
            self.vis_processor = vis_processor
            self.text_processor = text_processor
            self.instruction_pool = [
                '[vqa] Question: {} Short answer:',
            ]
            
            self.use_ocr = use_ocr


        def __getitem__(self, index):
            ann = self.annotation[index]
            image_path = ann['flickr_original_url']
            image = ann["image"].convert("RGB")

            img_id = ann['image_id']
            sel_anns = [textcap_dataset_val['anns'][ann] for ann in ann_ids if ann.startswith(img_id)]
            sel_caps = [ann['utf8_string'] for ann in sel_anns]
            sel_caps = [cap for cap in sel_caps if cap not in ['\n', '.']]
            captions = " ".join(sel_caps)
        
            image = self.vis_processor(image)
            question = self.text_processor(ann["question"])
            captions = self.text_processor(captions)

            instruction = random.choice(self.instruction_pool).format(question)
            
            if self.use_ocr and img_id in all_img_ids:
                instruction = "<s>[INST] <Img><ImageHere></Img> OCR tokens: {}. {} [/INST]".format(captions, instruction)
            else:
                instruction = "<s>[INST] <Img><ImageHere></Img> {} [/INST]".format(instruction)
            
            
            answers = find_most_frequent_item(ann["answers"])
            
            return {
                "image": image,
                "text_input": question,
                "answer": answers,
                'image_path': image_path,
                "instruction_input": instruction,
                "question_id": ann["question_id"],
    #             "instance_id": ann["instance_id"],
            }

    dataset = textVQAEvalDataset(vis_processor, text_processor, use_ocr=cfg.run_cfg.use_ocr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.run_cfg.batch_size_eval, num_workers=cfg.run_cfg.num_workers)

    result_dir = cfg.run_cfg.output_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    epoch = cfg.model_cfg.ckpt.split('/')[-1].split('.')[0].split('_')[1]
    exp_id = cfg.model_cfg.ckpt.split('/')[-2] + "_" + epoch
    
    val_result = task.evaluation(model, dataloader, cuda_enabled=True)
    task.after_evaluation(val_result, split_name=f'{args.name}_textvqa_val', result_dir=result_dir)


    anno_files = f'{args.eval_file_path}/TextVQA/gt_ann.json'
    ques_files = f'{args.eval_file_path}/TextVQA/gt_ques.json'

    from minigpt4.common.vqa_tools.vqa import VQA
    from minigpt4.common.vqa_tools.vqa_eval import VQAEval
    import logging, json, os

    result_file = '{}/{}_textvqa_val.json'.format(result_dir, args.name)
    vqa = VQA(anno_files, ques_files)
    vqa_result = vqa.loadRes(
        resFile=result_file, quesFile=ques_files
    )

    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqa_scorer = VQAEval(vqa, vqa_result, n=2)
    logging.info("Start VQA evaluation.")
    vqa_scorer.evaluate()

    # print accuracies
    overall_acc = vqa_scorer.accuracy["overall"]
    print('textvqa val acc: ', overall_acc, flush=True)
