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

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--cfg-path", required=False, default='scienceqa_eval.yaml', help="path to configuration file.")
parser.add_argument("--ckpt_path", required=False, help="path to configuration file.")
parser.add_argument("--lora_r", type=int, default=64, help="path to configuration file.")
parser.add_argument("--lora_alpha", type=int, default=16, help="path to configuration file.")
parser.add_argument("--name", type=str)

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
    

    from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

    class textVQAEvalDataset(VQADataset):
        def __init__(self, vis_processor, text_processor, vis_root=None, ann_paths=None):
    #         super().__init__(vis_processor, text_processor, vis_root, ann_paths)

            from datasets import load_dataset

            from datasets import load_dataset
            self.annotation = load_dataset("derek-thomas/ScienceQA", split='test')

            ## select anns with image 
            self.annotation = [ann for ann in self.annotation if ann['image']]

            self.vis_processor = vis_processor
            self.text_processor = text_processor
            self.instruction_pool = [
                '[vqa] Question: {} Answer:',
            ]

            self.alphabet_options = ["A", "B", "C", "D", "E"]
            self.max_num_choices = len(self.alphabet_options)


        def __getitem__(self, index):
            ann = self.annotation[index]
            image = ann['image'].convert("RGB")

            image = self.vis_processor(image)
            question = self.text_processor(ann["question"])
            lecture = self.text_processor(ann['lecture'])
            hint = self.text_processor(ann['hint'])
            options = ann['choices']
            
            num_choices = len(ann['choices'])
            if len(ann['choices'])<self.max_num_choices:
                options = options + (self.max_num_choices - len(ann['choices'])) * [""]
            else:
                options = options

            options = [self.text_processor(opt) for opt in options]

            instruction = random.choice(self.instruction_pool).format(question, lecture)
            
            instruction = "<s>[INST]<Img><ImageHere></Img> {} [/INST]".format(instruction)
            
            answer = options[ann['answer']]
            
            return {
                "image": image,
                "question": question,
                "choices": options,
#                 "choices": self.alphabet_options,
                "num_choices": num_choices,
                "answer": answer,
                "instruction_input": instruction,
                "question_id": index,
            }
    
    dataset = textVQAEvalDataset(vis_processor, text_processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.run_cfg.batch_size_eval, \
                                             num_workers=cfg.run_cfg.num_workers)


    print(len(dataset), len(dataloader))
    result_dir = cfg.run_cfg.output_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
        
    epoch = cfg.model_cfg.ckpt.split('/')[-1].split('.')[0].split('_')[1]
    exp_id = cfg.model_cfg.ckpt.split('/')[-2] + "_" + epoch

    val_result = task.evaluation(model, dataloader, cuda_enabled=True)
    task.after_evaluation(val_result, split_name=f'{args.name}_scienceqa_val', result_dir=result_dir)

    from minigpt4.common.vqa_tools.vqa import VQA
    from minigpt4.common.vqa_tools.vqa_eval import VQAEval
    import json, logging, os
    result_file = '{}/{}_scienceqa_val.json'.format(result_dir, args.name)
    results = json.load(open(result_file, "r"))
    acc = []
    vqa_tool = VQAEval()

    for res in results:

        gt_ans = res["gt_ans"]
        pred = res["pred_ans"]

        pred = vqa_tool.processPunctuation(pred)
        pred = vqa_tool.processDigitArticle(pred)

        vqa_acc = 1 if pred == gt_ans else 0

        acc.append(vqa_acc)

    accuracy = sum(acc) / len(acc) * 100
    print('scienceqa val acc: ', accuracy, flush=True)
    
