"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import wandb

import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE")

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.logger import setup_logger
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    # parser.add_argument("-f", help="jupyter notebook")
    parser.add_argument(
        "--cfg-path", 
        default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/minigpt4/projects/qformer_moe_vicuna/eval/mix_vqa_coco_vicuna_eval.yaml",
        help="path to configuration file.")
    parser.add_argument(
        "--gpu-id", 
        type=int, 
        default=4, 
        help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# Test About Building Task
# build config
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
cfg = Config(parse_args())
setup_seeds(cfg)
print(cfg._convert_node_to_json(cfg.config))

setup_logger()
cfg.pretty_print()

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls

job_id = now()
# model = task.build_model(cfg)
model = None
# task.build_tensorboard(cfg)

runner = get_runner_class(cfg)(
    cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
)


"""
    Dataset & DataLoader Setup
"""
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
import webdataset as wds
import logging

batch_sizes = {dataset_name: getattr(runner.config.datasets_cfg, dataset_name).batch_size
                for dataset_name in runner.datasets.keys()}
print(batch_sizes)


datasets, batch_sizes = reorg_datasets_by_split(runner.datasets, batch_sizes)
runner.datasets = datasets
# self.datasets = concat_datasets(datasets)
print(runner.datasets.keys()) # dict_keys(['train', 'val', 'test'])

# print dataset statistics after concatenation/chaining
for split_name in runner.datasets:
    if isinstance(runner.datasets[split_name], tuple) or isinstance(
        runner.datasets[split_name], list
    ):
        # mixed wds.DataPipeline and torch.utils.data.Dataset
        num_records = sum(
            [
                len(d)
                if not type(d) in [wds.DataPipeline, ChainDataset]
                else 0
                for d in runner.datasets[split_name]
            ]
        )

    else:
        if hasattr(runner.datasets[split_name], "__len__"):
            # a single map-style dataset
            num_records = len(runner.datasets[split_name])
        else:
            # a single wds.DataPipeline
            num_records = -1
            logging.info(
                "Only a single wds.DataPipeline dataset, no __len__ attribute."
            )

    if num_records >= 0:
        logging.info(
            "Loaded {} records for {} split from the dataset.".format(
                num_records, split_name
            )
        )

split_names = sorted(runner.datasets.keys())

datasets = [runner.datasets[split] for split in split_names]
batch_sizes = [batch_sizes[split] for split in split_names]
is_trains = [split in runner.train_splits for split in split_names]

print("split_names: ",split_names)
print("is_trains: ",is_trains)
print("batch sizes: ", batch_sizes)


collate_fns = []
for dataset in datasets:
    if isinstance(dataset, tuple) or isinstance(dataset, list):
        collate_fns.append([getattr(d, "collater", None) for d in dataset])
    else:
        collate_fns.append(getattr(dataset, "collater", None))

dataloaders = runner.create_loaders(
    datasets=datasets,
    num_workers=runner.config.run_cfg.num_workers,
    batch_sizes=batch_sizes,
    is_trains=is_trains,
    collate_fns=collate_fns,
)
_dataloaders = {k: v for k, v in zip(split_names, dataloaders)}


loader = _dataloaders['train']
loader = _dataloaders['val']
loader_idx = random.choices(range(len(loader.loaders)), loader.ratios, k=1)[0]
print(loader_idx)
next(loader.loaders[loader_idx])['question_id'], next(loader.loaders[loader_idx])['source']




import json
file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/prompt_moe/gqa_943k_raw_QformerMoE_train_qf_train_qt_linear_gate_20ex_top2_3loss_textinqf_epo3_1115/20231115234/result/val_vqa_result_gqa.json"
data = json.load(open(file, "r"))
cnt = 0
for i in range(len(data)):
    d = data[i]
    # if d['gt_ans'] in d['pred_ans'].lower():
    if d['gt_ans'] in d['answer'].lower():
        cnt += 1
    else:
        print(d)
    if i == 100:
        break

print(cnt/len(data))


# from minigpt4.common.vqa_tools.vqa import VQA
# from minigpt4.common.vqa_tools.vqa_eval import VQAEval

# split = 'val'
# source = 'vqav2'
# print(task.anno_files[split][source],task.ques_files[split][source])
# vqa = VQA(task.anno_files[split][source], task.ques_files[split][source])

# result_file = '/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/prompt_moe/mix_coco_gqa_1610k_raw_postMoE_train_qf_train_qt_linear_gate_20ex_top2_3loss_textinqf_epo3_1101/20231031224/result/val_vqa_result_2.json'
# print('result_file: ',result_file)
# vqa_result = vqa.loadRes(resFile=result_file, quesFile=task.ques_files[split][source])

# vqa_scorer = VQAEval(vqa, vqa_result, n=2)
# vqa_scorer.evaluate()
# overall_acc = vqa_scorer.accuracy["overall"]
# perAnswerType = vqa_scorer.accuracy["perAnswerType"]

# print(overall_acc)
# print(perAnswerType)
