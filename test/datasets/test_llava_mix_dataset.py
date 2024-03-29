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
        default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/minigpt4/projects/qformer_moe_post_vicuna/train/mix_qformer_moe_post_blip2_vicuna7b_data_balance_finetuned.yaml",
        help="path to configuration file.")
    parser.add_argument(
        "--gpu-id", 
        type=int, 
        default=5, 
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

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls

# Test About Building Task
# build config
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
cfg = Config(parse_args())
setup_seeds(cfg)
print(cfg._convert_node_to_json(cfg.config))

setup_logger()
cfg.pretty_print()

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)

job_id = now()
# model = task.build_model(cfg)
# model = None
task.build_tensorboard(cfg)

runner = get_runner_class(cfg)(
    cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
)

data_loader = runner.train_loader
data_loader = runner.dataloaders['val']
