from collections import OrderedDict
import json
import os
import random
import torch
from PIL import Image
from minigpt4.datasets.datasets.vqa_datasets import VQADataset  #, VQAEvalDataset

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from tqdm import tqdm

import sys
# sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS")
from minigpt4.common.registry import registry
import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess
from minigpt4.common.logger import setup_logger
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    # parser.add_argument("-f", help="jupyter notebook")
    parser.add_argument("--cfg-path", default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/minigpt4/projects/minigpt/train/minigptv2_finetune.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=5, help="specify the gpu to load the model.")
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

    
# build config
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
cfg = Config(parse_args())
setup_seeds(cfg)
print(cfg._convert_node_to_json(cfg.config))

setup_logger()
cfg.pretty_print()

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)