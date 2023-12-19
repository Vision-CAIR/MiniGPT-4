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
model = task.build_model(cfg)
# model = None
task.build_tensorboard(cfg)

runner = get_runner_class(cfg)(
    cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
)

for name, param in model.named_parameters():
    if param.requires_grad == True:
        if name == 'Qformer.bert.encoder.layer.10.experts.experts.2.intermediate_query.dense.weight':
            print(name)
            print(param)
    if name == 'Qformer.bert.encoder.layer.10.intermediate_query.dense.weight':
        print(name)
        print(param)
    

for name, param in model.named_parameters():
    if param.requires_grad == False:
        if 'Qformer' in name and '10' in name:
            print(name)

for key in m1['model'].keys():
    if 'Qformer' in key:
        print(key)

import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE")
from minigpt4.models.QformerMoE import (
    BertConfig, 
    BertMoELMHeadModel
)
vision_width = 1408
cross_attention_freq = 2
num_query_token = 32

# init_QformerMoE
moe_encoder_config = BertConfig.from_pretrained("/mnt/pfs-guan-ssai/nlu/wanghanzi/models/bert-base-uncased")
moe_encoder_config.encoder_width = vision_width
# insert cross-attention layer every other block
moe_encoder_config.add_cross_attention = True
moe_encoder_config.cross_attention_freq = cross_attention_freq
moe_encoder_config.query_length = num_query_token
moe_encoder_config.moebert_expert_num = 4
moe_encoder_config.moebert_route_method = "gate-sentence"
moe_encoder_config.moe_topk = 2
moe_encoder_config.moebert_load_balance = 0.1
moe_encoder_config.moebert_share_importance = 512 # TODO: meaning?
MoEQformer = BertMoELMHeadModel.from_pretrained(
    "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/bert-base-uncased", config=moe_encoder_config
)



"""
    forward
"""
import torch
import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE")
from minigpt4.models import load_model
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = load_model("blip2", "pretrain", device=device)

samples = {
    'q_input':["What is around the open window?",  # n23181
                    "Is the ground blue or brown?", # n168412
                    "What color are the pants?", # n446242
                    "What is the airplane flying above?"], # n414992
    'llm_input':["What is around the open window?",  # n23181
                    "Is the ground blue or brown?", # n168412
                    "What color are the pants?", # n446242
                    "What is the airplane flying above?"], # n414992
    'text_output':["drapes",
                    "brown",
                    "red",
                    "ocean"
                    ],
    'image': torch.randn(4, 3, 224, 224).half().to(device)
}

Qformer, query_tokens = model.init_QformerMoE(
        num_query_token=32, 
        vision_width=1408,
        moebert_expert_num=5,
        moebert_route_method="gate-sentence",
        moebert_load_balance=0.1,
        moe_topk=2,
        cross_attention_freq=2
    )
Qformer = Qformer.to(device)

def maybe_autocast(device, dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = device != torch.device("cpu")
    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()
    
image = samples["image"]
with maybe_autocast(device):
    image_embeds = model.ln_vision(model.visual_encoder(image))
image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
bs = image.size(0)
query_tokens = query_tokens.expand(bs, -1, -1).to(device)



# image = samples["image"]
# image_atts = torch.ones(4, 257).to(device)
# image_embeds = torch.randn(4, 257, 1408).to(device)
# bz = image_embeds.shape[0]
# query_tokens = query_tokens.expand(bz, -1, -1).to(device)

text_Qformer = model.tokenizer(
    samples["q_input"],
    padding='longest',
    truncation=True,
    max_length=32,
    return_tensors="pt",
).to(image.device)
query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1).to(device)

query_output = Qformer.bert(
    text_Qformer.input_ids,
    attention_mask=Qformer_atts,
    query_embeds=query_tokens,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_atts,
    return_dict=True,
)

