import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=2, help="specify the gpu to load the model.")
    parser.add_argument("--temperature", type=int, default=0.9, help="specify the gpu to load the model.")
    parser.add_argument("--english", type=bool, default=True, help="chinese or english")
    parser.add_argument("--prompt-en", type=str, default="can you describe the current picture?", help="Can you describe the current picture?")
    parser.add_argument("--prompt-zh", type=str, default="你能描述一下当前的图片？", help="Can you describe the current picture?")
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


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

while True:
    if not args.english:
        image_path = input("请输入图像路径或URL（回车进入纯文本对话）： ")
    else:
        image_path = input("Please enter the image path or URL (press Enter for plain text conversation): ")

    if image_path == 'stop':
        break
    if len(image_path) > 0:
        query = args.prompt_en if args.english else args.prompt_zh

    
    while True:
        if query == "clear":
            break
        if query == "stop":
            sys.exit(0)
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(image_path, chat_state, img_list)
        chat.ask(query, chat_state)
        llm_message = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=args.num_beams,
            temperature=args.temperature,
            max_new_tokens=300,
            max_length=2000
        )[0]
        # chatbot[-1][1] = llm_message
        print("MiniGPT4:"+llm_message)
        query = input("user:")
