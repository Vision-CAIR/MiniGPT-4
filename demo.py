import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from GroundingModel import GroundingModule
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from eval_scripts.conversation import Chat, CONV_VISION
# NOTE&TODO: put this before minigpt4 import will cause circular import 
# possibly because `imagebind` imports `minigpt4` and `minigpt4` also imports `imagebind`
from imagebind.models.image_bind import ModalityType


# imports modules for registration


def parse_args():
    parser = argparse.ArgumentParser(description="Qualitative")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
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

# model_config = cfg.model_cfg
# model_config.device_8bit = args.gpu_id
# model_cls = registry.get_model_class(model_config.arch)
# print(model_config)
# model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model = None

# TODO: Fix hard-coding  `cc12m`
vis_processor_cfg = cfg.datasets_cfg.cc12m.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
processors = {ModalityType.VISION: vis_processor}
chat = Chat(model, processors, device='cuda:{}'.format(args.gpu_id))
grounding = GroundingModule(device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, emb_list):
    if chat_state is not None:
        chat_state.messages = []
    if emb_list is not None:
        emb_list = []
    return None, gr.update(value=None, interactive=True), \
           gr.update(placeholder='Please upload your image first', interactive=False), \
           gr.update(value="Upload & Start Chat", interactive=True), \
           chat_state, emb_list


def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    emb_list = []
    # chat.upload_img(gr_img, chat_state, emb_list)
    return gr.update(interactive=False), \
           gr.update(interactive=True, placeholder='Type and press Enter'), \
           gr.update(value="Start Chatting", interactive=False), \
           chat_state, emb_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(image, chatbot, chat_state, emb_list, num_beams, temperature):
    # llm_message = chat.answer(conversation=chat_state,
    #                           emb_list=emb_list,
    #                           num_beams=num_beams,
    #                           temperature=temperature,
    #                           max_new_tokens=300,
    #                           max_length=2000)[0]
    llm_message = "I don't know"
    chatbot[-1][1] = llm_message
    ground_img = grounding.prompt2mask(image, 'dog')
    return ground_img, chatbot, chat_state, emb_list


title = """<h1 align="center">Demo of BindGPT-4</h1>"""
description = """<h3>This is the demo of BindGPT-4. Upload your images and start chatting!</h3>"""
# article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
# """

# TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            with gr.Row():
                image = gr.Image(type="pil")
                image2 = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            emb_list = gr.State()
            chatbot = gr.Chatbot(label='BindGPT-4')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)

    upload_button.click(upload_img, [image, text_input, chat_state],
                        [image, text_input, upload_button, chat_state, emb_list])

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        image, upload_img, gradio_answer, [chatbot, chat_state, emb_list, num_beams, temperature], [image2, chatbot, chat_state, emb_list]
    )
    clear.click(gradio_reset, [chat_state, emb_list], [chatbot, image, text_input, upload_button, chat_state, emb_list],
                queue=False)

demo.launch(share=True, enable_queue=True)
