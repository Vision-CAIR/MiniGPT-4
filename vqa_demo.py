import argparse

import gradio as gr

# install gptcache with `pip install gptcache`
from gptcache import cache
from gptcache.processor.pre import get_image, get_image_question
from gptcache.embedding import Timm
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager.factory import manager_factory

from gptcache.adapter.minigpt4 import MiniGPT4


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--dir", type=str, default=".", help="path for data storage.")
    parser.add_argument("--map", action='store_true', help="use map for exact match cache.")
    parser.add_argument('--no-map', dest='map', action='store_false', help="use sqlite and faiss for similar search cache.")
    parser.set_defaults(map=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


args = parse_args()

print("Initializing GPTCache")
if args.map:
    data_manager = manager_factory("map", args.dir)
    cache.init(
        pre_embedding_func=get_image_question,
        data_manager=data_manager
    )  # init with map method
else:
    timm = Timm()
    data_manager = manager_factory("sqlite,faiss", args.dir, vector_params={"dimension": timm.dimension})
    cache.init(
        pre_embedding_func=get_image,
        data_manager=data_manager,
        embedding_func=timm.to_embeddings,
        similarity_evaluation=SearchDistanceEvaluation()
    )
print("GPTCache Initialization Finished")

print("Initializing Chat")
pipeline = MiniGPT4.from_pretrained(cfg_path=args.cfg_path, gpu_id=args.gpu_id, options=args.options, return_hit=True)
print(" Chat Initialization Finished")


# ========================================
#             Gradio Setting
# ========================================


title = """<h1 align="center">Demo of MiniGPT-4 and GPTCache</h1>"""
description = """<h3>This is the demo of MiniGPT-4 and GPTCache. Upload your images and ask question, and it will be cached.</h3>"""
article = """<p><a href="https://github.com/zilliztech/GPTCache"><img src="https://img.shields.io/badge/Github-Code-blue"></a></p>"""

# show examples below


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(article)
    with gr.Row():
        with gr.Column():
            inp0 = gr.Image(source="upload", type="filepath")
            inp1 = gr.Textbox(label="Question")
        with gr.Column():
            out0 = gr.Textbox()
            out1 = gr.Textbox(label="is hit")
    btn = gr.Button("Submit")
    btn.click(fn=pipeline, inputs=[inp0, inp1], outputs=[out0, out1])

demo.launch(share=True)
