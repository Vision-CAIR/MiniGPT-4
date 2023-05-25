import argparse
import json
import os

import shortuuid
from tqdm import tqdm

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
# TODO: check the circular import problem in `eval_scripts.conversation`
from eval_scripts.conversation import Chat, CONV_VISION
from imagebind.models.image_bind import ModalityType


def parse_args():
    parser = argparse.ArgumentParser(description="Quantitative")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument("--question-path", required=True, help="path to question file.")
    parser.add_argument("--answer-path", required=True, help="path to answer result file.")
    parser.add_argument("--image-folder", required=True, help="path to the image queries (COCO 2014 val).")
    args = parser.parse_args()
    return args


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

# TODO: fix hard-coding `cc12m`
vis_processor_cfg = cfg.datasets_cfg.cc12m.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
processors = {ModalityType.VISION: vis_processor}
chat = Chat(model, processors, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Prompt Setting
# ========================================
prompt = "Give the following image: <Vision>ImageContent</Vision>. " \
         "You will be able to see the image once I provide it to you. Please answer my question."

# ========================================
#             Question Loading
# ========================================
import pdb; pdb.set_trace()
questions = [json.loads(q) for q in open(args.question_path, "r")]
answer_file = open(args.answer_path, "w")
for i, line in enumerate(tqdm(questions)):
    idx = line["question_id"]
    image_file = os.path.join(args.image_folder, "COCO_val2014_" + line["image"])
    question = line["text"]
    state = CONV_VISION.copy()
    emb_list = []
    chat.upload_img(image_file, state, emb_list)
    chat.ask(question, state)
    answer, _ = chat.answer(state, emb_list)
    ans_id = shortuuid.uuid()
    answer_file.write(json.dumps({"question_id": idx,
                                  "prompt": question,
                                  "text": answer,
                                  "answer_id": ans_id,
                                  "model_id": model_config.arch,
                                  "metadata": {}}) + "\n")
    answer_file.flush()
answer_file.close()

