from cog import BasePredictor, Input, Path
from minigpt4.common.config import Config
import torch
import argparse
from PIL import Image

from minigpt4.models import MiniGPT4
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

import os

# setting cache directory
os.environ["TORCH_HOME"] = "/src/model_cache"


class Predictor(BasePredictor):
    def setup(self):
        args = argparse.Namespace()
        args.cfg_path = "/src/eval_configs/minigpt4_eval.yaml"
        args.gpu_id = 0
        args.options = []

        config = Config(args)

        model = MiniGPT4.from_config(config.model_cfg).to("cuda")
        vis_processor_cfg = config.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device="cuda")

    def predict(
        self,
        image: Path = Input(description="Image to discuss"),
        prompt: str = Input(description="Prompt for mini-gpt4 regarding input image"),
        num_beams: int = Input(
            description="Number of beams for beam search decoding",
            default=3,
            ge=1,
            le=10,
        ),
        temperature: float = Input(
            description="Temperature for generating tokens, lower = more predictable results",
            default=1.0,
            ge=0.01,
            le=2.0,
        ),
        top_p: float = Input(
            description="Sample from the top p percent most likely tokens",
            default=0.9,
            ge=0.0,
            le=1.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            default=1.0,
            ge=0.01,
            le=5,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate", ge=1, default=3000
        ),
        max_length: int = Input(
            description="Total length of prompt and output in tokens",
            ge=1,
            default=4000,
        ),
    ) -> str:
        img_list = []
        image = Image.open(image).convert("RGB")
        with torch.inference_mode():
            chat_state = CONV_VISION.copy()
            self.chat.upload_img(image, chat_state, img_list)
            self.chat.ask(prompt, chat_state)
            answer = self.chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        return answer[0]
