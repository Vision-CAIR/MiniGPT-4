import random
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import re
from torch import Tensor
from transformers import LlamaTokenizer

from imagebind.models.image_bind import imagebind_huge, ImageBindJoiner, ModalityType
from minigpt4.common.registry import registry
from minigpt4.models.blip2 import BaseModel
from minigpt4.models.modeling_llama import LlamaForCausalLM


def filter_prompt(input_embeds: Dict[str, Tensor], prompt_list: List[str]) -> List[str]:
    if not prompt_list:
        return prompt_list
    input_modal_set = set([k.title() for k in input_embeds if input_embeds[k] is not None])
    prompt_modal_sets = [set(re.findall("<([^<>]+)><ModalityHere></\\1>", prompt)) for prompt in prompt_list]
    results = [prompt_list[i] for i, prompt_modal_set in enumerate(prompt_modal_sets) if
               prompt_modal_set == input_modal_set]
    return results


def arrange_modalities(input_embeds: Dict[str, Tensor], prompt: str) -> List[Tensor]:
    prompt_modalities = re.findall("<([^<>]+)><ModalityHere></\\1>", prompt)
    return [input_embeds[modality.lower()] for modality in prompt_modalities]


def concat_all_embeddings(input_embeds: Dict[str, Tensor], dim: int) -> Tensor:
    embeds = [input_embeds[key] for key in input_embeds if input_embeds[key] is not None]
    return torch.cat(embeds, dim=dim)


def filter_modalities(inputs):
    filtered_inputs = {}

    for k in ModalityType.__dict__.values():
        if k in inputs:
            filtered_inputs[k] = inputs[k]

    return filtered_inputs


@registry.register_model("bind_gpt4")
class BindGPT4(BaseModel):
    """
    ImageBind GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/bindgpt4.yaml",
    }

    def __init__(
            self,
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            freeze_imagebind=True,
            freeze_qformer=False,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()
        assert not low_resource, "Low Resource Mode is Currently Unavailable."

        self.low_resource = low_resource

        print('Loading ImageBind')
        self.multimodal_encoder = imagebind_huge(pretrained=True, freeze_imagebind=freeze_imagebind)
        print('Loading ImageBind Done')

        print(f'Loading LLAMA from {llama_model}')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model,
            torch_dtype=torch.float16,
        )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        print('Loading Q-Former and Adapter/Projector')
        self.multimodal_joiner = ImageBindJoiner(vision_query_token_num=num_query_token,
                                                 vision_qformer_frozen=freeze_qformer,
                                                 vision_post_dims=[768, self.llama_model.config.hidden_size],
                                                 audio_query_token_num=num_query_token,
                                                 audio_post_dims=[768, self.llama_model.config.hidden_size]
                                                 # vision_qformer_model=q_former_model,
                                                 # vision_pre_dims=(1280, 1408)
                                                 )
        print('Loading Q-Former and Adapter/Projector Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        print("Preparing Prompts")
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            self.prompt_list = [prompt_template.format(p) for p in raw_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        print("Preparing Prompts Done")

    def encode_inputs(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        imagebind_outputs = self.multimodal_encoder(inputs)
        llama_inputs = self.multimodal_joiner(imagebind_outputs)
        return llama_inputs

    def prompt_wrap(self, inputs: Dict[str, Tensor], prompt: str) -> Tuple[Tensor, Tensor]:
        if not prompt:
            input_embeds = concat_all_embeddings(inputs, dim=1)
            attns_input = torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(input_embeds.device)
            return input_embeds, attns_input
        input_embeds_list = arrange_modalities(inputs, prompt)
        batch_size = input_embeds_list[0].shape[0]
        prompt_slices = prompt.split('<ModalityHere>')
        prompt_tokens = [self.llama_tokenizer(prompt_slice, return_tensors="pt", add_special_tokens=False)
                             .to(input_embeds_list[0].device) for prompt_slice in prompt_slices]
        prompt_embeds = [self.llama_model.model.embed_tokens(prompt_token.input_ids).expand(batch_size, -1, -1)
                         for prompt_token in prompt_tokens]
        result_embeds = [emb for pair in zip(prompt_embeds[:-1], input_embeds_list)
                         for emb in pair] + [prompt_embeds[-1]]
        wrapped_input_embeds = torch.cat(result_embeds, dim=1)
        wrapped_atts_input = torch.ones(wrapped_input_embeds.size()[:-1],
                                        dtype=torch.long).to(wrapped_input_embeds.device)
        return wrapped_input_embeds, wrapped_atts_input

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # filter `inputs` as it may contain informatioins other than modalities
        modality_inputs = filter_modalities(inputs)
        embeds = self.encode_inputs(modality_inputs)
        filtered_prompts = filter_prompt(embeds, self.prompt_list)
        if filtered_prompts:
            prompt = random.choice(filtered_prompts)
        else:
            prompt = None
        input_embs, input_atts = self.prompt_wrap(embeds, prompt)

        # NOTE: No modifications from the next line to the end. Except for the autocast part.

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in inputs["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(input_embs.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([input_atts.shape[0], input_atts.shape[1] + 1],
                       dtype=torch.long).to(input_embs.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = input_embs.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = input_atts[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, input_embs, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, input_atts, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        q_former_model = cfg.get("q_former_model",
                                 "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        freeze_imagebind = cfg.get("freeze_imagebind", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            q_former_model=q_former_model,
            freeze_imagebind=freeze_imagebind,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load ImageBind-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
