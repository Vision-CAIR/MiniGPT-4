"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import string
import random
import copy
import json
import os
import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

@registry.register_model("blip2_t5_qformer_moe")
class Blip2T5InstructQformerMoE(Blip2Base):
    """
    BLIP2 Instruct T5 model Qformer MoE
    Supported model types:
        - flant5xxl
    Usage:
        >>> from minigpt4.models import load_model
        >>> import torch
        >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        >>> model = load_model("blip2_t5_qformer_moe", "flant5xxl", device=device)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl_qformer_moe.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2-flant5-xxl/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_llm=True,
        freeze_qformer=False,
        freeze_t5_proj=False,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        moebert_expert_num=5,
        moebert_route_method="gate-sentence",
        moebert_load_balance = 0.1,
        moe_topk = 1,
        use_balance_loss=True,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        print("Init BLIP2 Instruct Flant5xxl Prompt MoE")

        print('Initing & Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        # freeze vit
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     # freeze ln vision
        #     for name, param in self.ln_vision.named_parameters():
        #         param.requires_grad = False
        #     self.ln_vision = self.ln_vision.eval()
        #     self.ln_vision.train = disabled_train
        #     logging.info("freeze vision encoder")
        # print('Loading VIT Done')

        print('Initing MoE Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        print('Loading T5')
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config, use_safetensors=False
        )
        # freeze t5 llm 
        if freeze_llm:
            for name, param in self.t5_model.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16()
        print('Loading T5 Done')


        print("Initing t5 linear projection")
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        # load BLIP2 Pretrain
        print("Loading BLIP2 Parameters from :", q_former_model)
        self.load_from_pretrained(url_or_filename=q_former_model)

        # freeze qformer        
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            logging.info("freeze Qformer")

        # After loading, freeze t5_proj
        if freeze_t5_proj:
            for name, param in self.t5_proj.named_parameters():
                param.requires_grad = False
            self.t5_proj = self.t5_proj.eval()
            self.t5_proj.train = disabled_train

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

    def forward(self, samples):
        
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        bz = image_embeds.shape[0]
        query_tokens = self.query_tokens.expand(bz, -1, -1)

        ## Q-former Forward with one query tokens
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["q_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        query_output_to_linear = query_output.last_hidden_state[:,:query_tokens.size(1),:]
        # gate_loss = query_output.gate_loss
        inputs_t5 = self.t5_proj(query_output_to_linear)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["llm_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_output_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            # final_loss = loss + self.moebert_load_balance * gate_loss
            return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        # image embed
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                samples["q_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)
            
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        query_output_to_linear = query_output.last_hidden_state[:,:query_tokens.size(1),:]

        inputs_t5 = self.t5_proj(query_output_to_linear)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        if isinstance(samples["llm_input"], str):
            samples["llm_input"] = [samples["llm_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["llm_input"][i])
                    for i in range(len(samples["llm_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["llm_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["llm_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["llm_input"]]
        else:
            text_input = samples["llm_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2-flant5-xxl/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_llm = cfg.get("freeze_llm", True)
        freeze_qformer = cfg.get("freeze_qformer", False)
        freeze_t5_proj = cfg.get("freeze_t5_proj", False)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)
        
        moebert_expert_num = cfg.get("moebert_expert_num", 5)
        moebert_route_method = cfg.get("moebert_route_method", "gate-sentence")
        moebert_load_balance = cfg.get("moebert_load_balance", 0.1)
        moe_topk = cfg.get("moe_topk", 1)
        use_balance_loss = cfg.get("use_balance_loss", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            q_former_model=q_former_model,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_llm=freeze_llm,
            freeze_qformer=freeze_qformer,
            freeze_t5_proj=freeze_t5_proj,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            moebert_expert_num=moebert_expert_num,
            moebert_route_method=moebert_route_method,
            moebert_load_balance=moebert_load_balance,
            moe_topk=moe_topk,
            use_balance_loss=use_balance_loss,
        )

        if qformer_text_input:
            # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
            model.load_from_pretrained(
                url_or_filename="/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2_pretrained/blip2_pretrained.pth"
            )

        model.load_checkpoint_from_config(cfg)
        
        # check update params
        print("Updating following parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name)

        # layer self attention: 2,363,904
        # layer pure ffn : 4,723,968
        # layer expert ffn : 4,723,968
        # layer cross attention: 3,346,944

        return model
