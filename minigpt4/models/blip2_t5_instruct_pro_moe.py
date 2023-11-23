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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from minigpt4.models.moe.prompt_moe import init_query_token_candidates, PrePromptMoE, PostPromptMoE

@registry.register_model("blip2_t5_instruct_pro_moe")
class Blip2T5InstructPromptMOE(Blip2Base):
    """
    BLIP2 Instruct T5 model Prompt MoE
    Supported model types:
        - flant5xxl
    Usage:
        >>> from minigpt4.models import load_model
        >>> import torch
        >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        >>> model = load_model("blip2_t5_instruct_pro_moe", "flant5xxl", device=device)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl_prompt_moe.yaml",
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
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
        repeat_to_init_qt_candidates=True,
        num_qt_candidates=5,
        moe_topk=2,
        moe_position="pre",
        embed_extract="t5",
        eval_gate_save=False,
        train_gate_save=False,
        gate_save_path="",
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
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            # freeze ln vision
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')


        print('Initing Q-Former')
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


        print('Init query token candidates')
        self.moe_position = moe_position
        if num_qt_candidates > 1:
            
            self.query_token_candidates = init_query_token_candidates(num_query_token, num_qt_candidates) # shape:[num_qt_candidates, num_query_token, q_former_hidden_size]
            if repeat_to_init_qt_candidates:
                self.query_token_candidates = torch.nn.Parameter(self.query_tokens.repeat(num_qt_candidates, 1, 1))
                self.query_tokens.requires_grad = False
            print(self.query_token_candidates.shape)
            
            if self.moe_position == "pre": # PromptMoE + Qformer
                self.embed_extract = embed_extract
                if self.embed_extract == "t5":
                    self.text_embed_size = self.t5_model.config.hidden_size

                elif self.embed_extract == "blip2_pretrain":
                    from minigpt4.models import load_model
                    self.embed_extractor = load_model(
                        "blip2",
                        "pretrain", 
                        is_eval=True, 
                    ) # BLIP2 first-stage model with Q-former and ViT.
                    for name, param in self.embed_extractor.named_parameters():
                        param.requires_grad = False
                    # self.text_embed_size = self.Qformer.config.hidden_size
                    self.text_embed_size = self.embed_extractor.text_proj.out_features
                
                elif self.embed_extract == "random":
                    self.text_embed_size = self.Qformer.config.hidden_size
                self.PromptMoE = PrePromptMoE(self.text_embed_size, num_qt_candidates, self.query_token_candidates, route_method="gate-single-token", topk=moe_topk)
            
            elif moe_position == "post": #  Qformer + PromptMoE
                self.text_embed_size = self.Qformer.config.hidden_size
                self.PromptMoE = PostPromptMoE(self.text_embed_size, num_qt_candidates, topk=moe_topk)


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

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.qformer_text_input = qformer_text_input

        self.num_qt_candidates = num_qt_candidates
        self.gate_save_path = gate_save_path
        self.train_gate_save = train_gate_save
        self.eval_gate_save = eval_gate_save
        if gate_save_path!="" and (not os.path.exists(gate_save_path)):
            print(gate_save_path)
            os.mkdir(gate_save_path)

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples.keys())
        # print(samples)
        # print(samples["text_output"])
        # print('-----------------')
        import torch
        samples = {
            'text_input':["What is around the open window?",  # n23181
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

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        bz = image_embeds.shape[0]

        if self.moe_position == "pre":
            if self.num_qt_candidates > 1:
                ## extract text_embeds
                with self.maybe_autocast(dtype=torch.bfloat16):
                    if self.embed_extract == "t5":
                        text_embeds = self._extract_text_embed_by_t5(samples['q_input'], samples['text_output'], image.device)
                    elif self.embed_extract == "blip2_pretrain":
                        text_embeds = self._extract_text_embed_by_qformer_pretrain_s1(samples['q_input'], image.device)
                    elif self.embed_extract == "random":
                        text_embeds = torch.randn(bz, 1, self.text_embed_size )
                    ## select proper query_tokens by prompt moe
                    select_query_tokens, balance_loss, importance_loss, gate_load, gate = self.PromptMoE._forward_gate_single_token(text_embeds)
                    query_tokens = select_query_tokens # torch.Size([bz, 32, 768])
            else:
                query_tokens = self.query_tokens.expand(bz, -1, -1)
                balance_loss, importance_loss = 0, 0

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


        elif self.moe_position == "post":
            # self.query_token_candidates : size[num_qt_candidates, 32, 768]
            candi_query_tokens = self.query_token_candidates.expand(bz, -1, -1, -1).reshape(-1, self.query_token_candidates.shape[1], self.query_token_candidates.shape[2]) # size[num_qt_candidates*bz, 32, 768]

            image_embeds_repeat = image_embeds.repeat_interleave(self.num_qt_candidates, dim=0)
            image_atts_repeat = image_atts.repeat_interleave(self.num_qt_candidates, dim=0)
                
            ## Q-former Forward with candidates query tokens
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    samples["q_input"],
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                text_Qformer_input_ids_repeat = text_Qformer.input_ids.repeat_interleave(self.num_qt_candidates, dim=0) # [bz*num_qt_candidates, batch_seq_len]
                text_Qformer_attn_mask_repeat = text_Qformer.attention_mask.repeat_interleave(self.num_qt_candidates, dim=0) # [bz*num_qt_candidates, batch_seq_len]

                query_atts = torch.ones(candi_query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts,text_Qformer_attn_mask_repeat],dim=1)

                query_output = self.Qformer.bert(
                    text_Qformer_input_ids_repeat,
                    attention_mask=Qformer_atts,
                    query_embeds=candi_query_tokens,
                    encoder_hidden_states=image_embeds_repeat,
                    encoder_attention_mask=image_atts_repeat,
                    return_dict=True,
                ) # query_output.last_hidden_state size [torch.Size([bz*num_qt_candidates, 32+batch_seq_len, 768])]
                query_output_to_linear = query_output.last_hidden_state[:,:self.query_token_candidates.size(1),:]
            else:
                query_output = self.Qformer.bert(
                    query_embeds=candi_query_tokens,
                    encoder_hidden_states=image_embeds_repeat,
                    encoder_attention_mask=image_atts_repeat,
                    return_dict=True,
                ) # query_output.last_hidden_state size [torch.Size([bz*num_qt_candidates, 32, 768])]
            # [(sample1, query1), (sample1, query2),..., (sample2, query1),(sample2, query2), ... , (sample_bz, query1),..., (sample_bz, queryn)]
            text_cls = query_output.last_hidden_state[:,self.query_token_candidates.size(1),:] # torch.Size([bz*num_qt_candidates, 768])
            text_cls_split = text_cls.view(bz, self.num_qt_candidates, -1) # torch.Size([bz, num_qt_candidates, 768])
            query_tokens_output = query_output.last_hidden_state[:, :self.query_token_candidates.size(1), :] # torch.Size([bz*num_qt_candidates, 32, 768])
            query_output_to_linear, balance_loss, importance_loss, gate_load, gate = self.PromptMoE._forward_gate_text_single_token(text_cls_split, query_tokens_output)

        inputs_t5 = self.t5_proj(query_output_to_linear)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        fs_embeds, fs_atts = None, None
        if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
            fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples['few_shot_samples'])

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

            if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            if self.train_gate_save:
                self._save_gate(
                    samples['q_input'],
                    samples['text_output'],
                    gate,
                    samples['image_id'],
                    gate_load,
                    os.path.join(self.gate_save_path, "train_gate.txt")
                )

            final_loss = loss + balance_loss + importance_loss
            return {"loss": final_loss}

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


        if self.moe_position == "pre":
            if self.num_qt_candidates > 1:
                
                with self.maybe_autocast(dtype=torch.bfloat16):
                    if self.embed_extract == "t5":
                        text_embeds = self._extract_text_embed_by_t5(samples["q_input"], samples['text_output'], image.device)
                    elif self.embed_extract == "blip2_pretrain":
                        text_embeds = self._extract_text_embed_by_qformer_pretrain_s1(samples["q_input"], image.device)
                    elif self.embed_extract == "random":
                        text_embeds = torch.randn(bs, 1, self.text_embed_size )
                    select_query_tokens, _, _, gate_load, gate = self.PromptMoE._forward_gate_single_token(text_embeds)
                    query_tokens = select_query_tokens # torch.Size([bz, 32, 768])
            else: # back to one query token
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

        elif self.moe_position == "post":
            # self.query_token_candidates : size[num_qt_candidates, 32, 768]
            candi_query_tokens = self.query_token_candidates.expand(bs, -1, -1, -1).reshape(-1, self.query_token_candidates.shape[1], self.query_token_candidates.shape[2]) # size[num_qt_candidates*bz, 32, 768]
            image_embeds_repeat = image_embeds.repeat_interleave(self.num_qt_candidates, dim=0)
            image_atts_repeat = image_atts.repeat_interleave(self.num_qt_candidates, dim=0)

            ## Q-former Forward with candidates query tokens
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    samples["q_input"],
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                text_Qformer_input_ids_repeat = text_Qformer.input_ids.repeat_interleave(self.num_qt_candidates, dim=0) # [bz*num_qt_candidates, batch_seq_len]
                text_Qformer_attn_mask_repeat = text_Qformer.attention_mask.repeat_interleave(self.num_qt_candidates, dim=0) # [bz*num_qt_candidates, batch_seq_len]

                query_atts = torch.ones(candi_query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts,text_Qformer_attn_mask_repeat],dim=1)

                query_output = self.Qformer.bert(
                    text_Qformer_input_ids_repeat,
                    attention_mask=Qformer_atts,
                    query_embeds=candi_query_tokens,
                    encoder_hidden_states=image_embeds_repeat,
                    encoder_attention_mask=image_atts_repeat,
                    return_dict=True,
                ) # query_output.last_hidden_state size [torch.Size([bz*num_qt_candidates, 32+batch_seq_len, 768])]
                query_output_to_linear = query_output.last_hidden_state[:,:self.query_token_candidates.size(1),:]
            else:
                query_output = self.Qformer.bert(
                    query_embeds=candi_query_tokens,
                    encoder_hidden_states=image_embeds_repeat,
                    encoder_attention_mask=image_atts_repeat,
                    return_dict=True,
                ) # query_output.last_hidden_state size [torch.Size([bz*num_qt_candidates, 32, 768])]
            # [(sample1, query1), (sample1, query2),..., (sample2, query1),(sample2, query2), ... , (sample_bz, query1),..., (sample_bz, queryn)]
            text_cls = query_output.last_hidden_state[:,self.query_token_candidates.size(1),:] # torch.Size([bz*num_qt_candidates, 768])
            text_cls_split = text_cls.view(bs, self.num_qt_candidates, -1) # torch.Size([bz, num_qt_candidates, 768])
            query_tokens_output = query_output.last_hidden_state[:, :self.query_token_candidates.size(1), :] # torch.Size([bz*num_qt_candidates, 32, 768])
            query_output_to_linear, _, _, gate_load, gate = self.PromptMoE._forward_gate_text_single_token(text_cls_split, query_tokens_output)

        # For video data deleted : TODO

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
        
        if self.eval_gate_save:
            if "image_name" in samples.keys():
                id_lst = samples['image_name']
            elif "image_id" in samples.keys():
                id_lst = samples['image_id']
            try:
                self._save_gate(
                    samples['q_input'],
                    output_text,
                    gate,
                    id_lst,
                    gate_load,
                    os.path.join(self.gate_save_path, "eval_gate.txt")
                )
            except Exception as e:
                print("Evaluate save gate Error:", e)
                # : TODO Evaluate save gate Error: local variable 'id_lst' referenced before assignment

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

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - prompt: the instruction
            candidates:
                (list): A list of candidate class names;
            n_segments:
                (int): Split the candidates into n_segments and predict one by one. This is useful when the number of candidates is too large.
        Returns:
            output_class: predicted class index
        """

        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_t5 = self.t5_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
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

            inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            candidates, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        n_cands = len(candidates)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
            )

            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                # this_encoder_outputs = copy.deepcopy(encoder_outputs)
                this_encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0].clone(),
                )

                this_encoder_outputs['last_hidden_state'] = this_encoder_outputs[0].repeat_interleave(seg_len, dim=0)
                this_encoder_atts = encoder_atts.repeat_interleave(seg_len, dim=0)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens_ids = output_tokens.input_ids[start_i:end_i].repeat(bs, 1)
                this_output_tokens_atts = output_tokens.attention_mask[start_i:end_i].repeat(bs, 1)

                this_targets = this_output_tokens_ids.masked_fill(this_output_tokens_ids == self.t5_tokenizer.pad_token_id, -100)

                outputs = self.t5_model(
                    encoder_outputs=this_encoder_outputs,
                    attention_mask=this_encoder_atts,
                    decoder_attention_mask=this_output_tokens_atts,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )
                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

            # encoder_outputs['last_hidden_state'] = encoder_outputs[0].repeat_interleave(n_cands, dim=0)
            # encoder_atts = encoder_atts.repeat_interleave(n_cands, dim=0)
            # output_tokens.input_ids = output_tokens.input_ids.repeat(bs, 1)
            # output_tokens.attention_mask = output_tokens.attention_mask.repeat(bs, 1)

            # # compute the LM loss for each candidate (sum logprob across all tokens) and select the highest
            # targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)

            # outputs = self.t5_model(
            #     encoder_outputs=encoder_outputs,
            #     attention_mask=encoder_atts,
            #     decoder_attention_mask=output_tokens.attention_mask,
            #     return_dict=True,
            #     labels=targets,
            #     reduction="none",
            # )
            # loss = outputs.loss

            # loss = loss.reshape(bs, n_cands)
            # output_class_ranks = torch.argsort(loss, dim=-1) # (bs, num_candidates)

        return output_class_ranks

    def prepare_few_shot_embeds(self, samples):
        this_n_fs = random.choices(
            list(range(self.num_few_shot_examples + 1)),
            weights=[1 - self.few_shot_prob] + [self.few_shot_prob / self.num_few_shot_examples] * self.num_few_shot_examples
        )[0]

        if this_n_fs == 0:
            return None, None

        images = []
        text_input = []
        for sample in samples:
            for n in range(this_n_fs):
                images.append(sample['image'][n])
                text_input.append(sample['text_input'][n])
        images = torch.stack(images, dim=0)

        image = images

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text_input,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask = Qformer_atts,
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

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        if this_n_fs > 1:
            encoder_atts = encoder_atts.reshape(encoder_atts.size(0) // this_n_fs, encoder_atts.size(1) * this_n_fs)
            inputs_embeds = inputs_embeds.reshape(inputs_embeds.size(0) // this_n_fs, inputs_embeds.size(1) * this_n_fs, inputs_embeds.size(2))

        return inputs_embeds, encoder_atts



    def _extract_text_embed_by_qformer_pretrain_s1(
        self,
        text_input, 
        device
    ):  
        text_inputs = self.embed_extractor.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len, 
            return_tensors="pt",
        ).to(device)
        text_feats = self.embed_extractor.forward_text(text_inputs)
        # return text_feats.unsqueeze(1) # torch.Size([bz, 1, 768])

        text_embeds = F.normalize(self.embed_extractor.text_proj(text_feats))
        return text_embeds.unsqueeze(1) # torch.Size([bz, 1, 256])


    def _extract_text_embed_by_t5(
        self,
        text_input, 
        text_output, 
        device
    ):  
        bz = len(text_input)

        input_tokens = self.t5_tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        output_tokens = self.t5_output_tokenizer(
            text_output,
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            return_tensors="pt",
        ).to(device)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

        text_outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=input_tokens.attention_mask,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        last_token_embeds = list()
        for i in range(bz):
            seq_pos = (torch.nonzero(input_tokens.attention_mask[i]).squeeze())[-1].item() # 取最后<seq>位置
            last_token_embed = text_outputs.encoder_last_hidden_state[i][seq_pos]
            last_token_embeds.append(last_token_embed.unsqueeze(0))
        text_embeds = torch.concat(last_token_embeds, dim=0).unsqueeze(1) # torch.Size([bz, 1, 4096])

        return text_embeds

    def _save_gate(self, input_text, output_text, gate, id_lst, gate_load, gate_save_file):
        tt = list()
        for tinput, toutput, g, id_ in zip(input_text, output_text, gate, id_lst):
            tt.append({
                'text_input': tinput,
                'text_output': toutput,
                'gate': g.tolist(),
                'image': id_,
                'batch_gate_load': gate_load.tolist()
            })
        with open(gate_save_file, "a") as f:
            f.write(f"{json.dumps(tt)}\n")

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

        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        qformer_text_input = cfg.get("qformer_text_input", True)

        repeat_to_init_qt_candidates= cfg.get("repeat_to_init_qt_candidates", True)
        num_qt_candidates = cfg.get("num_qt_candidates", 5)
        moe_topk = cfg.get("moe_topk", 2)
        moe_position = cfg.get("moe_position", "pre")
        embed_extract = cfg.get("embed_extract", "t5")
        train_gate_save = cfg.get("train_gate_save", False)
        eval_gate_save = cfg.get("eval_gate_save", False)
        gate_save_path = cfg.get("gate_save_path", "")

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
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            qformer_text_input=qformer_text_input,
            repeat_to_init_qt_candidates=repeat_to_init_qt_candidates,
            num_qt_candidates=num_qt_candidates,
            moe_topk=moe_topk,
            moe_position=moe_position,
            embed_extract=embed_extract,
            eval_gate_save=eval_gate_save,
            train_gate_save=train_gate_save,
            gate_save_path=gate_save_path,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)
        
        # check update params
        print("Updating following parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name)

        return model
