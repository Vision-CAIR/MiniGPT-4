"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version
import re

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train

@registry.register_model("blip2_vicuna_instruct")
class Blip2VicunaInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from minigpt4.models import load_model
        >>> import torch
        >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b_qfmoe_route", device=device)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b_instruct": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna7b_pretrain": "configs/models/blip2/blip2_pretrain_vicuna7b.yaml",
        "vicuna7b_qfmoe_post": "configs/models/blip2/blip2_qformer_moe_post_vicuna7b.yaml",
        "vicuna7b_qfmoe_route": "configs/models/blip2/blip2_pretrain_vicuna7b_route_moe.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_llm=True,
        freeze_qformer=False,
        freeze_proj=False,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        use_moeqformer=False,
        use_route_moe=False,
        moebert_num_beams=2,
        moebert_expert_num=5,
        moebert_route_method="gate-sentence",
        moebert_load_balance = 0.1,
        moe_topk = 1,
        use_balance_loss = True,
        moe_weight_type = "l2_norm",
        gate_save_path = None,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        from minigpt4.models.modeling_llama import LlamaForCausalLM
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        print('Initing & Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            # freeze ln vision
            # for name, param in self.ln_vision.named_parameters():
                # param.requires_grad = False
            # self.ln_vision = self.ln_vision.eval()
            # self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder but not ln_vision")
        print('Loading VIT Done')

        if use_moeqformer:
            if use_route_moe:
                self.Qformer, self.query_tokens = self.init_RouteMoEQformer(
                    num_query_token=num_query_token, 
                    vision_width=self.visual_encoder.num_features,
                    moebert_expert_num=moebert_expert_num,
                    moebert_num_beams=moebert_num_beams,
                    route_method=moebert_route_method,
                    cross_attention_freq=2
                )
            else:
                self.Qformer, self.query_tokens = self.init_QformerMoE(
                    num_query_token=num_query_token, 
                    vision_width=self.visual_encoder.num_features,
                    moebert_expert_num=moebert_expert_num,
                    moebert_route_method=moebert_route_method,
                    moebert_load_balance=moebert_load_balance,
                    moe_topk=moe_topk,
                    use_balance_loss=use_balance_loss,
                    moe_weight_type=moe_weight_type,
                    cross_attention_freq=2
                )
        else:
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )

        # import pdb;pdb.set_trace()
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        print("Loading LLM")
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        # self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        # self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        # self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]
        if freeze_llm:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        if qformer_text_input:
            # Hard-coded to load from BLIP-2 stage-1 pre-trained model( to init ffn but not ideal)
            self.load_from_pretrained(
                url_or_filename="/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2_pretrained/blip2_pretrained.pth",
                num_query_token=num_query_token
            )
        
        if use_moeqformer:
            # load blip2_vicuna_pretrain to init query_ffn
            self.load_from_pretrained(
                url_or_filename=q_former_model,
                num_query_token=num_query_token
            )

            # init MoE Layer(init moe ffn by blip2 query ffn)
            state_dict = self.Qformer.state_dict()
            for name, param in self.Qformer.named_parameters():
                if "_query" in name and "experts.experts" in name:
                    pattern = r'\.experts\.experts\.\d+'
                    key_orig = re.sub(pattern, '', name)
                    param.data.copy_(state_dict[key_orig]) # copy state_dict[key_orig] to param
                if "experts.intermediate_query" in name or "experts.output_query" in name:
                    key_orig = re.sub(r'experts\.', '', name)
                    param.data.copy_(state_dict[key_orig]) # copy state_dict[key_orig] to param
                if "_query" in name and "experts" not in name: # raw ffn_query not update
                    param.requires_grad = False

        # freeze qformer        
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            logging.info("freeze Qformer")

        # After loading, freeze llm_proj
        if freeze_proj:
            for name, param in self.llm_proj.named_parameters():
                param.requires_grad = False
            self.llm_proj = self.llm_proj.eval()
            self.llm_proj.train = disabled_train

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        self.use_moeqformer = use_moeqformer
        self.use_route_moe = use_route_moe
        self.moebert_load_balance = moebert_load_balance

        self.gate_save_path = gate_save_path
        # if self.gate_save_path != None:
            # import os
            # if not os.path.exists(self.gate_save_path):
                # os.mkdir(self.gate_save_path)


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')
        # import pdb;pdb.set_trace()
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        bs = image.size(0)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["q_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_hidden_states=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_hidden_states=True,
            )

        query_output_to_linear = query_output.last_hidden_state[:,:query_tokens.size(1),:]
        
        if self.use_moeqformer and not self.use_route_moe:
            gate_loss = query_output.gate_loss # only available in QformerMoE
        
        if self.gate_save_path != None:
            all_hidden_states = query_output.hidden_states
            # prob_gate_normalized = query_output.gate_loads
            beam_scores = query_output.beam_scores
            expert_route = query_output.expert_route
            
            gate_route = list()
            import numpy as np
            import json
            import os
            try:
                for i in range(len(samples['image_id'])):
                    image_id = samples['image_id'][i]
                    gate_route.append({
                        'iters': samples['iters'],
                        'image_id':image_id,
                        'q_input': samples['q_input'][i],
                        'text_output': samples['text_output'][i],
                        'beam_scores': beam_scores[i].tolist(),
                        'expert_route': expert_route[i].tolist(),
                        # 'gate_route_11': prob_gate_normalized[10][i].tolist(),
                        # 'gate_route_9': prob_gate_normalized[8][i].tolist(),
                        # 'gate_route_7': prob_gate_normalized[6][i].tolist(),
                        # 'gate_route_5': prob_gate_normalized[4][i].tolist(),
                        # 'gate_route_3': prob_gate_normalized[2][i].tolist(),
                        # 'gate_route_1': prob_gate_normalized[0][i].tolist(),
                    })
                    # for layer in [6,8,10]:
                    #     layer_data  = all_hidden_states[layer]
                    #     file_path = os.path.join(self.gate_save_path, f'{image_id}_{str(layer)}.npy')
                    #     x = layer_data.data.cpu().numpy()
                    #     np.save(file_path,x) 

                with open(os.path.join(self.gate_save_path, 'train_save_beam.json'),'a+') as f:
                    f.write(f"{json.dumps(gate_route)}\n")
            except Exception as e:
                print("Gate Save Error....")
                print(e)


        inputs_llm = self.llm_proj(query_output_to_linear)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['llm_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        if self.use_moeqformer and not self.use_route_moe:
            loss = outputs.loss + self.moebert_load_balance * gate_loss
        else:
            loss = outputs.loss

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
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        image = samples["image"]
        bs = image.size(0)

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["q_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

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
                output_hidden_states=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_hidden_states=True,
            )

        if self.gate_save_path != None:
            all_hidden_states = query_output.hidden_states
            # prob_gate_normalized = query_output.gate_loads
            beam_scores = query_output.beam_scores
            expert_route = query_output.expert_route
            
            import numpy as np
            import json
            import os
            gate_route = list()
            try:
                for i in range(len(samples['image_id'])):
                    source = samples['source'][i]
                    if source in ['gqa']:
                        image_id =  samples['image_id'][i].split('.')[0]
                    else:
                        image_id = samples['image_id'][i].split('/')[-1].split('.')[0]
                    gate_route.append({
                        'source': source,
                        'image_id':image_id,
                        'q_input': samples['q_input'][i],
                        'beam_scores': beam_scores[i].tolist(),
                        'expert_route': expert_route[i].tolist(),
                        # 'gate_route_11': prob_gate_normalized[10][i].tolist(),
                        # 'gate_route_9': prob_gate_normalized[8][i].tolist(),
                        # 'gate_route_7': prob_gate_normalized[6][i].tolist(),
                        # 'gate_route_5': prob_gate_normalized[4][i].tolist(),
                        # 'gate_route_3': prob_gate_normalized[2][i].tolist(),
                        # 'gate_route_1': prob_gate_normalized[0][i].tolist(),
                    })
                    for layer in [6,8,10]:
                        if layer == 6:
                            layer_data  = all_hidden_states[layer][i, :32, :]
                        else:
                            layer_data  = all_hidden_states[layer][i*3, :32, :]
                        file_path = os.path.join(self.gate_save_path, f'{image_id}_{str(layer)}.npy')
                        x = layer_data.data.cpu().numpy()
                        np.save(file_path,x) # 大功告成

                with open(os.path.join(self.gate_save_path, 'generate_save_beam.json'),'a+') as f:
                    f.write(f"{json.dumps(gate_route)}\n")
            except Exception as e:
                print("Gate Save Error....")
                print(e)

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            samples['llm_input'],
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

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
        length_penalty=0,
        **kwargs
    ):
        if isinstance(samples["llm_input"], str):
            samples["llm_input"] = [samples["llm_input"]]

        # if prompt:
        #     if prompt.count("{}") == 2:
        #         if 'ocr_tokens' in samples:
        #             text_input = [
        #                 prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["llm_input"][i])
        #             for i in range(len(samples["llm_input"]))]
        #         elif 'choices' in samples:
        #             text_input = []
        #             for i in range(len(samples["llm_input"])):
        #                 this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
        #                 this_choices = " ".join(this_choices)
        #                 text_input.append(prompt.format(samples["llm_input"][i], this_choices))
        #     else:
        #         text_input = [prompt.format(question) for question in samples["llm_input"]]
        # else:
        #     text_input = samples["llm_input"]

        # samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
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
        q_former_model = cfg.get("q_former_model", "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2_vicuna7b/blip2_pretrained_vicuna7b.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_llm = cfg.get("freeze_llm", True)
        freeze_qformer = cfg.get("freeze_qformer", False)
        freeze_proj = cfg.get("freeze_proj", False)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        use_moeqformer = cfg.get("use_moeqformer", False)
        use_route_moe = cfg.get("use_route_moe", False)
        moebert_num_beams = cfg.get("moebert_num_beams", 2)
        moebert_expert_num = cfg.get("moebert_expert_num", 5)
        moebert_route_method = cfg.get("moebert_route_method", "gate-sentence")
        moebert_load_balance = cfg.get("moebert_load_balance", 0.1)
        moe_topk = cfg.get("moe_topk", 1)
        use_balance_loss = cfg.get("use_balance_loss", True)
        moe_weight_type = cfg.get("moe_weight_type",'l2_norm')
        gate_save_path = cfg.get("gate_save_path", None)

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
            freeze_proj=freeze_proj,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            use_moeqformer=use_moeqformer,
            use_route_moe=use_route_moe,
            moebert_num_beams=moebert_num_beams,
            moebert_expert_num=moebert_expert_num,
            moebert_route_method=moebert_route_method,
            moebert_load_balance=moebert_load_balance,
            moe_topk=moe_topk,
            use_balance_loss=use_balance_loss,
            moe_weight_type=moe_weight_type,
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
