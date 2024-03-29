"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

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
    Compare Qformer & QformerMoE
"""
# blip2_qformer
# calculate parameters
from minigpt4.models import load_model
model = load_model("blip2", "pretrain")
model.QformerMoE, model.query_tokens_moe = model.init_QformerMoE(
    num_query_token, model.visual_encoder.num_features, cross_attention_freq
)
model.Qformer, model.query_tokens = model.init_Qformer(
    num_query_token, model.visual_encoder.num_features, cross_attention_freq
)
state_dict = model.Qformer.state_dict()
for name, param in model.Qformer.named_parameters():
    if "_query" in name:
        key_orig = name.replace("_query", "")
        param.data.copy_(state_dict[key_orig])
        if "10" in name:
            print(name)


"""
    blip2_t5_qformer_moe
    Calculate Num Parameters
"""
import torch
import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE")
from minigpt4.models import model_zoo
from minigpt4.models import load_model
print(model_zoo)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = load_model("blip2_t5_qformer_moe", "flant5xxl", device=device)

num_parameters=0
for n, p in model.Qformer.named_parameters():
    if not p.requires_grad:
        continue  # frozen weights
    if "11.experts.experts" in n:
        print(n)
        num_parameters += p.data.nelement()
print(num_parameters) # 23,619,840
# total trainable parameter: 415,631,104

num_parameters=0
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue  # frozen weights
    num_parameters += p.data.nelement()
print(num_parameters) # 23,619,840
# total trainable parameter: 415,631,104


num_parameters=0
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue  # frozen weights
    if 'Qformer.bert.encoder.layer.6.crossattention' in n:
        num_parameters += p.data.nelement()
    # if 'Qformer.bert.encoder.layer.11.output' in n:
    #     num_parameters += p.data.nelement()
print(num_parameters)


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
    # 'image': torch.randn(4, 3, 336, 336).half().to(device)
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

