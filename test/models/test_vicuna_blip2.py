import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE")
from minigpt4.models import load_model
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = load_model("blip2_vicuna_instruct", "vicuna7b_pretrain", device="cpu")
# model = load_model("blip2_vicuna_instruct", "vicuna7b_instruct", device=device)
model = load_model("blip2_vicuna_instruct", "vicuna7b_qfmoe_post", device=device)


use_nucleus_sampling=False
num_beams=5
max_length=256
min_length=1
top_p=0.9
repetition_penalty=1.5
length_penalty=1
num_captions=1
temperature=1


if "prompt" in samples.keys():
    prompt = samples["prompt"]
else:
    prompt = model.prompt

image = samples["image"]

bs = image.size(0)

if isinstance(prompt, str):
    prompt = [prompt] * bs
else:
    assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

# For TextCaps
if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
    prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

query_tokens = model.query_tokens.expand(bs, -1, -1)

text_Qformer = model.tokenizer(
    prompt,
    padding='longest',
    truncation=True,
    max_length=model.max_txt_len,
    return_tensors="pt",
).to(image.device)
query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

with model.maybe_autocast():
    image_embeds = model.ln_vision(model.visual_encoder(image))
image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

query_output = model.Qformer.bert(
    text_Qformer.input_ids,
    attention_mask=Qformer_atts,
    query_embeds=query_tokens,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_atts,
    return_dict=True,
)
inputs_llm = model.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

llm_tokens = model.llm_tokenizer(
    prompt,
    padding="longest",
    return_tensors="pt"
).to(image.device)

inputs_embeds = model.llm_model.get_input_embeddings()(llm_tokens.input_ids)
inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
print(inputs_embeds.shape)

inputs_embeds0 = inputs_embeds[:2]
attention_mask0 = attention_mask[:2]

outputs = model.llm_model.generate(
    inputs_embeds=inputs_embeds0, # torch.Size([4, 41, 4096])
    attention_mask=attention_mask0, # torch.Size([4, 41])
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

from PIL import Image
image = "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/examples_v2/KFC-20-for-20-Nuggets.jpg"
raw_image = Image.open(image).convert('RGB')
image = model.vis_processor(raw_image).unsqueeze(0).to("cpu")
sample = {'text_input': ["What is around the open window?", "Is the ground blue or brown?"],
          'image':torch.randn(2, 3, 224, 224).to("cpu")}

samples = {
    'text_input':["What is around the open window?",  # n23181
                    "Is the ground blue or brown?", # n168412
                    "What color are the pants?", # n446242
                    "What is the airplane flying above?"], # n414992
    'image': torch.randn(4, 3, 224, 224).to("cpu")
}


for key in mo['model'].keys():
    if 'Qformer' not in key:
        print(key)