import json
from tqdm import tqdm
import argparse
import pathlib

import torch
import clip

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', type=pathlib.Path, required=True, dest='vocab_file')
parser.add_argument('--model-type', type=str, choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'], required=True, dest='model_type')
parser.add_argument('--out', type=pathlib.Path, required=True, dest='output_file')
args = parser.parse_args()

assert args.output_file.suffix == '.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.model_type, device=device)

with torch.no_grad():
    a = open(args.vocab_file).read().splitlines()
    mc_text = clip.tokenize(a).to(device)
    mc_text_features = torch.stack([model.encode_text(mct.unsqueeze(0)).cpu() for mct in tqdm(mc_text)], dim=1)[0]
    mc_text_features = mc_text_features.float()
    model_name = args.model_type.replace('/', '-').replace('@', '-')
    torch.save(mc_text_features, args.output_file)
