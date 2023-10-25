import os
from PIL import Image
from tqdm import tqdm
import argparse
import pathlib

import torch
import clip

from load_aokvqa import load_aokvqa, get_coco_path


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--coco-dir', type=pathlib.Path, required=True, dest='coco_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--model-type', type=str, choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'], required=True, dest='model_type')
parser.add_argument('--out', type=pathlib.Path, required=True, dest='output_file')
args = parser.parse_args()

assert args.output_file.suffix == '.pt'

## Load dataset

dataset = load_aokvqa(args.aokvqa_dir, args.split)

## Load model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.model_type, device=device)

## Encoding loop

with torch.no_grad():
    embeddings = {}

    for d in tqdm(dataset):
        q = d["question"]
        q_text = clip.tokenize(q).to(device)
        q_text_features = model.encode_text(q_text)

        img = Image.open(get_coco_path(args.split, d['image_id'], args.coco_dir))
        img = preprocess(img).unsqueeze(0).to(device)
        image_features = model.encode_image(img)

        embeddings[d['question_id']] = {
            'question' : q_text_features[0].float().cpu(),
            'image' : image_features[0].float().cpu(),
        }

    torch.save(embeddings, args.output_file)
