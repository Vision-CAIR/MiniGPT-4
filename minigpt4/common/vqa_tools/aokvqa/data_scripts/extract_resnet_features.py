import os
import argparse
import pathlib
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T

from load_aokvqa import load_aokvqa, get_coco_path


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--coco-dir', type=pathlib.Path, required=True, dest='coco_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--out', type=pathlib.Path, required=True, dest='output_file')
args = parser.parse_args()

assert args.output_file.suffix == '.pt'

## Load dataset

dataset = load_aokvqa(args.aokvqa_dir, args.split)

## Load model

resnet_preprocess = T.Compose([
    T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(size=(224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_model = models.resnet50(pretrained=True)
resnet_model = torch.nn.Sequential(
    *list(resnet_model.children())[:-1],
    nn.Flatten()
)  # strip classification layer
resnet_model = resnet_model.to(device)

## Encoding loop

with torch.no_grad():
    embeddings = {}

    for d in tqdm(dataset):
        img = Image.open(get_coco_path(args.split, d['image_id'], args.coco_dir)).convert('RGB')
        resnet_input = resnet_preprocess(img).unsqueeze(0).to(device)
        resnet_features = resnet_model(resnet_input)
        embeddings[d['question_id']] = {
            'image' : resnet_features[0].cpu()
        }

    torch.save(embeddings, args.output_file)
