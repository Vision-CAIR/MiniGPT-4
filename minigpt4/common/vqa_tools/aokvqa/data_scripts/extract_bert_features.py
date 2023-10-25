import os
import argparse
import pathlib
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

from load_aokvqa import load_aokvqa


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--out', type=pathlib.Path, required=True, dest='output_file')
args = parser.parse_args()

assert args.output_file.suffix == '.pt'

## Load dataset

dataset = load_aokvqa(args.aokvqa_dir, args.split)

## Load model

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

## Encoding loop

with torch.no_grad():
    embeddings = {}

    for d in tqdm(dataset):
        encoded_input = tokenizer([d['question']], padding=True, return_tensors='pt')
        encoded_input = {k:v.to(device) for k,v in encoded_input.items()}
        e = mean_pooling(model(**encoded_input), encoded_input['attention_mask'])
        embeddings[d['question_id']] = {
            'question' : e[0].cpu()
        }

    torch.save(embeddings, args.output_file)
