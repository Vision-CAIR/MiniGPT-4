import sys
import os
import argparse
import pathlib
from tqdm import tqdm
import json

import torch
import torch.nn as nn

# https://github.com/PyTorchLightning/pytorch-lightning/issues/11663
import sentencepiece; import pytorch_lightning as pl; import clip

from transfer_experiments.train import LinearClassifier
from load_aokvqa import load_aokvqa
from evaluation.remap_predictions import map_to_choices


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--features', type=pathlib.Path, required=True)
parser.add_argument('--out', type=argparse.FileType('w'), dest='output_file')
#
parser_weights = parser.add_mutually_exclusive_group(required=True)

parser_weights.add_argument('--ckpt', type=pathlib.Path, dest='checkpoint_path')

parser_weights.add_argument('--zero-shot', action='store_true', dest='clip_zero_shot')
parser.add_argument('--inputs', nargs='+', type=str, choices=['question', 'image'], required=('--zero-shot' in sys.argv))
#
parser.add_argument('--vocab', type=argparse.FileType('r'))
parser.add_argument('--vocab-features', type=pathlib.Path, dest='vocab_features')
parser.add_argument('--mc', action='store_true', dest='multiple_choice')

parser.add_argument('--clip-model-type', type=str,
                    choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
                    dest='clip_model_type', required=('--zero-shot' in sys.argv and '--mc' in sys.argv))
#
args = parser.parse_args()


## Load dataset

aokvqa_set = load_aokvqa(args.aokvqa_dir, args.split)

## Load models

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.checkpoint_path is not None:
    classifier = LinearClassifier.load_from_checkpoint(args.checkpoint_path)
    classifier.to(device)
    hp = classifier.hparams
elif args.clip_zero_shot:
    classifier = nn.Identity().to(device)
    hp = pl.utilities.AttributeDict(backbone='clip', clip_model_type=args.clip_model_type, objective='zero-shot', inputs=args.inputs)

# Load input features

embeddings = torch.load(args.features)
if hp.backbone == 'clip':
    for q in embeddings.keys():
        embeddings[q]['question'] = embeddings[q]['question'] / embeddings[q]['question'].norm(dim=-1, keepdim=True)
        embeddings[q]['image'] = embeddings[q]['image'] / embeddings[q]['image'].norm(dim=-1, keepdim=True)

# Load vocab, vocab features, clip

if (hp.objective == 'classifier') or \
   (hp.objective in ['contrastive', 'zero-shot'] and args.multiple_choice is False):
        vocab = args.vocab.read().splitlines()

if hp.objective in ['contrastive', 'zero-shot']:
    if args.multiple_choice is False:
        vocab_features = torch.load(args.vocab_features).cpu()
        vocab_features /= vocab_features.norm(dim=-1, keepdim=True)
    else:
        clip_model = clip.load(hp.clip_model_type, device=device)[0]
        logit_scale = clip_model.logit_scale.exp().cpu()

## Prediction loop

predictions = {}

with torch.no_grad():
    for o in tqdm(aokvqa_set):
        q = o['question_id']

        # Load input embedding (from question / image)
        if hp.objective == 'zero-shot' and ('question' in hp.inputs and 'image' in hp.inputs):
            e = embeddings[q]['question'] + embeddings[q]['image']
        elif 'question' in hp.inputs and 'image' in hp.inputs:
            e = torch.cat((embeddings[q]['question'], embeddings[q]['image']))
        elif 'question' in hp.inputs:
            e = embeddings[q]['question']
        elif 'image' in hp.inputs:
            e = embeddings[q]['image']

        # Pass inputs through model
        e = e.unsqueeze(0).to(device)
        x = classifier(e)[0].cpu()

        # Predict
        if hp.objective in ['contrastive', 'zero-shot']:
            if args.multiple_choice:
                vocab = o['choices']
                # Encode choices
                vocab_features = clip.tokenize(vocab).to(device)
                vocab_features = torch.stack([
                    clip_model.encode_text(v.unsqueeze(0)) for v in vocab_features
                ], dim=1)[0]
                vocab_features /= vocab_features.norm(dim=-1, keepdim=True)
                vocab_features = vocab_features.float().cpu()

            x = logit_scale * x @ vocab_features.t()
            x = x.softmax(dim=-1)

        predictions[q] = vocab[x.argmax().item()]

## Save and evaluate predictions

# Map prediction to nearest neighbor choice (by word embeddings)
if args.multiple_choice and hp.objective == 'classifier':
    predictions = map_to_choices(aokvqa_set, predictions)

json.dump(predictions, args.output_file)
