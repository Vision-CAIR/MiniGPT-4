import os
import json
import numpy as np
import argparse
import pathlib
from collections import Counter

from load_aokvqa import load_aokvqa


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--mc', action='store_true', dest='multiple_choice')
parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
args = parser.parse_args()

np.random.seed(0)

train_set = load_aokvqa(args.aokvqa_dir, 'train')
train_freq = dict(Counter(
    [d['choices'][d['correct_choice_idx']] for d in train_set]
))

if args.multiple_choice is False:
    choices = list(train_freq.keys())
    probs = [f / len(train_set) for f in train_freq.values()]

##

predictions = {}

eval_set = load_aokvqa(args.aokvqa_dir, args.split)

for d in eval_set:
    if args.multiple_choice:
        choices = d['choices']
        probs = [train_freq.get(c, 0) for c in choices]
        if probs == [0, 0, 0, 0]:
            probs = [1, 1, 1, 1]
        probs = [p / sum(probs) for p in probs]

    q = d['question_id']
    predictions[q] = np.random.choice(choices, size=1, p=probs)[0]

json.dump(predictions, args.output_file)
