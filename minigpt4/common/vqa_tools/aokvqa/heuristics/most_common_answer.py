import os
import json
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


train_set = load_aokvqa(args.aokvqa_dir, 'train')
train_freq = dict(Counter(
    [d['choices'][d['correct_choice_idx']] for d in train_set]
))
most_common_answer = max(train_freq.keys(), key=train_freq.get)

##

eval_set = load_aokvqa(args.aokvqa_dir, args.split)

predictions = {}

for d in eval_set:
    q = d['question_id']
    predictions[q] = most_common_answer

    if args.multiple_choice:
        choices = [c for c in d['choices'] if c in train_freq.keys()]
        if len(choices) > 0:
            predictions[q] = max(choices, key=train_freq.get)

json.dump(predictions, args.output_file)
