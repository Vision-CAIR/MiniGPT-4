import os
import json
from random import seed, sample
import argparse
import pathlib

from load_aokvqa import load_aokvqa


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--mc', action='store_true', dest='multiple_choice')
parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
args = parser.parse_args()

seed(0)

train_set = load_aokvqa(args.aokvqa_dir, 'train')

if args.multiple_choice is False:
    choices = list(set(
        [d['choices'][d['correct_choice_idx']] for d in train_set]
    ))

##

predictions = {}

eval_set = load_aokvqa(args.aokvqa_dir, args.split)

for d in eval_set:
    q = d['question_id']
    if args.multiple_choice:
        choices = d['choices']
    predictions[q] = sample(choices, 1)[0]

json.dump(predictions, args.output_file)
