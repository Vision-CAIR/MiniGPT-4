import os
import argparse
from collections import Counter
import pathlib

from load_aokvqa import load_aokvqa


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--out', type=pathlib.Path, required=True, dest='output_file')
args = parser.parse_args()


# Build vocab from train set: correct choices + (direct answers appearing in >= 3 )

train_set = load_aokvqa(args.aokvqa_dir, 'train')

vocab = []
all_choices = Counter()
direct_answers = Counter()

for i in train_set:
    vocab.append( i['choices'][i['correct_choice_idx']] )
    all_choices.update(i['choices'])
    direct_answers.update(set(i['direct_answers']))
vocab += [k for k,v in all_choices.items() if v >= 3]
vocab += [k for k,v in direct_answers.items() if v >= 3]

vocab = sorted(set(vocab))
print(f"Vocab size: {len(vocab)}")

# Save vocabulary Output

with open(args.output_file, 'w') as f:
    for v in vocab:
        print(v, file=f)

## Check validation set coverage

val_set = load_aokvqa(args.aokvqa_dir, 'val')

val_acc = [v['choices'][v['correct_choice_idx']] in vocab for v in val_set]
val_acc = sum(val_acc) / len(val_acc) * 100
print(f"Val set coverage: {val_acc:.2f}" )
