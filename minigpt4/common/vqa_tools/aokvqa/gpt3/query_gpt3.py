import os
import random
import json
from tqdm import tqdm
import argparse
import pathlib

import openai
openai.organization = os.getenv('OPENAI_ORG')
openai.api_key = os.getenv('OPENAI_API_KEY')

from load_aokvqa import load_aokvqa


random.seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--n', type=int, default=10, dest='num_examples')
    parser.add_argument('--train-context', type=argparse.FileType('r'), dest='train_context_file')
    parser.add_argument('--prefix', type=str, default='', dest='prompt_prefix')
    parser.add_argument('--include-choices', action='store_true', dest='include_choices')
    parser.add_argument('--context', type=argparse.FileType('r'), dest='context_file')
    parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
    args = parser.parse_args()


    train_set = load_aokvqa(args.aokvqa_dir, 'train')
    eval_set = load_aokvqa(args.aokvqa_dir, args.split)

    train_context = {}
    context = {}
    if args.context_file is not None:
        train_context = json.load(args.train_context_file)
        context = json.load(args.context_file)

    predictions = {}

    for d in tqdm(eval_set):
        q = d['question_id']

        prompt = args.prompt_prefix
        for e in random.sample(train_set, args.num_examples):
            prompt += prompt_element(e,
                context=train_context.get(q, None),
                include_choices=args.include_choices,
                answer=True
            )
            prompt += '\n\n'

        prompt += prompt_element(d,
            context=context.get(q, None),
            include_choices=args.include_choices,
            answer=False
        )

        response = openai.Completion.create(
            engine="text-curie-001",
            prompt=prompt,
            temperature=0.0,
            max_tokens=10,
        )

        predictions[q] = response.choices[0].text.strip()

    json.dump(predictions, args.output_file)


def prompt_element(d, context=None, include_choices=False, answer=False):
    return (f"Context: {context}\n" if context is not None else '') + \
            f"Q: {d['question']}\n" + \
           (f"Choices: {', '.join(d['choices'])}.\n" if include_choices else '') + \
            f"A:" + (f" {d['choices'][d['correct_choice_idx']]}" if answer else '')

if __name__ == '__main__':
    main()
