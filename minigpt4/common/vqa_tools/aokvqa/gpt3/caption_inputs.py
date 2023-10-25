import os
import json
import argparse
import pathlib

from load_aokvqa import load_aokvqa


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--coco-dir', type=pathlib.Path, required=True, dest='coco_dir')
parser.add_argument('--split', type=str, choices=['train', 'val'], required=True)
parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
args = parser.parse_args()

aokvqa_set = load_aokvqa(args.aokvqa_dir, args.split)

coco_captions = json.load(open(os.path.join(args.coco_dir, 'annotations', f'captions_{args.split}2017.json')))['annotations']
coco_captions = {c['image_id'] : c['caption'] for c in coco_captions}

captions = { d['question_id'] : coco_captions[d['image_id']] for d in aokvqa_set }

json.dump(captions, args.output_file)
