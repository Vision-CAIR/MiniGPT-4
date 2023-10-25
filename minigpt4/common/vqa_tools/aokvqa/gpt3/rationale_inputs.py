import json
import argparse
import pathlib

from load_aokvqa import load_aokvqa


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test_w_ans'], required=True)
parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
args = parser.parse_args()

aokvqa_set = load_aokvqa(args.aokvqa_dir, args.split)
rationales = {d['question_id'] : d['rationales'][0] for d in aokvqa_set}
json.dump(rationales, args.output_file)
