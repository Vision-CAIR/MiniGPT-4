import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData

from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
import random


def list_of_str(arg):
    return list(map(str, arg.split(',')))


parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--split", type=list_of_str, default='testB', help="dataset split to evaluate")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
parser.add_argument("--img_path", type=str)
parser.add_argument("--eval_file_path", type=str)
args = parser.parse_args()

print(args.ckpt)
print(args.name)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_LLama2.copy()
conv_temp.system = ""

model.eval()

os.makedirs('results', exist_ok=True)

if 'okvqa' in args.dataset:
    img_path=os.path.join(args.img_path,"train")
    with open(os.path.join(args.eval_file_path,"ok_vqa/test_split.json")) as f:
        ok_vqa_test_split = json.load(f)

    data = OKVQAEvalData(ok_vqa_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    minigpt4_predict = []

    resamples = []

    for images, questions, question_ids, img_ids in eval_dataloader:
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)

        for answer, question_id, question, img_id in zip(answers, question_ids, questions, img_ids):
            result = dict()
            if "<unk>" in answer.lower():
                print("answer: ", answer)
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question_id'] = int(question_id)
            if answer == "":
                resamples.append({'image_id': img_id, 'question_id':question_id, 'question': [question.replace('[vqa] Based on the image, respond to this question with a short answer:','').strip()]})
            else:
                minigpt4_predict.append(result)

    if args.resample:
        for i in range(20):
            data = OKVQAEvalData(resamples, vis_processor, img_path)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
            for images, questions, question_ids, img_ids in eval_dataloader:
                texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)
                for answer, question_id, question in zip(answers, question_ids, questions):
                    result = dict()
                    answer = answer.lower().replace('<unk>','').strip()
                    result['answer'] = answer
                    result['question_id'] = int(question_id)
                    minigpt4_predict.append(result)
                    if answer == "":
                        resamples.append({'image_id': img_id, 'question_id':question_id, 'question': [question.replace('[vqa] Based on the image, respond to this question with a short answer:','').strip()]})
                    else:
                        minigpt4_predict.append(result)
            if len(resamples) == 0:
                break

    save_path=f'results/{args.name}_okvqa.json'
    with open(save_path,'w') as f:
        json.dump(minigpt4_predict, f)

    annFile     =f'{args.eval_file_path}/ok_vqa/mscoco_val2014_annotations_clean.json'
    quesFile    =f'{args.eval_file_path}/ok_vqa/OpenEnded_mscoco_val2014_questions_clean.json'

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(save_path, quesFile)

    vqaEval = VQAEval(vqa, vqaRes, n=2)

    vqaEval.evaluate()

    print ("Overall OKVQA Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), flush=True)

if 'vizwiz' in args.dataset:
    img_path=f'{args.img_path}/vizwiz/val'
    vizwiz = json.load(open(f'{args.eval_file_path}/vizwiz/val.json', 'r'))

    data = VizWizEvalData(vizwiz, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    minigpt4_predict = []
    total_acc = []
    for images, texts, gt_answers in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)

        for answer, gt_answer in zip(answers, gt_answers):
            result = dict()
            result['answer'] = answer.replace('<unk>','').strip()
            minigpt4_predict.append(result)
            count=0
            gt_answer = gt_answer.split('_')
            for gt in gt_answer:
                if gt.lower() == answer.lower():
                    count += 1
            acc = min(count/3.0, 1.0)
            total_acc.append(acc)
        
    save_path=f'results/{args.name}_vizwiz.json'
    with open(save_path,'w') as f:
        json.dump(minigpt4_predict, f)

    print('vizwiz Acc: ', np.average(total_acc)* 100.0, flush=True)

if 'aokvqa' in args.dataset:
    img_path=f'{args.img_path}/aokvqa/images'

    for split in args.split:
        with open(f'{args.eval_file_path}/aokvqa/annotations/aokvqa_v1p0_{split}.json','r') as f:
            aokvqa_v1p0 = json.load(f)
            
        data = AOKVQADAEvalData(aokvqa_v1p0, vis_processor, img_path)
        eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

        minigpt4_predict = defaultdict(dict)

        for images, texts, question_ids in tqdm(eval_dataloader):
            texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)

            for answer, question_id in zip(answers, question_ids):
                minigpt4_predict[question_id]['direct_answer'] = answer.lower().replace('<unk>','').strip()

        data = AOKVQAMCEvalData(aokvqa_v1p0, vis_processor, img_path)
        eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

        for images, texts, question_ids, answers in tqdm(eval_dataloader):
            instructions = ["[INST] <Img><ImageHere></Img> {} [/INST]".format(text) for text in texts]
            answer_ranks = model.multi_select(images, instructions, answers)
            candidates = [list(x) for x in zip(*answers)]
            for idx, question_id in enumerate(question_ids):
                minigpt4_predict[question_id]['multiple_choice'] = candidates[idx][answer_ranks[idx][0]]

        save_path=f'results/{args.name}_a_okvqa_{split}.json'
        with open(save_path,'w') as f:
            json.dump(minigpt4_predict, f)

        os.chdir('minigpt4/common/vqa_tools/aokvqa')
        print(os.system(f'python evaluation/eval_predictions.py --aokvqa-dir {args.eval_file_path}/aokvqa/annotations --split {split} --preds ../../../../{save_path}'), flush=True)
        os.chdir('../../../../')

if 'iconqa' in args.dataset:
    iconqa_text_val = json.load(open(f'{eval_file_path}/iconqa/choose_text_val.json','r'))
    img_path = f'{args.img_path}/iconqa/val/choose_txt'
    data = IconQAEvalData(iconqa_text_val, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    count = 0
    for images, texts, candidates, answers in tqdm(eval_dataloader):
        candidates = [candidate.split('_') for candidate in candidates]
        num_cand = [len(candidate) for candidate in candidates]
        for candidate in candidates:
            candidate.extend(['none'] * (max(num_cand) - len(candidate)))
        candidates = [list(x) for x in zip(*candidates)]
        instructions = ["[INST] <Img><ImageHere></Img> {} [/INST]".format(text) for text in texts]
        answer_ranks = model.multi_select(images, instructions, candidates, num_cand=num_cand)
        for idx, answer in enumerate(answers):
            if answer_ranks[idx][0] == answer:
                count += 1

    print('iconqa Acc: ', count / len(iconqa_text_val) * 100.0, flush=True)


if 'gqa' in args.dataset:
    img_path = f'{args.img_path}/gqa/images/val'
    gqa = json.load(open(f'{args.eval_file_path}/gqa/annotations/testdev_balanced_questions.json', 'r'))
    data = GQAEvalData(gqa, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    count=0
    total=0
    minigpt4_predict = []
    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            result['pred'] = answer.lower().replace('<unk>','').strip()
            result['gt'] = label
            minigpt4_predict.append(result)
            if answer.lower() == label:
                count+=1
            total+=1
    print('gqa val:', count / total * 100, flush=True)

    save_path=f'results/{args.name}_gqa.json'
    with open(save_path,'w') as f:
        json.dump(minigpt4_predict, f)

if 'vsr' in args.dataset:
    annotation = load_dataset("cambridgeltl/vsr_zeroshot", split='test')
    img_path = f'{args.img_path}/vsr/images'
    data = VSREvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    count=0
    total=0

    minigpt4_predict = []

    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        # print("texts",texts)
        answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            print(answer)
            result = dict()
            result['pred'] = answer.replace('<unk>','').strip()
            result['gt'] = label
            minigpt4_predict.append(result)
            if answer.lower() ==  label.lower():
                count+=1
            total+=1
    print('vsr test:', count / total * 100, flush=True)
    save_path=f'results/{args.name}_vsr.json'
    with open(save_path,'w') as f:
        json.dump(minigpt4_predict, f)

if 'hm' in args.dataset:
    img_path = f'{args.img_path}/hateful_meme'
    annotation = []
    with open(f'{args.eval_file_path}/hateful_meme/dev.jsonl', 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            annotation.append(json_obj)

    data = HMEvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    count=0
    total=0

    minigpt4_predict = []

    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            answer = 1 if answer.lower().__contains__('yes') else 0
            result['pred'] = int(str(answer).replace('<unk>','').strip())
            result['gt'] = int(label)
            minigpt4_predict.append(result)
            if answer == label:
                count+=1
            total+=1
    print('hm val:', count / total * 100, flush=True)

    save_path=f'results/{args.name}_hm.json'
    with open(save_path,'w') as f:
        json.dump(minigpt4_predict, f)
