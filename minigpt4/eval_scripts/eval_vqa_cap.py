# python eval_vqa.py --dataset vizwiz
import os
import re
import json
import argparse
from collections import defaultdict
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE")

from minigpt4.common.logger import setup_logger
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,VSREvalData,HMEvalData
from minigpt4.datasets.datasets.coco_caption import NoCapsEvalDataset
from minigpt4.common.vqa_tools.vqa import VQA
from minigpt4.common.vqa_tools.vqa_eval import VQAEval
from minigpt4.common.config import Config
from minigpt4.common.caption_tools.caption_utils import nocap_eval


def list_of_str(arg):
    return list(map(str, arg.split(',')))

def eval_parser():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--device",
        default=3,
        help="device to run the model",
    )
    parser.add_argument(
        "--cfg-path", 
        default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/minigpt4/projects/qformer_moe_vicuna/eval/vqa_benchmark_evaluation.yaml",
        help="path to configuration file.")
    parser.add_argument(
        "--dataset",
        # default=['vizwiz','hm','vsr'],
        default=['nocap'],
        type=list_of_str,
        help="dataset to evaluate",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    return parser

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def init_model(cfg, device):
    print('Initialization Model')
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)

    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    txt_processor_cfg  = cfg.datasets_cfg.get(key).text_processor.train
    text_processor = registry.get_processor_class(txt_processor_cfg.name).from_config(txt_processor_cfg)
    print('Initialization Finished')
    return model, vis_processor, text_processor

parser = eval_parser()
args = parser.parse_args()
cfg = Config(args)
setup_seeds(cfg)
print(cfg._convert_node_to_json(cfg.config))
setup_logger()
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

print("----------------------------------", device, "----------------------------------")
print("Datasets: ", args.dataset)                   


model, vis_processor, text_processor = init_model(cfg, device)
model.eval()

run_cfg = cfg.run_cfg
save_path = cfg.run_cfg.save_path
num_beams = run_cfg.get("num_beams", 3)
max_len = run_cfg.get("max_len", 20)
min_len = run_cfg.get("min_len", 1)
inference_method = run_cfg.get("inference_method", "rank")
num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
prompt = run_cfg.get("prompt", "")
if not os.path.exists(save_path):
    os.mkdir(save_path)

if 'vizwiz' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vizwiz"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["vizwiz"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vizwiz"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vizwiz"]["max_new_tokens"]

    vizwiz = json.load(open(eval_file_path, 'r'))

    data = VizWizEvalData(vizwiz, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    predicts = []
    total_acc = []
    for samples in tqdm(eval_dataloader):
        # samples['image'] = samples['image'].half().to(device)
        samples['image'] = samples['image'].to(device)
        texts = samples['q_input']
        gt_answers = samples['gt_ans']
        image_ids = samples['image_id']

        answers = model.predict_answers(
            samples=samples,
            inference_method=inference_method,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            num_ans_candidates=num_ans_candidates,
            prompt=prompt,
        )

        for i in range(len(answers)):
            question, answer, gt_answer, img_id = texts[i], answers[i], gt_answers[i], image_ids[i]
            result = {'img_id':img_id, 'question':question}
            result['answer'] = answer.replace('<unk>','').strip()
            count=0
            gt_answer = gt_answer.split('_')
            for gt in gt_answer:
                if gt.lower() == answer.lower():
                    count += 1
            acc = min(count/3.0, 1.0)
            total_acc.append(acc)
            result['gt_ans'] = gt_answer
            predicts.append(result)

    vizwiz_acc = np.average(total_acc)* 100.0
    print('vizwiz Acc: ', vizwiz_acc, flush=True)

    file_save_path = os.path.join(save_path, "vizwiz.json")
    with open(file_save_path,'a+') as f:
        json.dump(predicts, f)

    with open(os.path.join(save_path, f"evaluate_vizwiz.txt"), "a") as f:
        f.write(json.dumps({'agg_metrics': vizwiz_acc}) + "\n")

if 'hm' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["hm"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["hm"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["hm"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["hm"]["max_new_tokens"]

    annotation = []
    with open(eval_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            annotation.append(json_obj)

    data = HMEvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=20, shuffle=False)
    count=0
    total=0

    predict = []

    for samples in tqdm(eval_dataloader):
        samples['image'] = samples['image'].half().to(device)
        texts = samples['q_input']
        labels = samples['gt_ans']

        answers = model.predict_answers(
            samples=samples,
            inference_method=inference_method,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            num_ans_candidates=num_ans_candidates,
            prompt=prompt,
        )

        for answer, label in zip(answers, labels):
            result = dict()
            if answer.lower().strip() =="yes":
                answer=1
            elif answer.lower().strip()=="no":
                answer=0
            else:
                print("non-matching answer",answer)

            result['pred'] = answer
            result['gt'] = int(label)
            predict.append(result)
            if answer == label:
                count+=1
            total+=1
        print(answers)

    hm_val = count / total * 100
    print('hm val:', hm_val, flush=True)
    file_save_path = os.path.join(save_path, "hm.json")
    with open(file_save_path,'w') as f:
        json.dump(predict, f)

    with open(os.path.join(save_path, f"evaluate_hm.txt"), "a") as f:
        f.write(json.dumps({'agg_metrics': hm_val}) + "\n")


if 'vsr' in args.dataset:
            
    img_path = cfg.evaluation_datasets_cfg["vsr"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vsr"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vsr"]["max_new_tokens"]
    from datasets import load_dataset

    annotation = load_dataset("/mnt/pfs-guan-ssai/nlu/wanghanzi/data/visual-spatial-reasoning/vsr_zeroshot", split='test')
    data = VSREvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0

    predict = []

    # import pdb;pdb.set_trace()
    for samples in tqdm(eval_dataloader):

        samples['image'] = samples['image'].half().to(device)
        texts = samples['q_input']
        labels = samples['gt_ans']
        image_ids = samples['image_id']

        answers = model.predict_answers(
            samples=samples,
            inference_method=inference_method,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            num_ans_candidates=num_ans_candidates,
            prompt=prompt,
        )

        # answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            result['pred'] = answer.replace('<unk>','').strip()
            result['gt'] = label
            predict.append(result)
            if answer.lower() ==  label.lower():
                count+=1
            total+=1
    print('vsr test:', count / total * 100, flush=True)

    file_save_path = os.path.join(save_path, "vsr.json")
    with open(file_save_path,'w') as f:
        json.dump(predict, f)

    vsr_result =  count / total * 100
    with open(os.path.join(save_path, f"evaluate_vsr.txt"), "a") as f:
        f.write(json.dumps({'agg_metrics': vsr_result}) + "\n")


if 'nocap' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["nocap"]["eval_file_path"]
    val_cap = eval_file_path.replace("nocaps_val.json", "nocaps_val_4500_captions.json")
    img_path = cfg.evaluation_datasets_cfg["nocap"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["nocap"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["nocap"]["max_new_tokens"]

    annotation = []
    with open(eval_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            annotation.append(json_obj)


    data = NoCapsEvalDataset(annotation[0], vis_processor, text_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    predict = []

    for samples in tqdm(eval_dataloader):
        samples['image'] = samples['image'].half().to(device)
        texts = samples['q_input']
        image_ids = samples['image_id']

        answers = model.predict_answers(
            samples=samples,
            inference_method=inference_method,
            num_beams=num_beams,
            max_len=30,
            min_len=min_len,
            num_ans_candidates=num_ans_candidates,
            prompt=prompt,
        )

        for answer, image_id in zip(answers, image_ids):
            result = dict()

            result['answer'] = answer
            result['img_id'] = str(image_id.item())
            predict.append(result)

    # save predict to json file
    file_save_path = os.path.join(save_path, "nocap.json")
    with open(file_save_path,'w') as f:
        json.dump(predict, f)

    eval = nocap_eval(file_save_path)

    agg_metrics = eval.eval["CIDEr"] + eval.eval["Bleu_4"]

    result = {k: v for k, v in eval.eval.items()}
    result["agg_metrics"] = agg_metrics

    with open(os.path.join(save_path, f"evaluate_nocap.txt"), "a") as f:
        f.write(json.dumps(result) + "\n")
