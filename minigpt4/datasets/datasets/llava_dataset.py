import os
import json
import pickle
import random
import time
# import iterto
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

class LlavaDetailDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.source = 'llava_detail'

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = 'COCO_train2014_{}.jpg'.format(info['id'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        answer = info['conversations'][1]['value']
        instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        
        question = self.text_processor(instruction)

        return {
            "image": image,
            "image_id": info['id'],
            "q_input": question,
            "llm_input": question,
            "text_input": question,
            "text_output": answer,
            "answer": answer,
            "source": 'llava_detail',
        }
    
class LlavaReasonDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
        self.source = 'llava_reason'

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = 'COCO_train2014_{}.jpg'.format(info['id'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        answer = info['conversations'][1]['value']
        instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()

        question = self.text_processor(instruction)

        return {
            "image": image,
            "image_id": info['id'],
            "q_input": question,
            "llm_input": question,
            "text_input": question,
            "text_output": answer,
            "answer": answer,
            "source": 'llava_reason',
        }
    
class LlavaConversationDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.source = 'llava_conver'

        self.ann=[]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        
        image_file =  info['image'].split('/')[-1]
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(info['text_input'])
        answer = self.text_processor(info['text_output'])

        return {
            "image": image,
            "image_id": info['id'],
            "q_input": question,
            "llm_input": question,
            "text_input": question,
            "text_output": answer,
            "answer": answer,
            "source": 'llava_conver',
        }

class LlavaPretrainDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.source = 'llava_pretrain'

        self.ann=[]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        
        image_path = os.path.join(self.vis_root, info['image'])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        answer = info['conversations'][1]['value']

        instruction = self.text_processor(instruction)
        answer = self.text_processor(answer)

        return {
            "image": image,
            "image_id": info['id'],
            "q_input": instruction,
            "llm_input": instruction,
            "text_input": instruction,
            "text_output": answer,
            "answer": answer,
            "source": 'llava_pretrain',
        }
    
    
class LlavaMixDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/) init with build_datasets() in data_builder
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.tmp_ann=[]
    
        with open(ann_path, 'r') as f:
            self.tmp_ann = json.load(f)

        self.ann = self.filter_process_data()
        self.source = 'llava_mix'

    def build_image_path(self, info):
        if 'image' not in info.keys(): # pure text data
            return None
        
        image_name = info['image']
        if 'coco' in image_name:
            image_file = 'COCO_train2014_{}.jpg'.format(info['id'])
            image_path = os.path.join(self.vis_root['coco'], image_file)
        elif 'gqa' in image_name:
            image_path = os.path.join(self.vis_root['gqa'], '{}.jpg'.format(info['id']))
        elif 'ocr' in image_name:
            image_path = os.path.join(self.vis_root['ocr'], '{}.jpg'.format(info['id']))
        elif 'textvqa' in image_name:
            image_path = os.path.join(self.vis_root['text'], '{}.jpg'.format(info['id']))
        elif 'vg' in image_name:
            # TODO
            # image_path = os.path.join(self.vis_root['vg'], '{}.jpg'.format(info['id']))
            image_path = None

        return image_path
    
    def process_convers(self, info):
        first_instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        # first_instruction = '<Img><ImageHere></Img> {} '.format(first_instruction)

        questions = [first_instruction]
        answers = []

        for i, item in enumerate(info["conversations"][1:]):
            if i % 2 ==0:  # assistant
                assistant_answer = item["value"]
                answers.append(assistant_answer)
            else:
                human_instruction = item["value"]+" "
                questions.append(human_instruction)

        return questions, answers


    def filter_process_data(self):
        ann = list()
        for info in self.tmp_ann:
            image_path = self.build_image_path(info)
            if image_path != None:
                questions, answers = self.process_convers(info)
                assert len(questions) == len(answers)
                for i in range(len(questions)):
                    ann.append({
                        'question_id': '{}_{}'.format(info['id'],str(i)),
                        'image_id': str(info['id']),
                        'image_path': image_path,
                        'text_input': questions[i],
                        'text_output': answers[i],
                    })
        return ann


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        info = self.ann[index]
        try:
            image = Image.open(info['image_path']).convert("RGB")
        except:
            image = Image.open(info['image_path'].replace('train','val')).convert("RGB")
        image = self.vis_processor(image)
        instruction = 'Answer the question using a single word or phrase.'

        return {
            "image": image,
            "image_id": info['image_id'],
            "text_input": info['text_input'],
            "text_output": info['text_output'],
            "q_input": info['text_input'].replace(instruction,''),
            "llm_input": info['text_input'],
            "question_id": info["question_id"],
        }