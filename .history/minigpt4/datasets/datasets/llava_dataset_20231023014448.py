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
        
        instruction = '<Img><ImageHere></Img> {} '.format(self.text_processor(instruction))

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['id'],
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

        instruction = '<Img><ImageHere></Img> {} '.format(self.text_processor(instruction))

        # instruction = '<Img><ImageHere></Img> {} '.format(self.text_processor(instruction))
        # answer = self.text_processor(answer)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['id'],
        }




class MiniGPT4v(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor


        self.instruction_pool = [
            'please describe this image as detailed as possible',
            'What do you see happening in this image?',
            "Can you elaborate on the elements of the picture provided?",
            "Describe the following image.",
            "Write a detailed description of the given image.",
            "Write a detailed description of the given image.",
            "Explain the visual content of the image in great detail"
        ]
        self.ann=[]

        with open(ann_path,"r") as f:
            for line in f.readlines():
                self.ann.append(json.loads(line))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        # image_file = 'COCO_train2014_{}.jpg'.format(info['image_path'])
        # print("info keys",info.keys())
        if "image_path" in info.keys():
            image_path = "/ibex/reference/CV/COCO/cocoapi/data/2017/images/jpeg/train/"+info['image_path']
            
        else:
            # print("coming here?")
            image_file = "images/"+info["image"]
            image_path = os.path.join(self.vis_root, image_file)
            # print(image_path)


        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        if "question" in info.keys():
            question = info['question']
        else:
            question = random.sample(self.instruction_pool,1)[0]


        answer = info["caption"]


        instruction = '<Img><ImageHere></Img> {} '.format(self.text_processor(question))

        # instruction = '<Img><ImageHere></Img> {} '.format(self.text_processor(instruction))
        # answer = self.text_processor(answer)
        # print("image path", image_path)
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            # "image_id": info['id'],
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

        self.ann=[]

    
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

        self.connect_sym = "!@#"

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = 'COCO_train2014_{}.jpg'.format(info['id'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        first_instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        first_instruction = '<Img><ImageHere></Img> {} '.format(first_instruction)

        questions = [first_instruction]
        answers = []

        for i, item in enumerate(info["conversations"][1:]):
            if i % 2 ==0:  # assistant
                assistant_answer = item["value"]
                answers.append(assistant_answer)
            else:
                human_instruction = item["value"]+" "
                questions.append(human_instruction)

        questions = self.connect_sym.join(questions)
        # questions = questions.replace("\\\\","\\")
        answers = self.connect_sym.join(answers)


        return {
            "image": image,
            "conv_q": questions,
            'conv_a': answers,
            "image_id": info['id'],
            "connect_sym": self.connect_sym
        }