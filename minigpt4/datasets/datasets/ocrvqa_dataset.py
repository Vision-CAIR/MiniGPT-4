import os
import json
import pickle
import random
import time
import itertools

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


class OCRVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.data = self.create_data(ann_path)

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

    def create_data(self, ann_path):
        processed_data = []
        with open(ann_path, 'r') as f:
            data = json.load(f)
        for k in data.keys():
            if data[k]['split'] != 1: continue  # 1 for training, 2 for validation, 3 for test
            ext = os.path.splitext(data[k]['imageURL'])[1]
            imageFile = k + ext
            assert len(data[k]['questions']) == len(data[k]['answers'])
            for q, a in zip(data[k]['questions'], data[k]['answers']):
                processed_data.append(
                    {'question': q,
                     'answer': a,
                     'image_path': imageFile,
                     'image_id': k,
                     'title': data[k]['title'],
                     'genre': data[k]['genre'],
                     }
                )
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open(os.path.join(self.vis_root, sample['image_path'])).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(sample["question"])
        answer = self.text_processor(sample["answer"])

        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": sample['image_id']
        }
    
