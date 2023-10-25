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


class GroundedDetailDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            '[grounding] please describe this image in details',
            '[grounding] describe this image as detailed as possible',
            '[grounding] summarize this image in details',
            '[grounding] give a thorough description of what you see in this image',
        ]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        # image_file = 'COCO_train2014_{}.jpg'.format(info['image_id'])
        image_file = '{}.jpg'.format(info['image_id'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        answer = info['grounded_caption']
        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['image_id'],
        }




class CaptionToObjectDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            '[detection] {}',
        ]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = '{}.jpg'.format(info['image_id'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        input = info["caption"]
        answer = info["output"]

        instruction = random.choice(self.instruction_pool).format(input)

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        print("CaptionToObject instruction", instruction)
        print("CaptionToObject answer", answer)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['image_id'],
        }




class PhraseToObjectDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            '[detection] {}',
        ]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        image_file = '{}.jpg'.format(info['image_id'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        input = info["phrase"]
        answer = "<p>"+input+"</p> "+info["bbox"]
        instruction = random.choice(self.instruction_pool).format(input)

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        print("PhraseToObject instruction", instruction)
        print("PhraseToObject answer", answer)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['image_id'],
        }
