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
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset



class TextCapDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = [
            'Briefly describe this image.',
            'Provide a concise depiction of this image.',
            'Present a short description of this image.',
            'Summarize this image in a few words.',
            'A short image caption:',
            'A short image description:',
            'A photo of ',
            'An image that shows ',
            'Write a short description for the image. ',
            'Write a description for the photo.',
            'Provide a description of what is presented in the photo.',
            'Briefly describe the content of the image.',
            'Can you briefly explain what you see in the image?',
            'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.',
            'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.',
        ]
        self.source = 'text_cap'

    def __getitem__(self, index):
        info = self.annotation[index]

        image_file = info['image_path']

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = info["caption_str"]
        caption = self.text_processor(caption)
        instruction = random.choice(self.instruction_pool)
        q_input = instruction
        llm_input = instruction

        return {
            "image": image,
            "image_id": info["image_name"],
            "answer": caption,
            "q_input": q_input,
            "llm_input": llm_input,
            "text_input": llm_input,
            "text_output": caption,
            "source": self.source,
        }

class TextCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = [
            'Briefly describe this image.',
            'Provide a concise depiction of this image.',
            'Present a short description of this image.',
            'Summarize this image in a few words.',
            'A short image caption:',
            'A short image description:',
            'A photo of ',
            'An image that shows ',
            'Write a short description for the image. ',
            'Write a description for the photo.',
            'Provide a description of what is presented in the photo.',
            'Briefly describe the content of the image.',
            'Can you briefly explain what you see in the image?',
            'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.',
            'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.',
        ]
        self.source = 'text_cap'

    def __getitem__(self, index):
        info = self.annotation[index]

        # image_file = '{}.jpg'.format(info['image_id'])
        image_file = info['image_path']

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = info["caption_str"]
        caption = self.text_processor(caption)
        instruction = random.choice(self.instruction_pool)
        q_input = instruction
        llm_input = instruction

        return {
            "image": image,
            "image_id": info["image_name"],
            "text_input":llm_input,
            "q_input": q_input,
            "llm_input": llm_input,
            "source": self.source,
        }