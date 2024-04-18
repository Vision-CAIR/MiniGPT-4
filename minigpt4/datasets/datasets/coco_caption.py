"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np
import random

from PIL import Image
from PIL import ImageFile
from collections import OrderedDict

ImageFile.LOAD_TRUNCATED_IMAGES = True

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )
    
class COCOCapDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0

        self.filter_anntation = []
        
        for ann in self.annotation:
            if "train" in ann["image"]:
                self.filter_anntation.append(ann)
        self.annotation = self.filter_anntation

        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

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
        self.source = 'coco_cap'
        
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        # img_file = ann["image"].split("/")[-1]
        img_file = ann["image"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        instruction = random.choice(self.instruction_pool)
        # q_input = ""
        q_input = instruction
        llm_input = instruction

        return {
            "image": image,
            "image_id": ann["image"],
            "answer": caption,
            "q_input": q_input,
            "llm_input": llm_input,
            "text_input": llm_input,
            "text_output": caption,
            "source": 'coco_cap',
        }


class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
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
        self.source = 'coco_cap'

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        try:
            image = self.vis_processor(image)
        except Exception as e:
            print(e)
            print(image_path)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]
        instruction = random.choice(self.instruction_pool)
        # q_input = ""
        q_input = instruction
        llm_input = instruction

        return {
            "image": image,
            "image_id": img_id,
            "text_input":llm_input,
            "q_input": q_input,
            "llm_input": llm_input,
            "source": self.source,
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, annotations, vis_processor, text_processor, vis_root):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        self.annotation = annotations
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
        self.source = 'no_cap'
        self.vis_root = vis_root    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_name = ann["image"].split("/")[-1]
        image_path = os.path.join(self.vis_root,image_name)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]
        instruction = random.choice(self.instruction_pool)
        q_input = instruction
        llm_input = instruction

        return {
            "image": image,
            "image_id": img_id,
            "q_input": q_input,
            "llm_input": llm_input,
            "source": self.source,
        }


class EvalCaptionData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        ann = dict()
        for item in self.loaded_data:
            image_id = item['image_id']
            ann[image_id] = item['image']
        self.ann = [{'image_id':image_id, 'image': ann[image_id]} for image_id in ann]

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, idx):
        data = self.ann[idx]
        image_id = data['image_id']
        img_file = data['image'].split('/')[-1]
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
            
        image = self.vis_processor(image)
        question = f"[caption] please describe this image?"
        return image, question, image_id
