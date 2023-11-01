"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from PIL import Image
import os

from minigpt4.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class OKVQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        question = data['question']
        question_id = data['question_id']
        img_file = '{:0>12}.jpg'.format(img_id)
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, question_id, img_id

class VizWizEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image']
        question = data['question']
        answers = data['answers']
        answers = '_'.join([answer['answer'] for answer in answers])
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] The question is '{question}' Based on the image, answer the question with a single word or phrase. and reply 'unanswerable' when the provided information is insufficient"
        return image, question, answers

class IconQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_id = data['image_id']
        question = data['question']
        image_path = os.path.join(self.root_path, image_id, 'image.png')
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image).half().cuda()
        candidates = '_'.join(data['choices'])
        answer = data['answer']
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, candidates, answer

class GQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["image"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["question"]
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        labels = ann["answer"]

        return image, question, labels

class HMEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["img"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["text"]
        question = f"This is an image writting '{question}'. Is this image hateful? Answer yes or no. Answer:"
        labels = ann["label"]

        return image, question, labels

class VSREvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_path = os.path.join(self.root_path, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["caption"]
        question = f'[vqa] Based on the image, is this statement true or false? {question}'
        labels = 'true' if ann["label"] == 1 else 'false'

        return image, question, labels