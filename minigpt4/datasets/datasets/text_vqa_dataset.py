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

from collections import OrderedDict

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class TextVQADataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool =[   
            '{}',
            'Q: {} A: ',
            'Based on the image, respond to this question with a short answer: {}',
            '{} A short answer to the question is ',
            'Question: {} Short answer:',
        ]
        self.source = 'text_vqa'

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, "train/{}.jpg".format(ann["image_id"]))
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        q_input = question
        llm_input = random.choice(self.instruction_pool).format(question)
        
        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])
        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

        return {
            "image": image,
            'image_id': ann["image_id"],
            "text_input": question,
            "text_output": answer,
            "q_input": q_input,
            "llm_input": llm_input,
            "gt_answers": answer,
            "source": "text_vqa",
        }

class TextVQAEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[   
            '{}',
            'Q: {} A: ',
            'Based on the image, respond to this question with a short answer: {}',
            '{} A short answer to the question is ',
            'Question: {} Short answer:',
        ]
        self.source = 'text_vqa'

    def __getitem__(self, index):
        info = self.annotation[index]

        image_path = os.path.join(self.vis_root, "train/{}.jpg".format(info["image_id"]))

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(info["question"])

        q_input = question
        llm_input = random.choice(self.instruction_pool).format(question)
        
        answer_weight = {}
        for answer in info["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(info["answers"])
            else:
                answer_weight[answer] = 1 / len(info["answers"])
        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

        return {
            "image": image,
            "image_id": info["image_id"],
            "question": question,
            # "q_input": llm_input,
            "q_input": q_input,
            "llm_input": llm_input,
            "text_input": question,
            "gt_answers": answer,
            "source": 'text_vqa',
        }