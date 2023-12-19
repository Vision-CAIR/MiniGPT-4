"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
import numpy as np
from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

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


class OKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[   
            '{}',
            'Q: {} A: ',
            'Based on the image, respond to this question with a short answer: {}',
            '{} A short answer to the question is ',
            'Question: {} Short answer:',
        ]
        
        exist_annotation = []
        for ann in self.annotation:
            # image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
            image_path = os.path.join(self.vis_root, ann["image"])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation
        self.source = 'okvqa'


    def get_data(self, index):
        ann = self.annotation[index]

        # image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights


        return {
            "image": image,
            "image_id":  ann["image"],
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        question = data['question']
        answer = self.text_processor(data['answer'])

        q_input = question
        llm_input = random.choice(self.instruction_pool).format(question)

        return {
            "image": data['image'],
            "image_id": data["image_id"],
            "question_id": data["question_id"],
            # "instruction_input": instruction,
            "q_input": q_input,
            "llm_input": llm_input,
            "text_input": question,
            "text_output": answer,
            "answer": answer,
            "source": 'okvqa',
        }


class OKVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        
        self.instruction_pool =[   
            '{}',
            'Q: {} A: ',
            'Based on the image, respond to this question with a short answer: {}',
            '{} A short answer to the question is ',
            'Question: {} Short answer:',
        ]
        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))
        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.source = 'okvqa'
        self.annotation_add = self.get_data()
        self._add_instance_ids()

    def get_data(self):
        ann_instruct = list()
        for i in range(len(self.annotation)):
            ann = self.annotation[i].copy()
            j = i % len(self.instruction_pool)
            question = self.text_processor(ann["question"])
            llm_input = self.instruction_pool[j].format(question)
            ann['llm_input'] = llm_input
            ann_instruct.append(ann)
        np.random.seed(10)
        np.random.shuffle(ann_instruct)
        return ann_instruct
    
    def __getitem__(self, index):
        ann = self.annotation_add[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann["question"])
        q_input = question
        llm_input = ann.get("llm_input",random.choice(self.instruction_pool).format(question))

        return {
            "image": image,
            "image_id": ann["image"],
            'image_path': image_path,
            "question_id": ann["question_id"],
            # "instance_id": ann["instance_id"],
            "question": question,
            "q_input": q_input,
            "llm_input": llm_input,
            "text_input": question,
            # "answer": ann["answer"],
            "source": 'okvqa',
        }
