"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import json
import os
import random
import torch

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "direct_answers": "; ".join(ann["direct_answers"]),
                "choices": "; ".join(ann["choices"]),
                "correct_choice": ann["choices"][ann["correct_choice_idx"]],
                "image": sample["image"],
            }
        )


class AOKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[   
            '{} Choose from {}.',
            'Q: {} Multi Choices: {} A: ',
            'Question: {} Multi Choices: {} Answer: ',
            "{} Choose one from the following possible answers: {}. ",
            '{} Choose from {}. The answer is',
        ]

        exist_annotation = []
        for ann in self.annotation:
            # image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
            image_path = os.path.join(self.vis_root, ann["image"])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation
        self.source = 'aokvqa'

    def get_data(self, index):
        ann = self.annotation[index]

        # image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_lst = ann["choices"]
        direct_answers = ann["direct_answers"]
        final_answer = random.choices(direct_answers, k=1)[0]
        for answer in answer_lst:
            if answer in direct_answers:
                final_answer = answer

        return {
            "image": image,
            "image_id": ann["image"],
            "question": question,
            "answer": final_answer,
            "choices": ", ".join(answer_lst)
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        question = self.text_processor(data["question"])

        answer = self.text_processor(data['answer'])
        q_input = question
        llm_input = random.choice(self.instruction_pool).format(question, data["choices"])

        return {
            "image": data['image'],
            "image_id": data["image_id"],
            "q_input": q_input,
            "llm_input": llm_input,
            "text_input": question,
            "text_output": answer,
            "answer": answer,
            "source": 'aokvqa',
        }


class AOKVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        self.instruction_pool =[   
            '{} Choose from {}.',
            'Q: {} Multi Choices: {} A: ',
            'Question: {} Multi Choices: {} Answer: ',
            "{} Choose one from the following possible answers: {}. ",
            '{} Choose from {}. The answer is',
        ]
        
        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.source = 'aokvqa'

    def collater(self, samples):
        (
            image_list,
            question_list,
            question_id_list,
            choices_list,
            correct_choice_idx_list,
            direct_answers_list,
            llm_input_list,
            q_input_list,
            source_list,
        ) = ([], [], [], [], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            choices_list.append(sample["choices"])
            correct_choice_idx_list.append(sample["correct_choice_idx"])
            direct_answers_list.append(sample["direct_answers"])
            llm_input_list.append(sample["llm_input"])
            q_input_list.append(sample["q_input"])
            source_list.append(sample["source"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "question_id": question_id_list,
            "choices": choices_list,
            "correct_choice_idx": correct_choice_idx_list,
            "direct_answers": direct_answers_list,
            "llm_input": llm_input_list,
            "q_input": q_input_list,
            "source": source_list,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        choices = ann["choices"]
        if "correct_choice_idx" in ann:
            correct_choice_idx = ann["correct_choice_idx"]
        else:
            correct_choice_idx = None

        if "direct_answers" in ann:
            direct_answers = ann["direct_answers"]
        else:
            direct_answers = None

        llm_input = random.choice(self.instruction_pool).format(question, ", ".join(choices))

        return {
            "image": image,
            "q_input": question,
            "llm_input": llm_input,
            "text_input": question,
            "question_id": ann["question_id"],
            "choices": choices,
            "correct_choice_idx": correct_choice_idx,
            "direct_answers": direct_answers,
            "source": 'aokvqa',
        }
    