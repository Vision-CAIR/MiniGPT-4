"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from PIL import Image
import os
import random

from minigpt4.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    # def collater(self, samples):
    #     image_list, question_list, answer_list, weight_list = [], [], [], []
    
    #     num_answers = []
    
    #     for sample in samples:
    #         image_list.append(sample["image"])
    #         question_list.append(sample["question"])
    
    #         weight_list.extend(sample["weights"])
    
    #         answers = sample["answer"]
    
    #         answer_list.extend(answers)
    #         num_answers.append(len(answers))
    
    #     return {
    #         "image": torch.stack(image_list, dim=0),
    #         "text_input": question_list,
    #         "answer": answer_list,
    #         "weight": torch.Tensor(weight_list),
    #         "n_answers": torch.LongTensor(num_answers),
    #     }


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
        # question = f"[vqa] {question} "
        return image, question, question_id, img_id

class VizWizEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.instruction_pool =[   
            '{}',
            'Q: {} A: ',
            'Based on the image, respond to this question with a short answer: {}',
            '{} A short answer to the question is ',
            'Question: {} Short answer:',
        ]
        self.source = 'vizwiz'

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
        question = question
        llm_input = random.choice(self.instruction_pool).format(question)
        sample = {
            'image':image,
            'q_input': question,
            'llm_input': llm_input,
            'image_id': img_id,
            'gt_ans': answers,
            'source': self.source,
        }
        return sample

class AOKVQADAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_file = data['image']
        question = data['question']
        question_id = data['question_id']
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        # question = f"[vqa] {question} "
        return image, question, question_id

class AOKVQAMCEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_file = data['image']
        question = data['question']
        question_id = data['question_id']
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image).half().cuda()
        candidates=data['choices']
        question = f"Given this image, choose one correct answer from {candidates} for this question: {question} "
        # question = f"Based on the image, respond to this question with a short answer: {question}"
        # question = f"{question} "
        return image, question, question_id, candidates

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


class HMEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.instruction_pool =[   
            '{}',
            'Q: {} A: ',
            'Based on the image, respond to this question with a short answer: {}',
            '{} A short answer to the question is ',
            'Question: {} Short answer:',
        ]
        self.source = 'hm'

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["img"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["text"]
        question = f"This is an image writting '{question}'. Is this image hateful? Answer yes or no."
        llm_input = random.choice(self.instruction_pool).format(question)
        labels = ann["label"]
        
        sample = {
            'image':image,
            'q_input': question,
            'llm_input': llm_input,
            'image_id':  image_id,
            'gt_ans': labels,
            'source': self.source,

        }

        return sample

class VSREvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.instruction_pool =[   
            '{}',
            'Q: {} A: ',
            'Based on the image, respond to this question with a short answer: {}',
            '{} A short answer to the question is ',
            'Question: {} Short answer:',
        ]
        self.source = 'vsr'

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_path = os.path.join(self.root_path, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["caption"]
        question = f'Is the statement: {question}, true or false?'
        llm_input = random.choice(self.instruction_pool).format(question)
        labels = 'true' if ann["label"] == 1 else 'false'

        sample = {
            'image':image,
            'q_input': question,
            'llm_input': llm_input,
            'image_id':  ann["image"],
            'gt_ans': labels,
            'source': self.source,

        }

        return sample