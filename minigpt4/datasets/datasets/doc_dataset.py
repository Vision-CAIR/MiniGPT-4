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


class SingleSlideVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.data = self.create_data(ann_path)

        # self.instruction_pool = [
        #     "###Human: <Img><ImageHere></Img> {}###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> From this slide, {}###Assistant: ",
        # ]
        self.instruction_pool = [
            "<Img><ImageHere></Img> {}",
            "<Img><ImageHere></Img> From this slide, {}",
        ]
    def create_data(self, ann_path):
        with open(ann_path, 'r') as f:
            samples = f.readlines()
        data = []
        for sample in samples:
            sample = json.loads(sample)
            if len(sample['evidence_pages']) != 1: continue  # skip questions that need more than one slide page
            page = sample['evidence_pages'][0]
            image_name = 'slide_{}_1024.jpg'.format(page)
            # assert [int(image_name.split('-')[-2]) for image_name in image_names] == list(range(1, 21))  # check the format
            image_path = os.path.join(sample['deck_name'], image_name)
            data.append({
                'qa_id': sample['qa_id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'image_path': image_path
            })
        
        print("single slide ",len(data))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open(os.path.join(self.vis_root, sample['image_path'])).convert("RGB")
        image = self.vis_processor(image)

        # instruction = self.text_processor(sample["question"])
        instruction = random.choice(self.instruction_pool).format(self.text_processor(sample["question"]))

        # instruction = random.choice(self.instruction_pool).format(self.text_processor(sample["question"]))
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": sample['answer'],
            "qa_id": sample['qa_id'],
        }


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
        print("ocr vqa", len(processed_data))
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
    




class TextOCRDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.data = self.create_data(ann_path)

        self.instruction_pool = [
            "<Img><ImageHere></Img> [OCR] {}"
        ]

    def create_data(self, ann_path):
        processed_data = []
        with open(ann_path, 'r') as f:
            data = json.load(f)
        for k in data["anns"].keys():
            # ext = os.path.splitext(data[k]['imageURL'])[1]
            imageFile = data["anns"][k]["image_id"]+".jpg"
            bbox = data["anns"][k]["bbox"]
            text = data["anns"][k]["utf8_string"]
            # assert len(data[k]['questions']) == len(data[k]['answers'])
            # for q, a in zip(data[k]['questions'], data[k]['answers']):

            processed_data.append(
                {'bbox': bbox,
                    'answer': text,
                    'image_path': imageFile,
                    'image_id': k,
                    }
            )

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open(os.path.join(self.vis_root, sample['image_path'])).convert("RGB")
        width, height = image.size
        image = self.vis_processor(image)

        new_bbox =""
        image_size = 100
        bbox = sample['bbox']
        for index in range(len(bbox)):
            
            x1 = int(bbox[0]/width*image_size)
            y1 = int(bbox[1]/height*image_size)
            x2 = x1 + int(bbox[2]/width*image_size)
            y2 = y1 + int(bbox[3]/height*image_size)
            assert x1>=0 and x1<=image_size
            assert x2>=0 and x2<=image_size
            assert y1>=0 and y1<=image_size
            assert y2>=0 and y2<=image_size
            
            new_bbox = " <"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">"
         
        instruction = random.choice(self.instruction_pool).format(new_bbox)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": sample['answer'],
            "image_id": sample['image_id']
        }



class PlotVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.data = self.create_data(ann_path)

        self.instruction_pool = [
            '{}',
            'Question: {}',
            '{} A short answer to the question is',
            'Q: {} A:',
            'Question: {} Short answer:',
            # 'Given the image, answer the following question with no more than three words. {}',
            'Based on the image, respond to this question with a short answer: {}.',
            'Use the provided image to answer the question: {} Provide your answer as short as possible.',
            'What is the answer to the following question? "{}"',
            'The question "{}" can be answered using the image. A short answer is'
        ]

    def create_data(self, ann_path):
        processed_data = []
        with open(ann_path, 'r') as f:
            data = json.load(f)
        for da in data["qa_pairs"]:
            # ext = os.path.splitext(data[k]['imageURL'])[1]

            imageFile = str(da["image_index"])+".png"
            question = da["question_string"]
            answer = str(da["answer"])
            # assert len(data[k]['questions']) == len(data[k]['answers'])
            # for q, a in zip(data[k]['questions'], data[k]['answers']):

            processed_data.append(
                {'question': question,
                    'answer': answer,
                    'image_path': imageFile,
                    'image_id': str(da["image_index"]),
                    }
            )

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open(os.path.join(self.vis_root, sample['image_path'])).convert("RGB")
        # width, height = image.size
        image = self.vis_processor(image)


        # image_shape = image.shape
        instruction = "<Img><ImageHere></Img> {} ".format(sample["question"])
         
        instruction = random.choice(self.instruction_pool).format(instruction)
        
        answer = sample["answer"]


        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": sample['image_id']
        }

