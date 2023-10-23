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





class TextCapDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

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
        
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)


    def __len__(self):
        return len(self.ann["data"])


    def __getitem__(self, index):
        info = self.ann["data"][index]

        image_file = '{}.jpg'.format(info['image_id'])

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = info["caption_str"]
        caption = self.text_processor(caption)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(random.choice(self.instruction_pool))
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
            "data_type": "bbox",
            "question_split": True
        }



class TextCapBboxToObjectDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # self.instruction_pool = [
        #     "<Img><ImageHere></Img> What text does it show in  {} ",
        #     "<Img><ImageHere></Img> Extract the text from {} ",
        #     "<Img><ImageHere></Img> What is the textual content in {} ",
        #     "<Img><ImageHere></Img> Extract the textual information present in the {} ",
        #     "<Img><ImageHere></Img> What is the text written within this defined region {}",
        #     "<Img><ImageHere></Img> Transcribe the text located inside {}",
        #     "<Img><ImageHere></Img> Can you read and extract the text from this specific area {}",
        #     ]
        
        self.instruction_pool = [
            "<Img><ImageHere></Img> [OCR] {}"
        ]
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
                
        self.new_ann = {"data":[]}
        for da in self.ann["data"]:
            if da["ocr_info"] !=[]:
                ocr_info_filter = []
                for d in da["ocr_info"]:
                    if (d["bounding_box"]["width"]+d["bounding_box"]["top_left_x"])<=1.0 and (d["bounding_box"]["height"]+d["bounding_box"]["top_left_y"]) <=1.0 \
                        and d["bounding_box"]["top_left_x"]>=0 and d["bounding_box"]["top_left_y"]>=0:
                        ocr_info_filter.append(d)
                    if ocr_info_filter !=[]:
                        da["ocr_info"]=ocr_info_filter
                        self.new_ann["data"].append(da) 
        self.ann = self.new_ann


    def __len__(self):
        return len(self.ann["data"])


    def __getitem__(self, index):
        
        info = self.ann["data"][index]


        image_file = '{}.jpg'.format(info['image_id'])

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        # image_width,image_length = image.size
        image = self.vis_processor(image)



        image_size = 100
        
        ocr_info = info["ocr_info"]

        sampled_ocr = random.sample(ocr_info,1)[0]

        # print("sampled ocr", sampled_ocr)

        word_text = sampled_ocr["word"]
        width = sampled_ocr["bounding_box"]["width"]
        height = sampled_ocr["bounding_box"]["height"]
        top_left_x = sampled_ocr["bounding_box"]["top_left_x"]
        top_left_y = sampled_ocr["bounding_box"]["top_left_y"]
        
        x1 = int(top_left_x*image_size)
        y1 = int(top_left_y*image_size)
        x2 = x1 + int(width*image_size)
        y2 = y1 + int(height*image_size)
        assert x1>=0 and x1<=image_size
        assert x2>=0 and x2<=image_size
        assert y1>=0 and y1<=image_size
        assert y2>=0 and y2<=image_size

        
        word_bbox = "{<"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">}"

        instruction = random.choice(self.instruction_pool).format(word_bbox)
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": word_text,
            "data_type": "bbox",
            "question_split": True
        }