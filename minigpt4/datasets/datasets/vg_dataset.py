import os
import json
import pickle
import random
import time
import itertools

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from visual_genome import local




class ReferVisualGenomeDataset(Dataset):
    def __init__(self, vis_processor, text_processor, data_dir):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.data_dir = data_dir

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        all_regions = local.get_all_region_descriptions(self.data_dir)
        all_regions = [region for regions in all_regions for region in regions]

        # follow OFA practice, only regions smaller than 16384 pixels are used for refer
        self.regions = [region for region in all_regions if region.width * region.height < 16384]


        self.instruction_pool = [
            "[refer] {}",
            "[refer] give me the location of {}",
            "[refer] where is {} ?",
            "[refer] from this image, tell me the location of {}",
            "[refer] the location of {} is",
            "[refer] could you tell me the location for {} ?",
            "[refer] where can I locate the {} ?",
        ]


    def __len__(self):
        return len(self.regions)

    def preprocess(self, index):
        region = self.regions[index]
        image_file = region.image.url.split('/')[-2:]
        image_path = os.path.join(self.data_dir, *image_file)
        image = Image.open(image_path).convert("RGB")
        image_orig_size = image.size
        image = self.vis_processor(image)
        image_new_size = [100,100]

        sample_sentence = region.phrase
        refer_sentence = self.text_processor(sample_sentence)

        bbox = [region.x, region.y, region.width, region.height]

        bbox = [
            bbox[0] / image_orig_size[0] * image_new_size[0],
            bbox[1] / image_orig_size[1] * image_new_size[1],
            (bbox[0] + bbox[2]) / image_orig_size[0] * image_new_size[0],
            (bbox[1] + bbox[3]) / image_orig_size[1] * image_new_size[1]
        ]
        bbox = [int(x) for x in bbox]
        bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
        return {
            "image": image,
            "refer_sentence": refer_sentence,
            "bbox": bbox,
            "image_id": region.image.id,
        }

    def __getitem__(self, index):
        data = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(data['refer_sentence'])

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": data['bbox'],
            "image_id": data['image_id'],
        }


