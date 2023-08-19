import os
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset
import torch

class ScienceQADataset(CaptionDataset):
    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        if "image_id" in ann:
            # if "id" in ann:
            #     img_file = ann["image_id"]
            #     input_prompt = ann["input"]
            # else:
            #     img_file = '{}.jpg'.format(ann["image_id"])
            img_file = ann["image_id"]
            input_prompt = ann["input"]
            
            image_path = os.path.join(self.vis_root, img_file)
            image = Image.open(image_path).convert("RGB")
            
            image = self.vis_processor(image)
            # print(image.shape)
            caption = ann["caption"]
            
            return {
                "image": image,
                "input_prompt": input_prompt,
                "text_input": caption,
                "image_id": self.img_ids[ann["image_id"].split("/")[0]]
            }
        else:
            input_prompt = ann["input"]
            caption = ann["caption"]
            return {
                "image": torch.zeros(3, 224, 224),
                "input_prompt": input_prompt,
                "text_input": caption,
                "image_id": -100,
            }