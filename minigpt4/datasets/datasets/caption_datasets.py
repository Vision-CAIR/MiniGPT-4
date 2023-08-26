"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


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


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            if "image_id" in ann:
                img_id = ann["image_id"]
                if "/" in img_id:
                    image_id = img_id.split("/")[0]
                    if image_id not in self.img_ids.keys():
                        self.img_ids[image_id] = n
                        n += 1
                else:
                    if img_id not in self.img_ids.keys():
                        self.img_ids[img_id] = n
                        n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        if "image_id" in ann:
            if "id" in ann:
                img_file = ann["image_id"]
                input_prompt = self.text_processor(ann["input"])
                image_path = os.path.join(self.vis_root, img_file)
                # print(image_path)
                image = Image.open(image_path).convert("RGB")
                image = self.vis_processor(image)
                # print(image.shape)
                caption = self.text_processor(ann["caption"])
                return {
                    "image": image,
                    "input_prompt": input_prompt,
                    "text_input": caption,
                    "image_id": self.img_ids[ann["image_id"].split("/")[0]],
                }
            else:
                img_file = '{:0>12}.jpg'.format(ann["image_id"])
                image_path = os.path.join(self.vis_root, img_file)
                image = Image.open(image_path).convert("RGB")
                image = self.vis_processor(image)
                caption = self.text_processor(ann["caption"])
                return {
                    "image": image,
                    "text_input": caption,
                    "image_id": self.img_ids[ann["image_id"]],
                }
        else:
            input_prompt = self.text_processor(ann["input"])
            caption = self.text_processor(ann["caption"])
            return {
                "image": torch.zeros(3, 224, 224),
                "input_prompt": input_prompt,
                "text_input": caption,
                "image_id": -100,
            }


class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
