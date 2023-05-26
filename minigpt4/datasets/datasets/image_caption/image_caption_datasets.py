"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from PIL import Image

from minigpt4.datasets.datasets.mixins.mixins import __ImageDisplMixin


class ImageCaptionDataset(BaseDataset, __ImageDisplMixin):
    def __init__(self, vision_processor, text_processor, x_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vision_processor, text_processor, x_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{:0>12}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.x_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.x_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "vision": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class CaptionEvalDataset(BaseDataset, __ImageDisplMixin):
    def __init__(self, vision_processor, text_processor, x_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vision_processor, text_processor, x_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.x_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.x_processor(image)

        return {
            "vision": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
