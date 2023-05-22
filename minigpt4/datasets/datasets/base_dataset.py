"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset):
    def __init__(
        self, x_processor=None, text_processor=None, x_root=None, ann_paths=[]
    ):
        """
        x_root (string): Root directory of data in modality X (e.g. coco/images/, etc.)
        ann_root (string): directory to store the annotation file
        """
        self.x_root = x_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])

        self.x_processor = x_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, x_processor, text_processor):
        self.x_processor = x_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

