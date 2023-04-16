"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from omegaconf import OmegaConf


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)
