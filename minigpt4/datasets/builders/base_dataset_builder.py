"""
 This file is from
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import shutil
import warnings

from omegaconf import OmegaConf
import torch.distributed as dist
from torchvision.datasets.utils import download_url

import minigpt4.common.utils as utils
from minigpt4.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from minigpt4.common.registry import registry
from minigpt4.processors.base_processor import BaseProcessor



class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        self._download_ann()
        self._download_vis()

    def _download_ann(self):
        """
        Download annotation files if necessary.
        All the vision-language datasets should have annotations of unified format.

        storage_path can be:
          (1) relative/absolute: will be prefixed with env.cache_root to make full path if relative.
          (2) basename/dirname: will be suffixed with base name of URL if dirname is provided.

        Local annotation paths should be relative.
        """
        anns = self.config.build_info.annotations

        splits = anns.keys()

        cache_root = registry.get_path("cache_root")

        for split in splits:
            info = anns[split]

            urls, storage_paths = info.get("url", None), info.storage

            if isinstance(urls, str):
                urls = [urls]
            if isinstance(storage_paths, str):
                storage_paths = [storage_paths]

            assert len(urls) == len(storage_paths)

            for url_or_filename, storage_path in zip(urls, storage_paths):
                # if storage_path is relative, make it full by prefixing with cache_root.
                if not os.path.isabs(storage_path):
                    storage_path = os.path.join(cache_root, storage_path)

                dirname = os.path.dirname(storage_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                if os.path.isfile(url_or_filename):
                    src, dst = url_or_filename, storage_path
                    if not os.path.exists(dst):
                        shutil.copyfile(src=src, dst=dst)
                    else:
                        logging.info("Using existing file {}.".format(dst))
                else:
                    if os.path.isdir(storage_path):
                        # if only dirname is provided, suffix with basename of URL.
                        raise ValueError(
                            "Expecting storage_path to be a file path, got directory {}".format(
                                storage_path
                            )
                        )
                    else:
                        filename = os.path.basename(storage_path)

                    download_url(url=url_or_filename, root=dirname, filename=filename)

    def _download_vis(self):

        storage_path = self.config.build_info.get(self.data_type).storage
        storage_path = utils.get_cache_path(storage_path)

        if not os.path.exists(storage_path):
            warnings.warn(
                f"""
                The specified path {storage_path} for visual inputs does not exist.
                Please provide a correct path to the visual inputs or
                refer to datasets/download_scripts/README.md for downloading instructions.
                """
            )

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = os.path.join(vis_info.storage, split)

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
            )

        return datasets


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg
