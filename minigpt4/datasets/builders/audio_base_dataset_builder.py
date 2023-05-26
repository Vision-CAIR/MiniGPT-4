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
from minigpt4.datasets.builders import load_dataset_config
from minigpt4.processors.base_processor import BaseProcessor


class AudioBaseDatasetBuilder:
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

        self.audio_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
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
        aud_proc_cfg = self.config.get("audio_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if aud_proc_cfg is not None:
            aud_train_cfg = aud_proc_cfg.get("train")
            aud_eval_cfg = aud_proc_cfg.get("eval")

            self.audio_processors["train"] = self._build_proc_from_cfg(aud_train_cfg)
            self.audio_processors["eval"] = self._build_proc_from_cfg(aud_eval_cfg)

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
        self._download_aud()

    def _download_ann(self):
        """
        Download annotation files if necessary.
        All the audio-language datasets should have annotations of unified format.

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
