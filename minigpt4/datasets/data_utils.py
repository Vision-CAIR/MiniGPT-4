"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import gzip
import logging
import os
import random as rnd
import tarfile
import zipfile
import random
from typing import List
from tqdm import tqdm

import decord
from decord import VideoReader
import webdataset as wds
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.base_dataset import ConcatDataset


decord.bridge.set_bridge("torch")
MAX_INT = registry.get("MAX_INT")


class ChainDataset(wds.DataPipeline):
    r"""Dataset for chaining multiple :class:`DataPipeline` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: List[wds.DataPipeline]) -> None:
        super().__init__()
        self.datasets = datasets
        self.prob = []
        self.names = []
        for dataset in self.datasets:
            if hasattr(dataset, 'name'):
                self.names.append(dataset.name)
            else:
                self.names.append('Unknown')
            if hasattr(dataset, 'sample_ratio'):
                self.prob.append(dataset.sample_ratio)
            else:
                self.prob.append(1)
                logging.info("One of the datapipeline doesn't define ratio and set to 1 automatically.")

    def __iter__(self):
        datastreams = [iter(dataset) for dataset in self.datasets]
        while True:
            select_datastream = random.choices(datastreams, weights=self.prob, k=1)[0]
            yield next(select_datastream)


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


def reorg_datasets_by_split(datasets, batch_sizes):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()
    reorg_batch_sizes = dict()

    # reorganize by split
    for dataset_name, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
                reorg_batch_sizes[split_name] = [batch_sizes[dataset_name]]
            else:
                reorg_datasets[split_name].append(dataset_split)
                reorg_batch_sizes[split_name].append(batch_sizes[dataset_name])

    return reorg_datasets, reorg_batch_sizes


def concat_datasets(datasets):
    """
    Concatenates multiple datasets into a single dataset.

    It supports may-style datasets and DataPipeline from WebDataset. Currently, does not support
    generic IterableDataset because it requires creating separate samplers.

    Now only supports conctenating training datasets and assuming validation and testing
    have only a single dataset. This is because metrics should not be computed on the concatenated
    datasets.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by split.

    Returns:
        Dict of concatenated datasets by split, "train" is the concatenation of multiple datasets,
        "val" and "test" remain the same.

        If the input training datasets contain both map-style and DataPipeline datasets, returns
        a tuple, where the first element is a concatenated map-style dataset and the second
        element is a chained DataPipeline dataset.

    """
    # concatenate datasets in the same split
    for split_name in datasets:
        if split_name != "train":
            assert (
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # if len(iterable_datasets) > 0:
            # concatenate map-style datasets and iterable-style datasets separately
            if len(iterable_datasets) > 1:
                chained_datasets = (
                    ChainDataset(iterable_datasets)
                )
            elif len(iterable_datasets) == 1:
                chained_datasets = iterable_datasets[0]
            else:
                chained_datasets = None

            concat_datasets = (
                ConcatDataset(map_datasets) if len(map_datasets) > 0 else None
            )

            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets

