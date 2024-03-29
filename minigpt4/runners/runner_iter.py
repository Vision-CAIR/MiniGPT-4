"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
import webdataset as wds
from minigpt4.common.dist_utils import download_cached_file, is_main_process, main_process
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from minigpt4.runners.runner_base import RunnerBase
from torch.utils.data.dataset import ChainDataset


@registry.register_runner("runner_iter")
class RunnerIter(RunnerBase):
    """
    Run training based on the number of iterations. This is common when
    the training dataset size is large. Underhood logic is similar to
    epoch-based training by considering every #iters_per_inner_epoch as an
    inner epoch.

    In iter-based runner, after every #iters_per_inner_epoch steps, we

        1) do a validation epoch;
        2) schedule the learning rate;
        3) save the checkpoint.

    We refer every #iters_per_inner_epoch steps as an inner epoch.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)

        self.start_iters = 0

        self.max_iters = int(self.config.run_cfg.get("max_iters", -1))
        assert self.max_iters > 0, "max_iters must be greater than 0."

        self.iters_per_inner_epoch = int(
            self.config.run_cfg.get("iters_per_inner_epoch", -1)
        )
        assert (
            self.iters_per_inner_epoch > 0
        ), "iters_per_inner_epoch must be greater than 0."

    @property
    def max_epoch(self):
        return int(self.max_iters / self.iters_per_inner_epoch)

    @property
    def cur_epoch(self):
        try:
            return self.train_loader.epoch
        except AttributeError:
            # pipeline data (e.g. LAION) is streaming, have no concept of epoch
            return 0

    def _progress(self, cur_iters):
        return "{}_iters={}".format(self.cur_epoch, cur_iters)

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_iters = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for start_iters in range(
            self.start_iters, self.max_iters, self.iters_per_inner_epoch
        ):
            end_iters = start_iters + self.iters_per_inner_epoch

            # training phase
            if not self.evaluate_only:
                logging.info(
                    "Start training, max_iters={}, in total {} inner epochs.".format(
                        self.max_iters, int(self.max_iters / self.iters_per_inner_epoch)
                    )
                )
                if start_iters == self.start_iters:
                    self.task.before_training(
                        model=self.unwrap_dist_model(self.model),
                        dataset=self.datasets,
                    )
                train_stats = self.train_iters(self.cur_epoch, start_iters)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=self._progress(end_iters)
                    )
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_iters, best_agg_metric = end_iters, agg_metrics

                                self._save_checkpoint(end_iters, is_best=True)

                            val_log.update({"best_iters": best_iters})
                            self.log_stats(val_log, split_name)

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each inner epoch.
                if not self.evaluate_only:
                    self._save_checkpoint(end_iters, is_best=False)

            if self.evaluate_only:
                break
            dist.barrier()

        # testing phase
        self.evaluate(cur_epoch=self.cur_epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def train_iters(self, epoch, start_iters):
        # train by iterations
        self.model.train()

        return self.task.train_iters(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_inner_epoch=self.iters_per_inner_epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @main_process
    def _save_checkpoint(self, cur_iters, is_best=False):
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }

        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "iters": cur_iters,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_iters),
        )
        logging.info("Saving checkpoint at iters {} to {}.".format(cur_iters, save_to))
        torch.save(save_obj, save_to)

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_iters = checkpoint["iters"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            # reoganize datasets by split and concatenate/chain if necessary
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)

            if dataset_ratios is None:
                # concatenate map-style datasets and chain wds.DataPipe datasets separately
                # training set becomes a tuple (ConcatDataset, ChainDataset), both are
                # optional but at least one of them is required. The resultant ConcatDataset
                # and ChainDataset will be sampled evenly.
                logging.info(
                    "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
                )

                datasets = reorg_datasets_by_split(self.datasets)
                self.datasets = concat_datasets(datasets)
            else:
                # create multi-loader with the provided ratios, without concatenating or chaining
                missing_keys = [k for k in dataset_ratios if k not in self.datasets]
                if len(missing_keys) > 0:
                    raise ValueError(
                        "Datasets with the following split names are not found: {}".format(
                            missing_keys
                        )
                    )

                unexpected_keys = [k for k in self.datasets if k not in dataset_ratios]
                if len(unexpected_keys) > 0:
                    raise ValueError(
                        "Datasets with the following split names are not expected: {}".format(
                            unexpected_keys
                        )
                    )

                dataset_ratios = [float(dataset_ratios[k]) for k in self.datasets]
                self.datasets = reorg_datasets_by_split(self.datasets)
                # to keep the same structure as return value of concat_datasets
                self.datasets = {
                    k: v[0] if len(v) == 1 else v for k, v in datasets.items()
                }

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    try:
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    except TypeError:
                        # a single wds.DataPipeline or ChainDataset
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders
