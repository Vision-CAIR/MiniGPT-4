"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from minigpt4.common.dist_utils import download_cached_file, is_dist_avail_and_initialized
from minigpt4.common.utils import get_abs_path, is_url
from omegaconf import OmegaConf


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot


class BaseEncoder(nn.Module):
    """
    Base class for primitive encoders, such as ViT, TimeSformer, etc.
    """

    def __init__(self):
        super().__init__()

    def forward_features(self, samples, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return list(self.parameters())[0].device


class SharedQueueMixin:
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs=None):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T

        if idxs is not None:
            idxs = concat_all_gather(idxs)
            self.idx_queue[:, ptr : ptr + batch_size] = idxs.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr


class MomentumDistilationMixin:
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))
