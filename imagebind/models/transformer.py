#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py ;
# https://github.com/facebookresearch/deit/blob/main/models.py
# and https://github.com/facebookresearch/vissl/blob/main/vissl/models/trunks/vision_transformer.py


import copy
import fnmatch
import logging
from functools import partial
from typing import Callable, List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, trunc_normal_


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiheadAttention(nn.MultiheadAttention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


class ViTAttention(Attention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        assert attn_mask is None
        return super().forward(x)


class BlockWithMasking(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: str = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()

        assert not isinstance(
            attn_target, nn.Module
        ), "attn_target should be a Callable. Otherwise attn_target is shared across blocks!"
        self.attn = attn_target()
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)
        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.norm_2(x)))
        else:
            x = (
                x
                + self.drop_path(self.attn(self.norm_1(x), attn_mask))
                * self.layer_scale_gamma1
            )
            x = x + self.drop_path(self.mlp(self.norm_2(x))) * self.layer_scale_gamma2
        return x


_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        block: Callable = BlockWithMasking,
        pre_transformer_layer: Callable = None,
        post_transformer_layer: Callable = None,
        drop_path_rate: float = 0.0,
        drop_path_type: str = "progressive",
        norm_layer: Callable = _LAYER_NORM,
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.0,
        layer_scale_type: str = None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value: float = 1e-4,  # from cait; float
        weight_init_style: str = "jax",  # possible values jax or pytorch
    ):
        """
        Simple Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the input except that it is a Tensor
        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init_style == "jax":
                # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init_style == "pytorch":
                # PyTorch ViT uses trunc_normal_
                trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor = None,
        use_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        checkpoint_blk_ids: List[int] = None,
    ):
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)
        if use_checkpoint and checkpoint_blk_ids is None:
            checkpoint_blk_ids = [
                blk_id
                for blk_id in range(len(self.blocks))
                if blk_id % checkpoint_every_n == 0
            ]
        if checkpoint_blk_ids:
            checkpoint_blk_ids = set(checkpoint_blk_ids)
        for blk_id, blk in enumerate(self.blocks):
            if use_checkpoint and blk_id in checkpoint_blk_ids:
                tokens = checkpoint.checkpoint(
                    blk, tokens, attn_mask, use_reentrant=False
                )
            else:
                tokens = blk(tokens, attn_mask=attn_mask)
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)
        return tokens