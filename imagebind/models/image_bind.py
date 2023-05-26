#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial
from types import SimpleNamespace
from typing import Union, Optional, Tuple, Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from imagebind.models.helper import (
    EinOpsRearrange,
    LearnableLogitScaling,
    Normalize,
    SelectElement,
    SelectEOSAndProject,
)
from imagebind.models.multimodal_formers import SequenceGenericQFormer, disabled_train
from imagebind.models.multimodal_preprocessors import (
    AudioPreprocessor,
    IMUPreprocessor,
    PadIm2Video,
    PatchEmbedGeneric,
    RGBDTPreprocessor,
    SpatioTemporalPosEmbeddingHelper,
    TextPreprocessor,
    ThermalPreprocessor,
)
from imagebind.models.multimodal_projectors import create_projectors

from imagebind.models.transformer import MultiheadAttention, SimpleTransformer

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)


class ImageBindJoiner(nn.Module):
    def __init__(self,
                 vision_query_token_num: int,
                 audio_query_token_num: int,
                 vision_qformer_frozen: bool = False,
                 vision_qformer_model: str = "",  # The url or path of pre-trained vision Q-Former model
                 vision_pre_dims: List[int] = (),  # Projection before Q-Former
                 vision_post_dims: List[int] = (1280, 768),  # Projection after Q-Former
                 audio_pre_dims: List[int] = (),
                 audio_post_dims: List[int] = (768, 768)
                 ):
        super().__init__()
        assert not (vision_qformer_frozen and vision_qformer_model == "")
        self.modality_pre_projectors = self._create_modality_pre_projectors(vision_pre_dims, audio_pre_dims)
        self.modality_qformers = self._create_modality_qformers(vision_query_token_num,
                                                                vision_qformer_frozen,
                                                                vision_qformer_model,
                                                                audio_query_token_num)
        self.modality_post_projectors = self._create_modality_post_projectors(vision_post_dims, audio_post_dims)

    def _create_modality_pre_projectors(self,
                                        vision_pre_dims,
                                        audio_pre_dims
                                        ):
        modality_pre_projectors = {
            ModalityType.VISION: create_projectors(vision_pre_dims),
            ModalityType.AUDIO: create_projectors(audio_pre_dims)
        }
        return modality_pre_projectors

    def _create_modality_qformers(self,
                                  vision_query_token_num,
                                  vision_qformer_frozen,
                                  vision_qformer_model,
                                  audio_query_token_num
                                  ):
        vision_qformer = SequenceGenericQFormer(num_query_token=vision_query_token_num,
                                                freeze_qformer=vision_qformer_frozen,
                                                encoder_width=1280,  # TODO: fix hard-coding
                                                q_former_model=vision_qformer_model)
        audio_qformer = SequenceGenericQFormer(num_query_token=audio_query_token_num,
                                               freeze_qformer=False,
                                               encoder_width=768)
        modality_qformers = {
            ModalityType.VISION: vision_qformer,
            ModalityType.AUDIO: audio_qformer
        }

        return nn.ModuleDict(modality_qformers)

    def _create_modality_post_projectors(self, vision_post_dims, audio_post_dims):
        vision_projector = create_projectors(vision_post_dims)
        audio_projector = create_projectors(audio_post_dims)
        modality_projectors = {
            ModalityType.VISION: vision_projector,
            ModalityType.AUDIO: audio_projector
        }

        return nn.ModuleDict(modality_projectors)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = {}
        for modality_key, modality_value in inputs.items():
            if modality_value is not None:
                modality_value = self.modality_pre_projectors[modality_key](modality_value)
                modality_value = self.modality_qformers[modality_key](modality_value)
                modality_value = self.modality_post_projectors[modality_key](modality_value)
                outputs[modality_key] = modality_value
        return outputs


class ImageBindModel(nn.Module):
    def __init__(
            self,
            video_frames=2,
            kernel_size=(2, 14, 14),
            audio_kernel_size=16,
            audio_stride=10,
            out_embed_dim=768,
            vision_embed_dim=1024,
            vision_num_blocks=24,
            vision_num_heads=16,
            audio_embed_dim=768,
            audio_num_blocks=12,
            audio_num_heads=12,
            audio_num_mel_bins=128,
            audio_target_len=204,
            audio_drop_path=0.1,
            text_embed_dim=768,
            text_num_blocks=12,
            text_num_heads=12,
            depth_embed_dim=384,
            depth_kernel_size=16,
            depth_num_blocks=12,
            depth_num_heads=8,
            depth_drop_path=0.0,
            thermal_embed_dim=768,
            thermal_kernel_size=16,
            thermal_num_blocks=12,
            thermal_num_heads=12,
            thermal_drop_path=0.0,
            imu_embed_dim=512,
            imu_kernel_size=8,
            imu_num_blocks=6,
            imu_num_heads=8,
            imu_drop_path=0.7,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

    def _create_modality_preprocessors(
            self,
            video_frames=2,
            vision_embed_dim=1024,
            kernel_size=(2, 14, 14),
            text_embed_dim=768,
            audio_embed_dim=768,
            audio_kernel_size=16,
            audio_stride=10,
            audio_num_mel_bins=128,
            audio_target_len=204,
            depth_embed_dim=768,
            depth_kernel_size=16,
            thermal_embed_dim=768,
            thermal_kernel_size=16,
            imu_embed_dim=512,
    ):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=48,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
            self,
            vision_embed_dim=1024,
            vision_num_blocks=24,
            vision_num_heads=16,
            text_embed_dim=768,
            text_num_blocks=12,
            text_num_heads=12,
            audio_embed_dim=768,
            audio_num_blocks=12,
            audio_num_heads=12,
            audio_drop_path=0.0,
            depth_embed_dim=768,
            depth_num_blocks=12,
            depth_num_heads=12,
            depth_drop_path=0.0,
            thermal_embed_dim=768,
            thermal_num_blocks=12,
            thermal_num_heads=12,
            thermal_drop_path=0.0,
            imu_embed_dim=512,
            imu_num_blocks=6,
            imu_num_heads=8,
            imu_drop_path=0.7,
    ):
        def instantiate_trunk(
                embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
            self,
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        )
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                    modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)

                # NOTE: No heads are needed any more.
                # head_inputs = modality_value["head"]
                # modality_value = self.modality_heads[modality_key](
                #     modality_value, **head_inputs
                # )

                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                # NOTE: The reduction operation has been modified.
                if reduce_list:
                    modality_value = modality_value.reshape(B, S, *modality_value.shape[1:])
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs


def imagebind_huge(pretrained=False, freeze_imagebind=False):
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained:
        if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )

        model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))

    if freeze_imagebind:
        for name, param in model.named_parameters():
            param.requires_grad = False
        model = model.eval()
        model.train = disabled_train

    return model
