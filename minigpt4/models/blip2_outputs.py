"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


@dataclass
class BlipSimilarity(ModelOutput):
    sim_i2t: torch.FloatTensor = None
    sim_t2i: torch.FloatTensor = None

    sim_i2t_m: Optional[torch.FloatTensor] = None
    sim_t2i_m: Optional[torch.FloatTensor] = None

    sim_i2t_targets: Optional[torch.FloatTensor] = None
    sim_t2i_targets: Optional[torch.FloatTensor] = None


@dataclass
class BlipIntermediateOutput(ModelOutput):
    """
    Data class for intermediate outputs of BLIP models.

    image_embeds (torch.FloatTensor): Image embeddings, shape (batch_size, num_patches, embed_dim).
    text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim).

    image_embeds_m (torch.FloatTensor): Image embeddings from momentum visual encoder, shape (batch_size, num_patches, embed_dim).
    text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder, shape (batch_size, seq_len, embed_dim).

    encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder.
    encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder for negative pairs.

    decoder_output (CausalLMOutputWithCrossAttentions): output from the image-grounded text decoder.
    decoder_labels (torch.LongTensor): labels for the captioning loss.

    itm_logits (torch.FloatTensor): logits for the image-text matching loss, shape (batch_size * 3, 2).
    itm_labels (torch.LongTensor): labels for the image-text matching loss, shape (batch_size * 3,)

    """

    # uni-modal features
    image_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None

    image_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    itm_logits: Optional[torch.FloatTensor] = None
    itm_labels: Optional[torch.LongTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None


@dataclass
class BlipOutput(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    sims: Optional[BlipSimilarity] = None

    intermediate_output: BlipIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_itc: Optional[torch.FloatTensor] = None

    loss_itm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None


@dataclass
class BlipOutputFeatures(ModelOutput):
    """
    Data class of features from BlipFeatureExtractor.

    Args:
        image_embeds: (torch.FloatTensor) of shape (batch_size, num_patches+1, embed_dim), optional
        image_features: (torch.FloatTensor) of shape (batch_size, num_patches+1, feature_dim), optional
        text_embeds: (torch.FloatTensor) of shape (batch_size, sequence_length+1, embed_dim), optional
        text_features: (torch.FloatTensor) of shape (batch_size, sequence_length+1, feature_dim), optional

        The first embedding or feature is for the [CLS] token.

        Features are obtained by projecting the corresponding embedding into a normalized low-dimensional space.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    image_embeds_proj: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None

    multimodal_embeds: Optional[torch.FloatTensor] = None
