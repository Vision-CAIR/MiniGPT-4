import logging
import os

import torch
from torch import nn, Tensor

from minigpt4.common.dist_utils import download_cached_file
from minigpt4.common.utils import is_url
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class BaseQFormer(nn.Module):
    def __init__(self, freeze_qformer=False):
        super().__init__()
        self.freeze_qformer = freeze_qformer
        self.Qformer = None

    def check_and_freeze(self):
        assert self.Qformer is not None
        if self.freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("Freeze This QFormer")

    @classmethod
    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


class SequenceGenericQFormer(BaseQFormer):
    def __init__(self,
                 num_query_token: int,
                 encoder_width: int = 768,
                 freeze_qformer: bool = False,
                 q_former_model: str = "",
                 cross_attention_freq: int = 2
                 ):
        super().__init__(freeze_qformer)
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, encoder_width, cross_attention_freq)
        if q_former_model != "":
            self.load_Qformer(q_former_model)
        self.check_and_freeze()

    def load_Qformer(self, q_former_model):
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

    @classmethod
    def init_Qformer(cls, num_query_token, encoder_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = encoder_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def forward(self, input_embeds: Tensor) -> Tensor:
        input_atts = torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(input_embeds.device)
        query_tokens = self.query_tokens.expand(input_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=input_embeds,
            encoder_attention_mask=input_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state
