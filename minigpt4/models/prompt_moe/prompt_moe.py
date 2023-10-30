import math
import os
import copy
import pickle
import torch
from torch import nn
from torch import nn
import torch.nn.functional as F
from minigpt4.models.Qformer import BertConfig

def init_query_token_candidates(num_query_token, num_cand):
    encoder_config = BertConfig.from_pretrained("/mnt/pfs-guan-ssai/nlu/wanghanzi/models/bert-base-uncased")
    query_token_candidates = nn.Parameter(
        torch.zeros(num_cand, num_query_token, encoder_config.hidden_size)
    )
    query_token_candidates.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return query_token_candidates

class PromptMoEBase(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super(PromptMoEBase, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

    def _balancing_loss(self, prob_gate, num_tokens):
        # From MOEBERT
        # compute the load balancing loss
        # prob_gate，是 [bz, num_expert]，每个样本被分配给每个expert的概率
        # 等价于 VMOE 中 _gshard_auxiliary_loss
        P = prob_gate.mean(0) # torch.Size([num_expert]) 每个expert被分配到样本的平均概率
        temp = num_tokens.float()
        f = temp / temp.sum(0, keepdim=True) # 每个expert被分配的sample比例
        balance_loss = self.num_experts * torch.sum(P * f) 
        return balance_loss

    def _importance_auxiliary_loss(self, prob_gate):
        # From VMOE
        # _importance_auxiliary_loss
        axis = tuple(range(prob_gate.ndim - 1))  # All except last.
        importance_per_expert = torch.sum(prob_gate, dim=axis)
        std_importance_per_expert = torch.std(importance_per_expert)
        mean_importance_per_expert = torch.mean(importance_per_expert)
        # Compute coefficient of variation (i.e. std/mean) squared.
        return (std_importance_per_expert / mean_importance_per_expert)**2

    def _weighted_select_expert(self, expert_ids, prob_gate_i, query_token_candidates):
        # expert_ids: torch.Size([topk]) 为sample选出的topk个expert的idx
        # prob_gate_i: torch.Size([topk]) 该sample对应expert的概率值
        # query_token_candidates: torch.Size([num_expert, 32, 768])
        # 先对 prob_gate 归一化，加权平均 expert_qt 的值
        weight = [prob_gate_i[expert_id].item() for expert_id in expert_ids]
        weight_norm = torch.tensor(weight) / torch.tensor(weight).sum()
        select_qts = [query_token_candidates[expert_id] for expert_id in expert_ids]
        weighted_qt = [select_qts[i] * weight_norm[i] for i in range(weight_norm.shape[0])]
        select = sum(weighted_qt).unsqueeze(0)
        return select

class PostPromptMoE(PromptMoEBase):
    def __init__(self, hidden_size, num_experts, topk=1):
        super(PostPromptMoE, self).__init__(hidden_size, num_experts)
        self.gate = nn.Linear(hidden_size, 1, bias=False).float()
        self.topk = topk

    def _forward_gate_text_single_token(self, text_embeds, candi_query_tokens):
        # text embedding output from the blip2: torch.Size([bz, num_qt_candidates, 768])
        # candidate query tokens to be selected : torch.Size([bz*num_qt_candidates, 32, 768])
        logits_gate = self.gate(text_embeds).squeeze(2) # torch.Size([bz, num_qt_candidates])
        prob_gate = F.softmax(logits_gate, dim=-1) # torch.Size([bz, num_qt_candidates])

        _, gate = torch.topk(prob_gate, self.topk, dim=1) # gate, 每个样本被分配的expert: torch.Size([bz, topk])
        num_tokens = F.one_hot(gate, self.num_experts).sum(1).gt(0).sum(0) # 每个expert被分配的样本数 torch.Size([num_expert])
        gate_load = num_tokens.clone()

        # load balancing loss
        balance_loss = self._balancing_loss(prob_gate, num_tokens)

        # importance loss
        importance_loss = self._importance_auxiliary_loss(prob_gate)

        # select expert(query_token) for each sample
        out = [self._weighted_select_expert(gate[i], prob_gate[i], candi_query_tokens[i*self.num_experts:(i+1)*self.num_experts])  for i in range(gate.shape[0])]
        out = torch.vstack(out) # [bz, 32, 768]
        return out, balance_loss, importance_loss, gate_load, gate


class PrePromptMoE(PromptMoEBase):
    def __init__(self, hidden_size, num_experts, query_token_candidates, route_method, topk=1):
        super(PrePromptMoE, self).__init__(hidden_size, num_experts)
        self.query_token_candidates = query_token_candidates
        self.route_method = route_method
        self.topk = topk
        if route_method in ["gate-token", "gate-single-token", "gate-sentence"]:
            self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()
            print(self.gate)
        else:
            raise KeyError("Routing method not supported.")

    def _forward_gate_single_token(self, x):
        bsz, seq_len, dim = x.size()

        x = x.view(-1, dim)
        logits_gate = self.gate(x)  # torch.Size([bz, num_expert])
        prob_gate = F.softmax(logits_gate, dim=-1) # torch.Size([bz, num_expert])
        
        _, gate = torch.topk(prob_gate, self.topk, dim=1) # gate, 每个样本被分配的expert: torch.Size([bz, topk])
        num_tokens = F.one_hot(gate, self.num_experts).sum(1).gt(0).sum(0) # 每个expert被分配的样本数 torch.Size([num_expert])

        # gate = torch.argmax(prob_gate, dim=-1) # 每个样本被分配的expert
        # num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0) 
        gate_load = num_tokens.clone()

        # load balancing loss
        balance_loss = self._balancing_loss(prob_gate, num_tokens)

        # importance loss
        importance_loss = self._importance_auxiliary_loss(prob_gate)

        # select expert(query_token) for each sample

        out = [self._weighted_select_expert(gate[i], prob_gate[i], self.query_token_candidates)  for i in range(gate.shape[0])]

        out = torch.vstack(out) # [bz, 32, 768]

        return out, balance_loss, importance_loss, gate_load, gate

    def _forward_gate_token(self, x):
        bsz, seq_len, dim = x.size()

        x = x.view(-1, dim)
        logits_gate = self.gate(x)  # torch.Size([bz, num_expert])
        prob_gate = F.softmax(logits_gate, dim=-1) # torch.Size([bz, num_expert])
        gate = torch.argmax(prob_gate, dim=-1) # 每个样本被分配的expert

        order = gate.argsort(0) # index of sorted gate(ascending)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0) # 每个expert被分配的样本数 torch.Size([num_expert])
        gate_load = num_tokens.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        # load balancing loss
        balance_loss = self._balancing_loss(prob_gate, num_tokens)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_tokens.tolist(), dim=0) # prob_gate tuple，根据expert分组

        def select_expert(prob_x, expert_idx):
            input_x = self.query_token_candidates[expert_idx] # [1, 32, 768]
            # input_x = input_x * prob_x
            input_x = input_x.expand(prob_x.shape[0], -1, -1)
            return input_x

        out = [select_expert(prob_gate[i], i) for i in range(self.num_experts)]
        out = torch.vstack(out)
        out = out[order.argsort(0)]  # restore original order

        return out, balance_loss, gate_load, gate

    def _forward_gate_sentence(self, x, attention_mask):
        ### TODO: refer MOEBERT
        return None

    def _forward_sentence_single_expert(self, x, attention_mask):
        ### TODO: refer MOEBERT
        return None


    # def forward(self, x, attention_mask=None):
    #     if self.route_method == "gate-single-token":
    #         x, balance_loss, gate_load, gate = self._forward_gate_single_token(x)
    #     elif self.route_method == "gate-token":
    #         x, balance_loss, gate_load, gate = self._forward_gate_token(x)
    #     elif self.route_method == "gate-sentence":
    #         if x.size(0) == 1:
    #             x, balance_loss, gate_load, gate = self._forward_sentence_single_expert(x, attention_mask)
    #         else:
    #             x, balance_loss, gate_load, gate = self._forward_gate_sentence(x, attention_mask)
    #     else:
    #         raise KeyError("Routing method not supported.")

    #     return x, balance_loss, gate_load, gate
