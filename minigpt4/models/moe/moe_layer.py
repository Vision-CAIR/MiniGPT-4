import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    def __init__(self, hidden_size, expert, num_experts, route_method, topk=1):
        # remove hash list
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        self.route_method = route_method
        self.topk = topk
        if route_method in ["gate-token", "gate-sentence"]:
            self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()
        else:
            raise KeyError("Routing method not supported.")

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

    def _forward_gate_token(self, x):
        bsz, seq_len, dim = x.size()

        x = x.view(-1, dim)
        logits_gate = self.gate(x)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_tokens.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_tokens.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            input_x = input_x * prob_x
            return input_x

        x = [forward_expert(x[i], prob_gate[i], i) for i in range(self.num_experts)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)

        return x, balance_loss, gate_load

    def _forward_gate_sentence_top1_raw(self, x, attention_mask):
        """
            x: query_attention_output , torch.Size([bz, 32, 768])
            attention_mask: torch.ones([bz, 32])
            
            ### Notice:
            the raw version of expert_attention_mask is the extended_attention_mask, 
            which will be add to attention_score directly
            the values of extended_attention_mask are -0.0 or -10000
            it should be adjust to 1/0 version to be processed by experts
        """
        attention_mask = torch.ones(attention_mask.shape[0], attention_mask.shape[1]).to(x.device)
        x_masked = x * attention_mask.unsqueeze(-1) # torch.Size([bz, 32, 768])
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1) # torch.Size([bz, 768])
        logits_gate = self.gate(x_average) # torch.Size([bz, num_experts])
        prob_gate = F.softmax(logits_gate, dim=-1) # torch.Size([bz, num_experts])
        gate = torch.argmax(prob_gate, dim=-1) # torch.Size([bz])

        order = gate.argsort(0)
        num_sentences = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_sentences.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_sentences.tolist(), dim=0)  # a list of length self.num_experts

        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_sentences.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_sentences.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            input_x = input_x * prob_x.unsqueeze(-1)
            return input_x

        result = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                result.append(forward_expert(x[i], prob_gate[i], i))
        result = torch.vstack(result)
        result = result[order.argsort(0)]  # restore original order

        return result, balance_loss, gate_load

    def _weighted_select_expert(self, expert_ids, prob_gate_i, ):
        # expert_ids: torch.Size([topk]) 为sample选出的topk个expert的idx
        # prob_gate_i: torch.Size([topk]) 该sample对应expert的概率值
        # 先对 prob_gate 归一化，加权平均 expert_qt 的值

        weight = [prob_gate_i[expert_id].item() for expert_id in expert_ids]
        weight_norm = torch.tensor(weight) / torch.tensor(weight).sum()
        select_qts = [query_token_candidates[expert_id] for expert_id in expert_ids]
        weighted_qt = [select_qts[i] * weight_norm[i] for i in range(weight_norm.shape[0])]
        select = sum(weighted_qt).unsqueeze(0)
        return select
    
    def _forward_gate_sentence(self, x, attention_mask):
        """
            x: query_attention_output , torch.Size([bz, 32, 768])
            attention_mask: torch.ones([bz, 32])
            
            ### Notice:
            the raw version of expert_attention_mask is the extended_attention_mask, 
            which will be add to attention_score directly
            the values of extended_attention_mask are -0.0 or -10000
            it should be adjust to 1/0 version to be processed by experts
        """
        attention_mask = torch.ones(attention_mask.shape[0], attention_mask.shape[1]).to(x.device)
        x_masked = x * attention_mask.unsqueeze(-1) # torch.Size([bz, 32, 768])
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1) # torch.Size([bz, 768])
        logits_gate = self.gate(x_average) # torch.Size([bz, num_experts])
        prob_gate = F.softmax(logits_gate, dim=-1) # torch.Size([bz, num_experts])
        select_prob_gate, gate = torch.topk(prob_gate, self.topk, dim=1) # gate, 每个样本被分配的expert: torch.Size([bz, topk])
        normalized_tensor = torch.nn.functional.normalize(select_prob_gate, p=2, dim=0) # L2 Normalization  torch.Size([bz, topk])

        num_sentences = F.one_hot(gate, self.num_experts).sum(1).gt(0).sum(0) # 每个expert被分配的样本数 torch.Size([num_expert])
        gate_load = num_sentences.clone()

        # load balancing loss
        balance_loss = self._balancing_loss(prob_gate, num_sentences)

        # importance loss
        importance_loss = self._importance_auxiliary_loss(prob_gate)

        # forward experts
        def forward_expert(input_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            return input_x

        result_lst = list()
        for i in range(self.topk):
            # top1、top2... 分别为一组，进行gate分组之后过expert，然后乘以概率后相加
            tmp_gate = gate[:,i]
            tmp_prob = normalized_tensor[:,i].unsqueeze(-1).unsqueeze(-1)
            order = tmp_gate.argsort(0)
            num_sentences_t = F.one_hot(tmp_gate, self.num_experts).gt(0).sum(0)
            x1 = x[order]  # reorder according to expert number
            x1 = x1.split(num_sentences_t.tolist(), dim=0)  # a list of length self.num_experts

            result = []
            for i in range(self.num_experts):
                if x1[i].size(0) > 0:
                    result.append(forward_expert(x1[i], i))
            result = torch.vstack(result)
            result = result[order.argsort(0)]  # restore original order
            result_lst.append(result * tmp_prob) # result * prob

        moe_result = sum(result_lst)

        return moe_result, (balance_loss+importance_loss), gate_load


    def _forward_sentence_single_expert(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)
        logits_gate = self.gate(x_average)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        gate_load = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        x = self.experts[gate.cpu().item()].forward(x)
        return x, 0.0, gate_load

    def forward(self, x, attention_mask):
        if self.route_method == "gate-token":
            x, balance_loss, gate_load = self._forward_gate_token(x)
        elif self.route_method == "gate-sentence":
            if x.size(0) == 1:
                x, balance_loss, gate_load = self._forward_sentence_single_expert(x, attention_mask)
            else:
                x, balance_loss, gate_load = self._forward_gate_sentence(x, attention_mask)
        else:
            raise KeyError("Routing method not supported.")

        return x, balance_loss, gate_load
