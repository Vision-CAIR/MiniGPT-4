import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, hidden_size, expert, num_experts, route_method, topk=1, use_balance_loss=True, weight_type='raw_prob, topk(softmax)'):
        # remove hash list
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        self.route_method = route_method
        self.topk = topk
        self.use_balance_loss = use_balance_loss
        self.weight_type = weight_type

        if route_method in ["gate-token", "gate-sentence"]:
            self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()
        elif route_method in ["gate-sentence-post"]:
            gate = nn.Linear(hidden_size, 1, bias=False).float()
            # self.gates = nn.ModuleList([copy.deepcopy(gate) for i in range(num_experts)])
            self.gate = gate
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

    def _forward_gate_sentence_post(self, x, attention_mask):
        """
            x: query_attention_output; torch.Size([bz, 32, 768])
            attention_mask: torch.ones([bz, 32])
            bz = 4
            x = torch.randn(bz,32,768)
            attention_mask = torch.ones([bz, 32])

        """
        # Prepare Input x
        attention_mask = torch.ones(attention_mask.shape[0], attention_mask.shape[1]).to(x.device)
        x_masked = x * attention_mask.unsqueeze(-1) # torch.Size([bz, 32, 768])
        
        # FeedForward(x) & Forward Gate
        outputs = list()
        logits_gate_lst = list()
        for expert_idx in range(self.num_experts):
            output_x = self.experts[expert_idx].forward(x_masked)
            outputs.append(output_x.unsqueeze(0))
            
            output_x_aver = torch.mean(output_x, dim=1)
            # gate_acore = self.gates[expert_idx](output_x_aver)
            gate_score = self.gate(output_x_aver)
            logits_gate_lst.append(gate_score)
        candidate_output = torch.cat(outputs) # torch.Size([num_expert, bz, 32, 768])
        logits_gate = torch.cat(logits_gate_lst,dim=1)# torch.Size([bz, num_expert])

        # Probabilities for each sample of what expert it should be sent to.
        prob_gate = F.softmax(logits_gate, dim=-1) # torch.Size([bz, num_experts])
        if 'softmax(topk)' in self.weight_type:
            prob_gate1, gate = torch.topk(logits_gate, self.topk, dim=1)
            select_prob_gate = F.softmax(prob_gate1, dim=-1)
        else:
            select_prob_gate, gate = torch.topk(prob_gate, self.topk, dim=1) # gate, 每个样本被分配的expert: torch.Size([bz, topk])

        # Calculate Balancing Loss
        if self.use_balance_loss:
            num_sentences = F.one_hot(gate, self.num_experts).sum(1).gt(0).sum(0) # 每个expert被分配的样本数 torch.Size([num_expert])
            balance_loss = self._balancing_loss(prob_gate, num_sentences)
        else:
            balance_loss = 0.0
        # Calculate Importance Loss
        importance_loss = self._importance_auxiliary_loss(prob_gate)

        # Reshap Prob_gate & Gate
        # expert_mask: [batch_size, topk, num_experts]
        # expert_gate: [batch_size, topk, num_experts]
        # combine_tensor: [batch_size, num_experts]
        expert_mask = F.one_hot(gate, self.num_experts) 
        expert_gate = select_prob_gate.unsqueeze(-1) * expert_mask
        combine_tensor = torch.sum(expert_gate, dim=1) 
        # combine_tensor = torch.zeros_like(prob_gate)
        # combine_tensor.scatter_(1, gate, select_prob_gate) # 等价操作，但可能不可导

        candidate_output_ad = torch.permute(candidate_output, (1, 0, 2, 3)) # torch.Size([bz, num_expert, 32, 768])
        results = candidate_output_ad * combine_tensor.unsqueeze(-1).unsqueeze(-1) # torch.Size([bz, num_expert, 32, 768])
        outputs = torch.sum(results, dim=1) # torch.Size([bz, 32, 768])
        import pdb;pdb.set_trace()

        return outputs, (balance_loss+importance_loss), combine_tensor

    def pre_router(self, x, attention_mask):
        # Prepare input x
        attention_mask = torch.ones(attention_mask.shape[0], attention_mask.shape[1]).to(x.device)
        x_masked = x * attention_mask.unsqueeze(-1) # torch.Size([bz, 32, 768])
        x_average = torch.mean(x_masked, dim=1) # torch.Size([bz, 768])
        
        # Forward Gate
        # logits_gate: [bz, num_experts]
        logits_gate = self.gate(x_average)

        # Probabilities for each sample of what expert it should be sent to.
        # prob_gate: [bz, num_experts]
        prob_gate = F.softmax(logits_gate, dim=-1)
        
        if 'softmax(topk)' in self.weight_type:
            prob_gate1, gate = torch.topk(logits_gate, self.topk, dim=1)
            select_prob_gate = F.softmax(prob_gate1, dim=-1)
        else:
            # topk(softmax)
            # Get Top-K experts for each sample
            # gate: [bz, topk]
            # select_prob_gate: [bz, topk]
            select_prob_gate, gate = torch.topk(prob_gate, self.topk, dim=1)

        # Reshap Prob_gate & Gate
        # expert_mask: [batch_size, topk, num_experts]
        # expert_gate: [batch_size, topk, num_experts]
        # combine_tensor: [batch_size, num_experts]
        expert_mask = F.one_hot(gate, self.num_experts) 
        expert_gate = select_prob_gate.unsqueeze(-1) * expert_mask
        combine_tensor = torch.sum(expert_gate, dim=1) 

        # Calculate Balancing Loss
        if self.use_balance_loss:
            num_sentences = F.one_hot(gate, self.num_experts).sum(1).gt(0).sum(0) # 每个expert被分配的样本数 torch.Size([num_expert])
            balance_loss = self._balancing_loss(prob_gate, num_sentences)
        else:
            balance_loss = 0.0

        # Calculate Importance Loss
        importance_loss = self._importance_auxiliary_loss(prob_gate)

        import pdb; pdb.set_trace()

        return expert_mask, combine_tensor, balance_loss, importance_loss

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
        # Forward Router
        expert_mask, combine_tensor, balance_loss, importance_loss = self.pre_router(x, attention_mask)
        
        # Forward Expert FFN
        result = []
        for expert_idx in range(self.num_experts):
            output_x = self.experts[expert_idx].forward(x)
            result.append(output_x.unsqueeze(0))
        expert_output = torch.cat(result).permute(1,0,2,3) # torch.Size([batch_size, num_expert, num_tokens, hidden_states])

        # multiply outputs of experts by the routing probability
        expert_outputs_combined = expert_output * combine_tensor.unsqueeze(-1).unsqueeze(-1) # torch.Size([batch_size, num_expert, num_tokens, hidden_states])
        outputs = torch.sum(expert_outputs_combined, dim=1) # torch.Size([batch_size, num_tokens, hidden_states])

        import pdb; pdb.set_trace()

        return outputs, (balance_loss+importance_loss), combine_tensor


    def forward(self, x, attention_mask):
        if self.route_method == "gate-token":
            x, balance_loss, gate_load = self._forward_gate_token(x)
        elif self.route_method == "gate-sentence":
            x, balance_loss, gate_load = self._forward_gate_sentence(x, attention_mask)
        elif self.route_method == "gate-sentence-post":
            x, balance_loss, gate_load = self._forward_gate_sentence_post(x, attention_mask)
        else:
            raise KeyError("Routing method not supported.")
        # import pdb; pdb.set_trace()
        return x, balance_loss, gate_load

if __name__ == '__main__':

    import sys
    sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE")
    from minigpt4.models.QformerRouteMoE import BertConfig
    from minigpt4.models.QformerRouteMoE import FeedForward
    from minigpt4.models.moe.utils import (
        moe_layer_judge,
    )

    vision_width = 1408
    cross_attention_freq = 2
    num_query_token = 32
    # init_QformerMoE
    config = BertConfig.from_pretrained("/mnt/pfs-guan-ssai/nlu/wanghanzi/models/bert-base-uncased")
    config.encoder_width = vision_width
    # insert cross-attention layer every other block
    config.add_cross_attention = True
    config.cross_attention_freq = cross_attention_freq
    config.query_length = num_query_token
    config.moebert_expert_num = 3
    config.moebert_num_beams = 2
    config.moebert_route_method = 'gate-sentence-post'
    config.moe_topk = 1
    config.use_balance_loss = False
    # config.moe_weight_type = 'raw_prob, softmax(topk)'
    config.moe_weight_type = 'raw_prob, topk(softmax)'

    batch_size = 4
    x2 = torch.randn(batch_size, 32, 768)
    beam_scores, expert_route = None, None

    for layer_num in [6, 8, 10]:
        layer_judge = moe_layer_judge(layer_num)
        ffn = FeedForward(config)
        gate = nn.Linear(768, config.moebert_expert_num, bias=False).float()

        experts_moe = MoELayer(
                hidden_size=config.hidden_size,
                expert=ffn,
                num_experts=config.moebert_expert_num,
                route_method=config.moebert_route_method,
                topk=config.moe_topk,
                use_balance_loss=config.use_balance_loss,
                weight_type=config.moe_weight_type,
            )
        attn_mask = torch.ones([batch_size, 32])
        layer_output = experts_moe(x2, attn_mask)
        hidden_states3, aux_loss, combine_tensor = layer_output
        
        print(combine_tensor)
        print(aux_loss)
        x2 = hidden_states3

        print("------------------------------------")
        import pdb; pdb.set_trace()