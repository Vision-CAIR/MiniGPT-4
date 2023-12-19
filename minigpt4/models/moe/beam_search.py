import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    def __init__(self, hidden_size, expert, gate, num_experts, route_method, topk=1, use_balance_loss=True, weight_type='l2_norm'):
        # remove hash list
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        self.route_method = route_method
        self.topk = topk
        self.use_balance_loss = use_balance_loss
        self.weight_type = weight_type

        if route_method in ["gate-token", "gate-sentence"]:
            self.gate = gate
        else:
            raise KeyError("Routing method not supported.")


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

        # 这里用l2 norm 去加权
        if self.weight_type == 'l2_norm':
            # normalized_tensor = torch.nn.functional.normalize(select_prob_gate, p=2, dim=0) # L2 Normalization  torch.Size([bz, topk])
            normalized_tensor = select_prob_gate

        num_sentences = F.one_hot(gate, self.num_experts).sum(1).gt(0).sum(0) # 每个expert被分配的样本数 torch.Size([num_expert])
        gate_load = num_sentences.clone()

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

            # result_lst.append(result * tmp_prob) # result * prob
            result_lst.append(result) # result * prob

        moe_result = sum(result_lst)
        print('Layer Qformer MoE: \n',prob_gate)
        return moe_result, select_prob_gate, gate


    def forward(self, x, attention_mask):
        if self.route_method == "gate-token":
            x, balance_loss, gate_load = self._forward_gate_token(x)
        elif self.route_method == "gate-sentence":
            if x.size(0) == 1:
                x, balance_loss, gate_load = self._forward_sentence_single_expert(x, attention_mask)
            else:
                x, balance_loss, gate_load = self._forward_gate_sentence(x, attention_mask)
        elif self.route_method == "gate-sentence-post":
            x, balance_loss, gate_load = self._forward_gate_sentence_post(x, attention_mask)
        else:
            raise KeyError("Routing method not supported.")

        return x, balance_loss, gate_load
    

class RouteMoELayer(nn.Module):
    def __init__(self, hidden_size, expert, gate, num_experts, num_beams=2, layer_judge=None, route_method="pre-route"):
        # remove hash list
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        self.num_beams = num_beams
        self.hidden_size = hidden_size
        self.layer_judge = layer_judge

        self.route_method = route_method
        if self.route_method == "pre-route":
            self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()
        elif self.route_method == "post-route":
            # gate = nn.Linear(hidden_size, 1, bias=False).float()
            self.gates = nn.ModuleList([copy.deepcopy(gate) for i in range(num_experts)])

    def forward_gate(self, x):
        """
            x : torch.Size([bz*num_beams, 32, 768]) or torch.Size([bz, 32, 768]) 
            prob_gate : torch.Size([bz*num_beams, num_experts])  or torch.Size([bz, num_experts]) 
        """
        attention_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)
        x_masked = x * attention_mask.unsqueeze(-1) # torch.Size([bz*num_beams, 32, 768])
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1) # torch.Size([bz*num_beams, 768])
        logits_gate = self.gate(x_average) # torch.Size([bz*num_beams, num_experts])
        prob_gate = F.softmax(logits_gate, dim=-1) #  torch.Size([bz*num_beams, num_experts])
        return prob_gate

    def beam_search(self, current_scores_log, beam_scores, expert_route, batch_size):
        import pdb;pdb.set_trace()
        if self.layer_judge=='first' and self.route_method=='pre-route':
            assert beam_scores==None and expert_route==None
            current_scores = torch.exp(current_scores_log)
            topk_values, gate = torch.topk(current_scores, self.num_beams, dim=1) # gate, 每个样本被分配的expert: torch.Size([bz, topk])
            beam_scores = topk_values.view(self.num_beams * batch_size) # torch.Size([bz * num_beams])
            expert_route = gate.view(self.num_beams * batch_size).unsqueeze(1) # torch.Size([bz * num_beams,1])
            beam_idx = None
        else:
            if self.layer_judge=='first' and self.route_method == 'post-route':
                batch_size = batch_size
                next_scores_raw1 = torch.exp(current_scores_log) # torch.Size([bz, num_experts])
            else:
                batch_size = int(batch_size // self.num_beams)
                next_scores_raw = current_scores_log + torch.log(beam_scores).unsqueeze(1)  # torch.Size([4*3, 5]) # 取log 之后，可以直接相加概率
                next_scores_exp = torch.exp(next_scores_raw)
                next_scores_raw1 = next_scores_exp.view(
                    batch_size, self.num_beams * self.num_experts
                )  # torch.Size([bz, num_beams*num_experts])

            next_scores, next_experts = torch.topk(next_scores_raw1, self.num_beams, dim=1, largest=True, sorted=True)
            # next_scores torch.Size([bz, num_beams])
            # next_tokens torch.Size([bz, num_beams])
            print(next_scores_raw1)
            print(next_scores)
            print(next_experts)

            next_batch_beam = list()
            for batch_idx in range(batch_size):
                next_sent_beam = list()
                for rank, (expert_id, expert_score) in enumerate(
                    zip(next_experts[batch_idx], next_scores[batch_idx])
                ):
                    expert_id = expert_id.item()
                    beam_id = expert_id // self.num_experts
                    ex_id = expert_id % self.num_experts
                    effective_beam_id = batch_idx*self.num_beams + beam_id

                    next_sent_beam.append((expert_score, ex_id, effective_beam_id))
                next_batch_beam.extend(next_sent_beam)

            import pdb;pdb.set_trace()

            if self.layer_judge=='first' and self.route_method == 'post-route':
                beam_scores = next_scores.view(self.num_beams * batch_size) # torch.Size([bz * num_beams])
                expert_route = next_experts.view(self.num_beams * batch_size)
                beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
                beam_experts = expert_route.new([x[1] for x in next_batch_beam]).unsqueeze(-1)
                beam_idx = expert_route.new([int(x[2]/self.num_beams) for x in next_batch_beam])
                expert_route = beam_experts
            else:
                beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
                beam_experts = expert_route[:,-1].new([x[1] for x in next_batch_beam])
                beam_idx = expert_route[:,-1].new([x[2] for x in next_batch_beam])
                pre_route = expert_route[beam_idx,:]
                expert_route = torch.cat([pre_route, beam_experts.unsqueeze(1)], dim=-1)

        import pdb;pdb.set_trace()

        return beam_scores, expert_route, beam_idx
    
    
    def forward_expert_ffn(self, x, expert_select, beam_scores):
        """
            x_repeat : [bz*num_beams, 32,768]
            expert_select : [bz*num_beams]
        """
        # add_1212 l2_normalization
        # normalized_tensor = torch.nn.functional.normalize(beam_scores, p=2, dim=0) # L2 Normalization  torch.Size([bz, topk])
        # tmp_prob = normalized_tensor.unsqueeze(-1).unsqueeze(-1)

        outputs = list()
        for i in range(x.shape[0]):
            output_x = self.experts[expert_select[i]].forward(x[i])
            outputs.append(output_x.unsqueeze(0))
        candidate_output = torch.cat(outputs) 

        # candidate_output = candidate_output * tmp_prob
        return candidate_output # torch.Size([bz*num_beams, 32, 768])


    def forward_pre_route(self, x, beam_scores, expert_route, use_log=True):
        
        current_scores = self.forward_gate(x) # [bz*num_beams, 5]

        if use_log:
            current_scores_log = torch.log(current_scores) # 取log之后可以直接相加
        else:
            current_scores_log = current_scores

        batch_size, num_tokens = x.shape[0], x.shape[1]
        beam_scores, expert_route, _ = self.beam_search(current_scores_log, beam_scores, expert_route, batch_size)

        current_expert_select = expert_route[:,-1]

        if self.layer_judge=='first': # expand first dim to batch_size * num_beams
            replicated_tensor = x.unsqueeze(1).expand(batch_size, self.num_beams, num_tokens, self.hidden_size)
            x = replicated_tensor.contiguous().view(-1, num_tokens, self.hidden_size) # [bz*num_beams, 32,768]

        candidate_output = self.forward_expert_ffn(x, current_expert_select, beam_scores) # [bz*num_beams, 32,768]
        
        return candidate_output, beam_scores, expert_route


    def forward_post_route(self, x, beam_scores, expert_route, use_log=True):
        
        # if self.layer_judge=='first': # expand first dim to batch_size * num_beams
        #     batch_size, num_tokens = x.shape[0], x.shape[1]
        #     replicated_tensor = x.unsqueeze(1).expand(batch_size, self.num_beams, num_tokens, self.hidden_size)
        #     x = replicated_tensor.contiguous().view(-1, num_tokens, self.hidden_size) # [bz*num_beams, 32,768]
    
        attention_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)
        x_masked = x * attention_mask.unsqueeze(-1) # torch.Size([bz, 32, 768])
        
        def forward_expert(input_x, expert_idx):
            output_x = self.experts[expert_idx].forward(input_x)
            return output_x

        outputs = list()
        logits_gate_lst = list()
        for expert_idx in range(self.num_experts):
            output_x = forward_expert(x_masked, expert_idx)
            outputs.append(output_x.unsqueeze(0))
            output_x_aver = output_x.sum(1) / attention_mask.unsqueeze(-1).sum(1) # torch.Size([bz*num_beam, 768])
            gate_acore = self.gates[expert_idx](output_x_aver)
            logits_gate_lst.append(gate_acore)
        candidate_output = torch.cat(outputs) # torch.Size([num_expert, bz*num_beam, 32, 768])
        logits_gate = torch.cat(logits_gate_lst,dim=1)# torch.Size([bz*num_beam, num_expert])
        current_scores = F.softmax(logits_gate, dim=-1) # torch.Size([bz*num_beam, num_experts])

        if use_log:
            current_scores_log = torch.log(current_scores) # 取log之后可以直接相加
        else:
            current_scores_log = current_scores
        
        import pdb;pdb.set_trace()

        batch_size = x.shape[0] # bz*num_beam
        beam_scores, expert_route, beam_idx = self.beam_search(current_scores_log, beam_scores, expert_route, batch_size)
        # beam_scores torch.Size([bz*num_beam])
        # expert_route torch.Size([bz*num_beam, layer_n])
        current_select_expert = expert_route[:,-1]
        
        output = list()
        for i in range(beam_idx.shape[0]):
            b_idx = beam_idx[i]
            ex_idx = current_select_expert[i]
            ex_out = candidate_output[ex_idx, b_idx, :,:]
            output.append(ex_out.unsqueeze(0))

        final_output = torch.concat(output, dim=0)

        return final_output, beam_scores, expert_route, beam_idx


    def forward(self, x, attention_mask, beam_scores, expert_route, use_log=True):
        """
            if first_layer: x [bz, 32, 768]
            else: x [bz*num_beams, 32, 768]
        
        """
        if self.route_method == 'pre-route':
            candidate_output, beam_scores, expert_route, _ = self.forward_pre_route(x, beam_scores, expert_route, use_log=True)
        elif self.route_method == "post-route":
            candidate_output, beam_scores, expert_route, beam_idx = self.forward_post_route(x, beam_scores, expert_route, use_log=True)

        return candidate_output, beam_scores, expert_route, beam_idx
    

if __name__ == '__main__':

    import sys
    sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE")
    from minigpt4.models.QformerRouteMoE import BertConfig
    from minigpt4.models.QformerRouteMoE import FeedForward

    from minigpt4.models.moe.utils import (
        use_experts,
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
    config.moebert_num_beams = 3
    config.moebert_route_method = 'gate-sentence'
    config.moe_topk = 2
    config.use_balance_loss = False
    config.moe_weight_type = 'l2_norm'

    batch_size = 4
    x = torch.randn(batch_size, 32, 768)
    beam_scores, expert_route = None, None

    x1 = x
    x2 = x
    beam_scores1, expert_route1 = None, None

    for layer_num in [6, 8, 10]:
        layer_judge = moe_layer_judge(layer_num)
        ffn = FeedForward(config)
        gate = nn.Linear(768, config.moebert_expert_num, bias=False).float()

        # experts = RouteMoELayer(
        #             hidden_size=768,
        #             expert=ffn,
        #             gate = gate,
        #             num_experts=config.moebert_expert_num,
        #             num_beams=config.moebert_num_beams,
        #             layer_judge = layer_judge,
        #             route_method = "pre-route"
        #         )
        # layer_output = experts(x, None, beam_scores, expert_route)
        # hidden_states1, beam_scores, expert_route,_ = layer_output

        # print(beam_scores)
        # print(expert_route)

        gate1 = nn.Linear(768, 1, bias=False).float()
        experts_post = RouteMoELayer(
                    hidden_size=768,
                    expert=ffn,
                    gate = gate1,
                    num_experts=config.moebert_expert_num,
                    num_beams=config.moebert_num_beams,
                    layer_judge = layer_judge,
                    route_method = "post-route"
                )
        layer_output = experts_post(x1, None, beam_scores1, expert_route1, False)
        hidden_states2, beam_scores1, expert_route1, beam_idx = layer_output

        print(beam_scores1)
        print(expert_route1)
        print(beam_idx)

        # experts_moe = MoELayer(
        #         hidden_size=config.hidden_size,
        #         expert=ffn,
        #         gate=gate,
        #         num_experts=config.moebert_expert_num,
        #         route_method=config.moebert_route_method,
        #         topk=config.moe_topk,
        #         use_balance_loss=config.use_balance_loss,
        #         weight_type=config.moe_weight_type,
        #     )
        # attn_mask = torch.ones([batch_size, 32])
        # layer_output = experts_moe(x2, attn_mask)
        # hidden_states3, select_prob_gate, gate_load,_ = layer_output
        
        # print(select_prob_gate)
        # print(gate_load)


        # x = hidden_states1
        x1 = hidden_states2
        # x2 = hidden_states3

        print("------------------------------------")

