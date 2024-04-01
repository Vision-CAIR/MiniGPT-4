import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniRouteMoELayer(nn.Module):
    def __init__(self, hidden_size, expert, num_experts, num_beams=2, layer_judge=None, route_method="pre-route", weight_type="ffn_prob"):
        # remove hash list
        nn.Module.__init__(self)
        self.num_experts = num_experts #(1+other)
        self.num_route_experts = num_experts-1
        self.num_beams = num_beams
        self.num_route_beam = num_beams-1

        self.experts = nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        self.hidden_size = hidden_size
        self.layer_judge = layer_judge
        self.weight_type = weight_type

        self.route_method = route_method
        if self.route_method in ['pre-route-uni', 'uni-cls-gate-route']:
            self.gate = nn.Linear(hidden_size, self.num_route_experts, bias=False).float()
        elif self.route_method in ["post-route-uni",'uni-cls-route', 'uni-cls-query-route', 'uni-cls-cross-route']:
            gate = nn.Linear(hidden_size, 1, bias=False).float()
            self.gate = gate

    def _importance_auxiliary_loss(self, prob_gate):
        # From VMOE
        # _importance_auxiliary_loss
        axis = tuple(range(prob_gate.ndim - 1))  # All except last.
        importance_per_expert = torch.sum(prob_gate, dim=axis)
        std_importance_per_expert = torch.std(importance_per_expert)
        mean_importance_per_expert = torch.mean(importance_per_expert)
        # Compute coefficient of variation (i.e. std/mean) squared.
        return (std_importance_per_expert / mean_importance_per_expert)**2

    def beam_search(self, current_scores_log, beam_scores, expert_route, batch_size):
        if self.layer_judge=='first' and self.route_method in ['pre-route-uni', 'post-route-uni','uni-cls-route', 'uni-cls-query-route', 'uni-cls-cross-route','uni-cls-gate-route']:
            # current_scores_log torch.Size([bz, num_experts-1])
            assert beam_scores==None and expert_route==None
            current_scores = torch.exp(current_scores_log)
            topk_values, gate = torch.topk(current_scores, self.num_route_beam, dim=1) # gate, 每个样本被分配的expert: torch.Size([bz, topk])
            beam_scores = topk_values.view(self.num_route_beam * batch_size) # torch.Size([bz * num_beams])
            expert_route = gate.view(self.num_route_beam * batch_size).unsqueeze(1) # torch.Size([bz * num_beams,1])
            beam_idx = torch.tensor(range(self.num_route_beam * batch_size))
            
        else:
            batch_size = int(batch_size // self.num_route_beam)
            next_scores_raw = current_scores_log + torch.log(beam_scores).unsqueeze(1)  # torch.Size([4*3, 5]) # 取log 之后，可以直接相加概率
            next_scores_exp = torch.exp(next_scores_raw)

            next_scores_raw1 = next_scores_exp.view(
                batch_size, self.num_route_beam * self.num_route_experts
            )  # torch.Size([bz, num_route_beam*num_route_experts])

            next_scores, next_experts = torch.topk(next_scores_raw1, self.num_route_beam, dim=1, largest=True, sorted=True)
            # next_tokens torch.Size([bz, num_route_beam])

            next_batch_beam = list()
            for batch_idx in range(batch_size):
                next_sent_beam = list()
                for rank, (expert_id, expert_score) in enumerate(
                    zip(next_experts[batch_idx], next_scores[batch_idx])
                ):
                    expert_id = expert_id.item()
                    beam_id = expert_id // self.num_route_experts
                    ex_id = expert_id % self.num_route_experts
                    effective_beam_id = batch_idx*self.num_route_beam + beam_id

                    next_sent_beam.append((expert_score, ex_id, effective_beam_id))
                next_batch_beam.extend(next_sent_beam)

            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_experts = expert_route[:,-1].new([x[1] for x in next_batch_beam])
            beam_idx = expert_route[:,-1].new([x[2] for x in next_batch_beam])
            pre_route = expert_route[beam_idx,:]
            expert_route = torch.cat([pre_route, beam_experts.unsqueeze(1)], dim=-1)

        return beam_scores, expert_route, beam_idx

    def forward_gate(self, x):
        """
            TODO: Pre forward gate
            x : torch.Size([bz*(num_beams-1), 32, 768]) or torch.Size([bz, 32, 768]) 
            prob_gate : torch.Size([bz*(num_beams-1), num_experts])  or torch.Size([bz, num_experts]) 
        """
        attention_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)
        x_masked = x * attention_mask.unsqueeze(-1) # torch.Size([bz*(num_beams-1), 32, 768])
        x_average = torch.mean(x_masked, dim=1) # torch.Size([bz*(num_beams-1), 768])
        logits_gate = self.gate(x_average) # torch.Size([bz*(num_beams-1), num_experts])
        prob_gate = F.softmax(logits_gate, dim=-1) #  torch.Size([bz*(num_beams-1), num_experts])
        return prob_gate

    def forward_expert_ffn(self, x, expert_select, current_scores):
        """
            x_repeat : [bz*num_beams, 32,768]
            expert_select : [bz*num_beams]
            current_scores : [bz*num_beams, num_experts] / [bz, num_experts]
        """
        # import pdb;pdb.set_trace()
        outputs = list()
        for i  in range(self.num_experts-1):
            output_x = self.experts[i].forward(x)
            outputs.append(output_x.unsqueeze(1))
        candidate_output = torch.cat(outputs, dim=1) 
        expert_select_matrix = F.one_hot(expert_select, self.num_experts)
        if self.weight_type == 'ffn_prob':
            tmp_prob = current_scores * expert_select_matrix
            candidate_output = candidate_output * tmp_prob.unsqueeze(-1).unsqueeze(-1)
        else:
            candidate_output = candidate_output * expert_select_matrix.unsqueeze(-1).unsqueeze(-1)
        output = torch.sum(candidate_output, dim=1)
        # import pdb;pdb.set_trace()
        return output # torch.Size([bz*(num_beams-1), 32, 768])

    def forward_pre_route(self, x, beam_scores, expert_route, use_log=True):
        
        current_scores = self.forward_gate(x) # [bz, num_beams] / [bz*num_beams, num_beams]

        importance_loss = self._importance_auxiliary_loss(current_scores)

        if use_log:
            current_scores_log = torch.log(current_scores) # 取log之后可以直接相加
        else:
            current_scores_log = current_scores
        # import pdb;pdb.set_trace()
        batch_size, num_tokens = x.shape[0], x.shape[1]
        beam_scores, expert_route, beam_idx = self.beam_search(current_scores_log, beam_scores, expert_route, batch_size)
        current_expert_select = expert_route[:,-1]

        if self.layer_judge=='first': # expand first dim to batch_size * num_beams
            replicated_tensor = x.unsqueeze(1).expand(batch_size, self.num_beams, num_tokens, self.hidden_size)
            x = replicated_tensor.contiguous().view(-1, num_tokens, self.hidden_size) # [bz*num_beams, 32,768]
            current_scores_t = current_scores.unsqueeze(1).expand(batch_size, self.num_beams, self.num_experts)
            current_scores = current_scores_t.contiguous().view(-1, self.num_experts) # [bz*num_beams, num_experts]

        input_x = x[beam_idx]
        candidate_output = self.forward_expert_ffn(input_x, current_expert_select, current_scores) # [bz*num_beams, 32,768]
        # import pdb;pdb.set_trace()
        return candidate_output, beam_scores, expert_route, beam_idx, importance_loss


    def calculate_cls_gate_score(self, cls_hidden, output_x):

        if self.route_method == 'uni-cls-route':
            # cls_hidden = [bz, 768]
            gate_score = self.gate(cls_hidden) # bz, 1
        elif self.route_method == 'uni-cls-query-route': # add cls_hiddin on query_token mean pool hidden
            mean_output = torch.mean(output_x, dim=1) # bz, 768
            gate_score = self.gate(mean_output+cls_hidden) # bz, 1
        elif self.route_method == 'uni-cls-cross-route':
            # cls_hidden as Q, output_x as K, V calculate scaled dot-product attention between Q and K and V
            # cls_hidden: bz, 768
            # output_x: bz, 32, 768
            Q = cls_hidden.unsqueeze(1) # bz, 1, 768
            K = output_x # bz, 32, 768
            V = output_x # bz, 32, 768
            # scaled dot-product attention
            QK = torch.matmul(Q, K.transpose(-1, -2)) / (K.size(-1) ** 0.5) # bz, 1, 32
            QK = F.softmax(QK, dim=-1) # bz, 1, 32
            gate_score = torch.matmul(QK, V) # bz, 1, 768
            gate_score = gate_score.squeeze(1) # bz, 768
            gate_score = self.gate(gate_score) # bz, 1
        return gate_score

    def adjust_cls_hidden(self, cls_hidden, output_x):
        if cls_hidden.shape[0]/self.num_beams == output_x.shape[0]/self.num_route_beam:
            cls_hidden_lst = list()
            for i in range(cls_hidden.shape[0]):
                if i % self.num_beams != 0:
                    cls_hidden_lst.append(cls_hidden[i,:])
            cls_hidden = torch.stack(cls_hidden_lst)
        return cls_hidden

    def forward_route_uni(self, x, beam_scores, expert_route, cls_hidden=None):
        
        if beam_scores == None:
            batch_size = x.shape[0]
            x_masked, x_uniexpert = x, x # torch.Size([bz, 32, 768])
        elif x.shape[0]/self.num_beams == beam_scores.shape[0]/self.num_route_beam:
            batch_size = int(x.shape[0]/self.num_beams)
            select_universal = [i*self.num_beams+self.num_route_beam for i in range(batch_size)]
            select_expert = [ x for x in range(batch_size*self.num_beams) if x not in select_universal]
            x_masked, x_uniexpert = x[select_expert],x[select_universal]
        num_tokens = x.shape[1]

        def forward_expert(input_x, expert_idx):
            output_x = self.experts[expert_idx].forward(input_x)
            return output_x
        
        ####################
        ### route expert
        ####################
        if cls_hidden is not None:
            cls_hidden = self.adjust_cls_hidden(cls_hidden, x_masked)

        outputs = list()
        logits_gate_lst = list()
        for expert_idx in range(self.num_route_experts): # num_expert-1
            output_x = forward_expert(x_masked, expert_idx)

            if self.route_method == 'post-route-uni':
                output_x_aver = torch.mean(output_x, dim=1)
                gate_score = self.gate(output_x_aver)
                logits_gate_lst.append(gate_score)

            elif self.route_method in ['uni-cls-route', 'uni-cls-query-route', 'uni-cls-cross-route'] and cls_hidden is not None:
                gate_score = self.calculate_cls_gate_score(cls_hidden, output_x)
                logits_gate_lst.append(gate_score)
            outputs.append(output_x.unsqueeze(0))

        candidate_output_raw = torch.cat(outputs) # torch.Size([num_expert-1, bz*(num_beam-1), 32, 768])

        if self.route_method == 'uni-cls-gate-route':
            # universal expert with cls_hidden state into nn.Linear(768,num_experts-1)
            logits_gate = self.gate(cls_hidden)
            current_scores = F.softmax(logits_gate, dim=-1) # torch.Size([bz*(num_beam-1), num_expert-1])
        else:
            logits_gate = torch.cat(logits_gate_lst,dim=1)# torch.Size([bz*(num_beam-1), num_expert-1])
            current_scores = F.softmax(logits_gate, dim=-1) # torch.Size([bz*(num_beam-1), num_expert-1])

        current_scores_log = torch.log(current_scores) # 取log之后可以直接相加 torch.Size([bz*(num_beam-1), num_expert-1])
        
        importance_loss = self._importance_auxiliary_loss(current_scores)
        beam_scores, expert_route, beam_idx = self.beam_search(current_scores_log, beam_scores, expert_route, current_scores_log.shape[0])
        # beam_scores torch.Size([bz*(num_beam-1)]), expert_route torch.Size([bz*(num_beam-1), layer_n])
        current_select_expert = expert_route[:,-1] # torch.Size([bz*(num_beam-1)])

        if self.layer_judge == 'first':
            replicated_tensor = candidate_output_raw.unsqueeze(2).expand(self.num_route_experts, batch_size, self.num_route_beam, num_tokens, self.hidden_size)
            candidate_output_raw = replicated_tensor.contiguous().view(self.num_route_experts, -1, num_tokens, self.hidden_size) # [bz*num_beams, 32,768]
            current_scores_t = current_scores.unsqueeze(1).expand(batch_size, self.num_route_beam, self.num_route_experts)
            current_scores = current_scores_t.contiguous().view(-1, self.num_route_experts) # [bz*(num_beams-1), num_experts-1]
        
        candidate_output = candidate_output_raw.permute(1, 0, 2, 3)[beam_idx] # torch.Size([8, 2, 32, 768])
        expert_select_matrix = F.one_hot(current_select_expert, self.num_route_experts)
        if self.weight_type == 'ffn_prob':
            tmp_prob = current_scores[beam_idx] * expert_select_matrix
            output = candidate_output * tmp_prob.unsqueeze(-1).unsqueeze(-1)
        else:
            output = candidate_output * expert_select_matrix.unsqueeze(-1).unsqueeze(-1)
        experts_output = torch.sum(output, dim=1) # [bz*num_beams-1, 32, 768]

        ####################
        ### universal expert
        ####################
        uni_output = forward_expert(x_uniexpert, self.num_experts-1) # [bz, 32, 768]

        ####################
        ### Combine expert
        ####################
        output = list()
        for i in range(batch_size):
            expert_tmp = experts_output[i*self.num_route_beam: i*self.num_route_beam+self.num_route_beam,:,:]
            combine_tmp = torch.cat((expert_tmp, uni_output[i].unsqueeze(0)))
            output.append(combine_tmp)
        final_output = torch.cat(output) # [bz*num_beam, 32 ,768]

        # import pdb; pdb.set_trace()

        return final_output, beam_scores, expert_route, beam_idx, importance_loss
    
    def forward(self, x, attention_mask, beam_scores, expert_route, cls_hidden):
        """
            if first_layer: x [bz, 32, 768]
            else: x [bz*num_beams, 32, 768]
        """
        if self.route_method == 'pre-route-uni':
            candidate_output, beam_scores, expert_route, beam_idx, importance_loss = self.forward_pre_route(x, beam_scores, expert_route)
        elif self.route_method in ['post-route-uni', 'uni-cls-route', 'uni-cls-query-route', 'uni-cls-cross-route','uni-cls-gate-route']:
            candidate_output, beam_scores, expert_route, beam_idx, importance_loss = self.forward_route_uni(x, beam_scores, expert_route, cls_hidden=cls_hidden)

        return candidate_output, beam_scores, expert_route, beam_idx, importance_loss


