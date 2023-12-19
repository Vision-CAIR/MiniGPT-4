import torch
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def forward_expert(input_x, expert_idx):
    input_x += torch.randn(32,768)
    return input_x
    # output_x = self.experts[expert_idx].forward(input_x)
    # return output_x


def forward_ffn(x_repeat, expert_select):
    """
        x_repeat : [bz*num_beams, 32,768]
        expert_select : [bz*num_beams]
    """
    outputs = list()
    num_beams_bz = x_repeat.shape[0]
    for i in range(num_beams_bz):
        output_x = forward_expert(x_repeat[i], expert_select[i]) # (32,768)
        outputs.append(output_x.unsqueeze(0))
    candidate_output = torch.cat(outputs) 
    return candidate_output # torch.Size([bz*num_beams, 32, 768])

def forward_gate(x, num_expert):
    """
        x : torch.Size([bz*num_beams, 32, 768]) or torch.Size([bz, 32, 768]) 
        prob_gate : torch.Size([bz*num_beams, num_experts])  or torch.Size([bz, num_experts]) 
    """
    # attention_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)
    # x_masked = x * attention_mask.unsqueeze(-1) # torch.Size([bz*num_beams, 32, 768])
    # x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1) # torch.Size([bz*num_beams, 768])
    # logits_gate = gate(x_average) # torch.Size([bz, num_experts])
    logits_gate = torch.randn(x.shape[0], num_expert)
    prob_gate = F.softmax(logits_gate, dim=-1) #  torch.Size([bz*num_beams, num_experts])
    return prob_gate

def beam_search(layer, current_scores, beam_scores, expert_route, num_beams):
    if layer == 0 and beam_scores==None and expert_route==None:
        topk_values, gate = torch.topk(current_scores, num_beams, dim=1) # gate, 每个样本被分配的expert: torch.Size([bz, topk])
        beam_scores = topk_values.view(num_beams*batch_size) # torch.Size([bz * num_beams])
        expert_route = gate.view(num_beams*batch_size).unsqueeze(1) # torch.Size([bz * num_beams])

    else:
        next_scores_raw = current_scores + beam_scores.unsqueeze(1)  # torch.Size([4*3, 5]) # 取log 之后，可以直接相加概率
        next_scores_raw1 = next_scores_raw.view(
            batch_size, num_beams * num_expert
        )  # torch.Size([4, 3*5])
        next_scores, next_experts = torch.topk(next_scores_raw1, num_beams, dim=1, largest=True, sorted=True)
        # next_scores torch.Size([4, 3*num_beams])
        # next_tokens torch.Size([4, 3*num_beams])

        next_batch_beam = list()
        for batch_idx in range(batch_size):
            next_sent_beam = list()
            print(batch_idx)
            for rank, (expert_id, expert_score) in enumerate(
                zip(next_experts[batch_idx], next_scores[batch_idx])
            ):
                expert_id = expert_id.item()
                beam_id = expert_id // num_expert
                ex_id = expert_id % num_expert
                effective_beam_id = batch_idx*num_beams + beam_id

                # print(expert_id, beam_id, ex_id, effective_beam_id, expert_score)

                next_sent_beam.append((expert_score, ex_id, effective_beam_id))
            next_batch_beam.extend(next_sent_beam)
            
            # print()

        import pdb;pdb.set_trace()

        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_experts = expert_route[:,-1].new([x[1] for x in next_batch_beam])
        beam_idx = expert_route[:,-1].new([x[2] for x in next_batch_beam])

        pre_route = expert_route[beam_idx,:]
        expert_route = torch.cat([pre_route, beam_experts.unsqueeze(1)], dim=-1)

    return beam_scores, expert_route


if __name__ == '__main__':

    batch_size = 3
    num_beams = 2
    num_expert = 5
    x = torch.randn(batch_size, 32, 768)
    beam_scores, expert_route = None, None

    for layer in range(0,3):
        # import pdb;pdb.set_trace()

        current_scores = forward_gate(x, num_expert)
        import pdb;pdb.set_trace()

        beam_scores, expert_route = beam_search(layer, current_scores, beam_scores, expert_route, num_beams)
        current_expert_select = expert_route[:,-1]

        if layer == 0:
            replicated_tensor = x.unsqueeze(1).expand(batch_size, num_beams, 32, 768)
            x = replicated_tensor.contiguous().view(-1, 32, 768) # [12,32,768] [bz*num_beams, 32,768]
        else:
            x = candidate_output

        candidate_output = forward_ffn(x, current_expert_select) # torch.Size([4*3, 5])

        x = candidate_output


    scores = beam_scores.view(batch_size, num_beams)
    topk_values, gate = torch.topk(scores, 1, dim=1)
    # gate [batch_size, 1]
    # topk_values [batch_size, 1]
    selects = [ (bz_idx * num_beams + gate[bz_idx].item()) for bz_idx in range(batch_size)]
    final_scores = beam_scores[selects]
    final_expert_route = expert_route[selects]
    final_output = candidate_output[selects]







# def forward_ffn_post(x_repeat, expert_select):
#     """
#         x_repeat : [bz*num_beams, 32,768]
#         expert_select : [bz*num_beams]
#         prob_gate : torch.Size([bz*num_beams, num_experts])
#     """
#     outputs = list()
#     logits_gate_lst = list()
#     # attention_mask = torch.ones([batch_size, 32])
#     for i in range(num_beams*batch_size):
#         output_x = forward_expert(x_repeat[i], expert_select[i]) # (32,768)
#         outputs.append(output_x.unsqueeze(0))
#         # output_x_aver = output_x.sum(1) / attention_mask.unsqueeze(-1).sum(1) # torch.Size([bz, 768])
#         # gate_acore = self.gates[expert_idx](output_x_aver)
#         # gate_score = self.gate(output_x_aver)
#         num_expert = 5
#         gate_score = torch.randn(1,num_expert)
#         logits_gate_lst.append(gate_score)

#     candidate_output = torch.cat(outputs) # torch.Size([bz*num_beams, 32, 768])
#     logits_gate = torch.cat(logits_gate_lst,dim=0)# torch.Size([bz*num_beams, num_expert])
#     prob_gate = F.softmax(logits_gate, dim=-1) # torch.Size([bz*num_beams, num_experts])
#     return prob_gate, candidate_output