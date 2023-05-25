import dataclasses
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Union, Dict

import torch
from PIL import Image
from torch import nn, Tensor
from transformers import StoppingCriteria, StoppingCriteriaList

from eval_scripts.eval_utils import load_image
from imagebind.models.image_bind import ModalityType
from minigpt4 import BaseProcessor

Roles = SimpleNamespace(
    HUMAN="Human",
    ASSISTANT="Assistant"
)


class Message:
    def __init__(self, role: str, content: Union[str, None]):
        self.role = role
        self.content = content


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    messages: List[Message]
    sep: str = "###"

    def get_prompt(self):
        ret = self.system + self.sep
        for message in self.messages:
            if message.content:
                ret += message.role + ": " + message.content + self.sep
            else:
                ret += message.role + ":"
        return ret

    def append_message(self, role, content):
        self.messages.append(Message(role, content))

    def copy(self):
        return Conversation(
            system=self.system,
            messages=deepcopy(self.messages),
            sep=self.sep)

    def dict(self):
        return {
            "system": self.system,
            "messages": [(msg.role, msg.content) for msg in self.messages],
            "sep": self.sep
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Vision>ImageContent</Vision>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    messages=[],
    sep="###",
)


# TODO: If needed and possible, rewrite this file and re-organize the definition of components.


class Chat:
    def __init__(self,
                 model: nn.Module,
                 processors: Dict[str, BaseProcessor],
                 device: str = 'cuda:0'
                 ):
        self.device = device
        self.model = model
        self.processors = processors
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.just_uploaded = False

    def ask(self, text, conversation):
        # NOTE: the hard code for postfix is removed.
        # end_token = '</Vision>'
        # if len(conversation.messages) > 0 and conversation.messages[-1].role == Roles.HUMAN \
        #         and conversation.messages[-1].content[-len(end_token):] == end_token:
        if self.just_uploaded:
            conversation.messages[-1].content = ' '.join([conversation.messages[-1].content, text])
            self.just_uploaded = False
        else:
            conversation.append_message(Roles.HUMAN, text)

    def answer(self, conversation, emb_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        # Generate an answer written by LLaMA
        conversation.append_message(Roles.ASSISTANT, None)
        embs = self.get_context_emb(conversation, emb_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknown token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conversation.messages[-1].content = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image: Union[str, Image.Image, Tensor], conversation: Conversation, emb_list: List[Tensor]):
        # Upload Image, Encode Image and Create a new message from human.
        image = load_image(image, self.processors[ModalityType.VISION]).to(self.device)
        all_embeddings = self.model.encode_inputs({ModalityType.VISION: image})
        image_emb = all_embeddings[ModalityType.VISION]
        emb_list.append(image_emb)
        conversation.append_message(Roles.HUMAN, "<Vision><ModalityHere></Vision>")
        self.just_uploaded = True

    def get_context_emb(self, conversation: Conversation, emb_list: List[Tensor]):
        # Insert the embeddings into the prompts and queries.
        # NOTE: Assume the placeholders have been aligned to the embeddings!
        prompt = conversation.get_prompt()
        prompt_segs = prompt.split('<ModalityHere>')
        assert len(prompt_segs) == len(emb_list) + 1, "Unmatched numbers of placeholders and embeddings."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], emb_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
