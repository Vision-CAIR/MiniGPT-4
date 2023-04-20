## How to Prepare Vicuna Weight
Vicuna is an open-source LLAMA-based LLM that has a performance close to ChatGPT. 
We currently use the v0 version of Vicuna-13B. 

To prepare Vicuna’s weight, first download Vicuna’s **delta** weight from [https://huggingface.co/lmsys/vicuna-13b-delta-v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0). 
In case you have git-lfs installed (https://git-lfs.com), this can be done by

```
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0  # more powerful, need at least 24G gpu memory
# or
git clone https://huggingface.co/lmsys/vicuna-7b-delta-v0  # smaller, need 12G gpu memory
```

Note that this is not directly the working weight, but the difference between the working weight and the original weight of LLAMA-13B. (Due to LLAMA’s rules, we cannot distribute the weight of LLAMA.)

Then, you need to obtain the original LLAMA-7B or LLAMA-13B weights in the HuggingFace format 
either following the instruction provided by HuggingFace 
[here](https://huggingface.co/docs/transformers/main/model_doc/llama) or from the Internet. 

When these two weights are ready, we can use tools from Vicuna’s team to create the real working weight.
First, Install their library that is compatible with v0 Vicuna by

```
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

Then, run the following command to create the final working weight

```
python -m fastchat.model.apply_delta --base /path/to/llama-13bOR7b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13bOR7b-delta-v0/
```

Now you are good to go!

