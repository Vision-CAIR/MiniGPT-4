## Finetune of MiniGPT-4


You firstly need to prepare the dataset. you can follow this step to prepare the dataset.
our [dataset preparation](dataset/README_MINIGPTv2_FINETUNE.md). 

In the train_configs/minigptv2_finetune.yaml, you need to set up the following paths:

llama_model checkpoint path: "/path/to/llama_checkpoint"

ckpt: "/path/to/pretrained_checkpoint"

ckpt save path: "/path/to/save_checkpoint"

For ckpt, you may load from our pretrained model checkpoints:
| MiniGPT-v2 (after stage-2) | MiniGPT-v2 (after stage-3) | MiniGPT-v2 (online developing demo) | 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/file/d/1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu/view?usp=sharing) |[Download](https://drive.google.com/file/d/1jAbxUiyl04SFJMN4sF1vvUU69Etuz4qa/view?usp=sharing) | [Download](https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view?usp=sharing) |


```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigptv2_finetune.yaml
```

