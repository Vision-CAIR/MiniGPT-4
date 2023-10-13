## Training of MiniGPT-4

The training of MiniGPT-4 contains two alignment stages.

**1. First pretraining stage**

In the first pretrained stage, the model is trained using image-text pairs from Laion and CC datasets
to align the vision and language model. To download and prepare the datasets, please check 
our [first stage dataset preparation instruction](dataset/README_1_STAGE.md). 
After the first stage, the visual features are mapped and can be understood by the language
model.
To launch the first stage training, run the following command. In our experiments, we use 4 A100. 
You can change the save path in the config file 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage1_pretrain.yaml)

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

A MiniGPT-4 checkpoint with only stage one training can be downloaded 
[here (13B)](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link) or [here (7B)](https://drive.google.com/file/d/1HihQtCEXUyBM1i9DQbaK934wW3TZi-h5/view?usp=share_link).
Compared to the model after stage two, this checkpoint generate incomplete and repeated sentences frequently.


**2. Second finetuning stage**

In the second stage, we use a small high quality image-text pair dataset created by ourselves
and convert it to a conversation format to further align MiniGPT-4.
To download and prepare our second stage dataset, please check our 
[second stage dataset preparation instruction](dataset/README_2_STAGE.md).
To launch the second stage alignment, 
first specify the path to the checkpoint file trained in stage 1 in 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage2_finetune.yaml).
You can also specify the output path there. 
Then, run the following command. In our experiments, we use 1 A100.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

After the second stage alignment, MiniGPT-4 is able to talk about the image coherently and user-friendly. 
