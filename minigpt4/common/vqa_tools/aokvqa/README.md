# A-OKVQA

Official repository for **A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge**.

Links: [[Paper]](https://arxiv.org/abs/2206.01718) [[Website]](http://a-okvqa.allenai.org) [[Leaderboard]](https://leaderboard.allenai.org/a-okvqa/submissions/public)

### Abstract

The Visual Question Answering (VQA) task aspires to provide a meaningful testbed for the development of AI models that can jointly reason over visual and natural language inputs. Despite a proliferation of VQA datasets, this goal is hindered by a set of common limitations. These include a reliance on relatively simplistic questions that are repetitive in both concepts and linguistic structure, little world knowledge needed outside of the paired image, and limited reasoning required to arrive at the correct answer. We introduce A-OKVQA, a crowdsourced dataset composed of a diverse set of about 25K questions requiring a broad base of commonsense and world knowledge to answer. In contrast to the existing knowledge-based VQA datasets, the questions generally cannot be answered by simply querying a knowledge base, and instead require some form of commonsense reasoning about the scene depicted in the image.  We demonstrate the potential of this new dataset through a detailed analysis of its contents and baseline performance measurements over a variety of state-of-the-art vision–language models.

![dataset_web](https://user-images.githubusercontent.com/28768645/170799740-f0d9ea60-6aff-4322-98d5-cae8e05983f4.svg)

<hr>

#### Table of Contents

- [Getting started](#getting-started)
  * [Downloading the dataset](#downloading-the-dataset)
- [Evaluation & Leaderboard](#evaluation)
- [Codebase](#codebase)
  * [Preparing data](#preparing-data)
  * [Models and Predictions](#models-and-predictions)

<hr>

## Getting started

```bash
git clone --single-branch --recurse-submodules https://github.com/allenai/aokvqa.git

cd aokvqa
export PYTHONPATH=.

conda env create --name aokvqa
conda activate aokvqa
```

### Downloading the dataset

```bash
export AOKVQA_DIR=./datasets/aokvqa/
mkdir -p ${AOKVQA_DIR}

curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}
```

<details> <summary><b>Downloading COCO 2017</b></summary>

```bash
export COCO_DIR=./datasets/coco/
mkdir -p ${COCO_DIR}

for split in train val test; do
    wget "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip" -d ${COCO_DIR}; rm "${split}2017.zip"
done

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ${COCO_DIR}; rm annotations_trainval2017.zip
```

</details>

Loading our dataset is easy! Just grab our [load_aokvqa.py](https://github.com/allenai/aokvqa/blob/main/load_aokvqa.py) file and refer to the following code.

```python
import os
aokvqa_dir = os.getenv('AOKVQA_DIR')

from load_aokvqa import load_aokvqa, get_coco_path
train_dataset = load_aokvqa(aokvqa_dir, 'train')  # also 'val' or 'test'
```

<details> <summary><b>Example dataset entry</b></summary>

```python
dataset_example = train_dataset[0]

print(dataset_example['question_id'])
# 22MexNkBPpdZGX6sxbxVBH

coco_dir = os.getenv('COCO_DIR')
image_path = get_coco_path('train', dataset_example['image_id'], coco_dir)
print(image_path)
# ./datasets/coco/train2017/000000299207.jpg

print(dataset_example['question'])
print(dataset_example['choices'])
# What is the man by the bags awaiting?
# ['skateboarder', 'train', 'delivery', 'cab']

correct_choice = dataset_example['choices'][ dataset_example['correct_choice_idx'] ]
# Corrrect: cab

print(dataset_example['rationales'][0])
# A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him so a cab is the only possible answer.
```

</details>

## Evaluation

Please prepare `predictions_{split}.json` files (for `split: {val,test}`) in the format below. You may omit either `multiple_choice` or `direct_answer` field if you only want to evaluate one setting.

```python
{
    '<question_id>' : {
        'multiple_choice' : '<prediction>',
        'direct_answer' : '<prediction>'
    }
}
```

You can run evaluation on the validation set as follows.

```bash
python evaluation/eval_predictions.py --aokvqa-dir ${AOKVQA_DIR} --split val --preds ./predictions_val.json
```

### Leaderboard

You may submit `predictions_test.json` to the [leaderboard](https://leaderboard.allenai.org/a-okvqa/submissions/get-started).

## Codebase

We provide all code and pretrained models necessary to replicate our experiments for Large-Scale Pretrained Models (sec. 5.2) and Rationale Generation (sec. 5.3).

### Preparing data

```bash
export FEATURES_DIR=./features/
mkdir -p ${FEATURES_DIR}
```

You can compute CLIP features for our vocabulary and dataset. These are most commonly used by our other experiments.

```bash
python data_scripts/encode_vocab_clip.py --vocab ${AOKVQA_DIR}/large_vocab_train.csv --model-type ViT-B/32 --out ${FEATURES_DIR}/clip-ViT-B-32_large_vocab.pt

for split in train val test; do
    python data_scripts/extract_clip_features.py --aokvqa-dir ${AOKVQA_DIR} --coco-dir ${COCO_DIR} --split ${split} --model-type ViT-B/32 --out ${FEATURES_DIR}/clip-ViT-B-32_${split}.pt
done
```

<details> <summary><b>For training ClipCap with a transformer mapping network</b></summary>

If you want to train our ClipCap models with the transformer mapping network (instead of an MLP, like we do), you'll also need to run `extract_clip_features.py` with `--model-type RN50x4`.

</details>

<details> <summary><b>For ResNet and BERT input features</b></summary>

Our ResNet and BERT classification experiments require these respective features instead of CLIP. To generate these, please run the following commands:

```bash
# ResNet
for split in train val test; do
    python data_scripts/extract_resnet_features.py --aokvqa-dir ${AOKVQA_DIR} --coco-dir ${COCO_DIR} --split ${split} --out ${FEATURES_DIR}/resnet_${split}.pt
done

# BERT
for split in train val test; do
    python data_scripts/extract_bert_features.py --aokvqa-dir ${AOKVQA_DIR} --split ${split} --out ${FEATURES_DIR}/bert_${split}.pt
done
```

</details>

### Models and Predictions

```bash
export LOG_DIR=./logs/
export PREDS_DIR=./predictions/
export PT_MODEL_DIR=./pretrained_models/
mkdir -p ${LOG_DIR} ${PREDS_DIR} ${PT_MODEL_DIR}
```

<details> <summary><b>Download our pretrained model weights</b></summary>

```bash
# Checkpoints for transfer learning experiments
curl -fsSL https://prior-model-weights.s3.us-east-2.amazonaws.com/aokvqa/transfer_exp_checkpoints.tar.gz | tar xvz -C ${PT_MODEL_DIR}/aokvqa_models

# Checkpoints for ClipCap models (generating answers and rationales)
curl -fsSL https://prior-model-weights.s3.us-east-2.amazonaws.com/aokvqa/clipcap_checkpoints.tar.gz | tar xvz -C ${PT_MODEL_DIR}/aokvqa_models
```

</details>

We have included instructions for replicating each of our experiments (see README.md files below).

All Python scripts should be run from the root of this repository. Please be sure to first run the installation and data preparation as directed above.

- [Heuristics](./heuristics/README.md)
- [Transfer Learning Experiments](./transfer_experiments/README.md)
- [Querying GPT-3](./gpt3/README.md)
- [ClipCap](https://github.com/allenai/aokvqa/blob/ClipCap/README.md)
- [Generating Captions & Rationales](https://github.com/allenai/aokvqa/blob/ClipCap/README.md)

For each experiment, we follow this prediction file naming scheme: `{model-name}_{split}-{setting}.json` (e.g. `random-weighted_val-mc.json` or `random-weighted_test-da.json`). As examples in these Readme files, we produce predictions on the validation set.

We unify predictions for each split before evaluation. (You can omit one of `--mc` or `--da` prediction file if you only want to evaluate one setting.)

```bash
python evaluation/prepare_predictions.py --aokvqa-dir ${AOKVQA_DIR} --split val --mc ./predictions_val-mc.json --da ./predictions_val-da.json --out ./predictions_val.json
# repeat for test split ...
```
