## Download the dataset for finetuning the MiniGPT-v2


Download the dataset

Image source | Download path
--- | :---:
COCO 2014 images | <a href="http://images.cocodataset.org/zips/train2014.zip">images</a> &nbsp;&nbsp;  <a href="https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json"> captions</a>
COCO VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_train.json">vqa train</a> &nbsp;&nbsp;  <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_val.json"> vqa val</a>
Visual Genome |  <a href="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip">images part1</a> &nbsp;&nbsp; <a href="https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip">images part2</a> &nbsp;&nbsp; <a href="https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"> image meta data </a>
TextCaps | <a href="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip">images</a>  &nbsp;&nbsp; <a href="https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json"> annotations</a> 
RefCOCO | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"> annotations </a>
RefCOCO+ | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"> annotations </a>
RefCOCOg | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"> annotations </a>
OKVQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/okvqa/okvqa_train.json"> annotations </a>
AOK-VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/aokvqa/aokvqa_v1p0_train.json"> annotations </a>
OCR-VQA | <a href="https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing"> annotations </a>
GQA | <a href="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip">images</a>  &nbsp;&nbsp; <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/gqa/train_balanced_questions.json"> annotations </a>
Filtered flickr-30k |  <a href="https://drive.google.com/drive/folders/19c_ggBI77AvdtYlPbuI0ZpnPz73T5teX?usp=sharing"> annotations </a>
Multi-task conversation |  <a href="https://drive.google.com/file/d/11HHqB2c29hbSk-WLxdta-nG8UCUrcCN1/view?usp=sharing"> annotations </a> 
Filtered unnatural instruction |  <a href="https://drive.google.com/file/d/1lXNnBcb5WU-sc8Fe2T2N8J0NRw4sBLev/view?usp=sharing"> annotations </a>
LLaVA | <a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json"> Compelex reasoning </a> &nbsp;&nbsp;<a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json"> Detailed description </a> &nbsp;&nbsp; <a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json"> Conversation </a> 



### COCO captions
Download the COCO 2014 images and captions

coco 2014 images path

```
${MINIGPTv2_DATASET}
├── coco
│   ├── images
...
```


coco caption annotation path

```
${MINIGPTv2_DATASET}
├── coco_captions
│   └── annotations
│       ├── coco_karpathy_train.json
...
```

Set **image_path** to the COCO 2014 image folder.
Similarly, set **ann_path** to the coco_karpathy_train.json path
- [minigpt4/configs/datasets/coco/caption.yaml](../minigpt4/configs/datasets/coco/caption.yaml)

### COCO VQA
Download the vqa v2 train and validation json files

```
├── ${MINIGPTv2_DATASET}
│   ├── vqav2
│       ├── vqa_train.json
|       ├── vqa_val.json
```

Set **image_path** to the COCO 2014 image folder.
Similarly, set **ann_path** to the vqa_train.json and vqa_val.json path
- [minigpt4/configs/datasets/coco/defaults_vqa.yaml](../minigpt4/configs/datasets/coco/defaults_vqa.yaml)


### Visual genome
Download visiual genome images and annotation files

```
${MINIGPTv2_DATASET}
├── visual_genome
│   ├── VG_100K
│   ├── VG_100K_2
│   └── region_descriptions.json
│   └── image_data.json
...
```

Set **image_path** to visual_genome folder.
Similarly, set **ann_path** to the visual_genome folder.

- [minigpt4/configs/datasets/vg/ref.yaml](../minigpt4/configs/datasets/vg/ref.yaml)


### TextCaps
Download the TextCaps images and annotation files

```
├── ${MINIGPTv2_DATASET}
│   ├── textcaps
│       ├── train_images
│       ├── TextCaps_0.1_train.json
```

Set **image_path** to TextCaps train_images folder.
Similarly, set **ann_path** to the TextCaps_0.1_train.json path

- [minigpt4/configs/datasets/textcaps/caption.yaml](../minigpt4/configs/datasets/textcaps/caption.yaml)

### RefCOCO, RefCOCO+, RefCOCOg
Download the RefCOCO, RefCOCO+, RefCOCOg annotation files

```

${MINIGPTv2_DATASET}
├── refcoco_annotations
│   ├── refcoco
│   │   ├── instances.json
│   │   ├── refs(google).p
│   │   └── refs(unc).p
│   ├── refcoco+
│   │   ├── instances.json
│   │   └── refs(unc).p
│   └── refcocog
│       ├── instances.json
│       ├── refs(google).p
│       └─── refs(und).p
...
```


Set **image_path** to the COCO 2014 image folder.
Similarly, set **ann_path** in all the following configs to the above folder *refcoco_annotations* that contains refcoco, refcoco+, and refcocog.

- [minigpt4/configs/datasets/coco_bbox/refcoco.yaml](../minigpt4/configs/datasets/coco_bbox/refcoco.yaml)
- [minigpt4/configs/datasets/coco_bbox/refcocog.yaml](../minigpt4/configs/datasets/coco_bbox/refcocog.yaml) 
- [minigpt4/configs/datasets/coco_bbox/refcocop.yaml](../minigpt4/configs/datasets/coco_bbox/refcocop.yaml)
- [minigpt4/configs/datasets/coco_bbox/invrefcoco.yaml](../minigpt4/configs/datasets/coco_bbox/invrefcoco.yaml)
- [minigpt4/configs/datasets/coco_bbox/invrefcocog.yaml](../minigpt4/configs/datasets/coco_bbox/invrefcocog.yaml) 
- [minigpt4/configs/datasets/coco_bbox/invrefcocop.yaml](../minigpt4/configs/datasets/coco_bbox/invrefcocop.yaml)




### OKVQA


```
Location_you_like
├── ${MINIGPTv2_DATASET}
│   ├── okvqa
│       ├── okvqa_train.json
```

Set **image_path** to the COCO 2014 image folder.
Similarly, set **ann_path** to the location of the OKVQA dataset
- [minigpt4/configs/datasets/okvqa/defaults.yaml](../minigpt4/configs/datasets/okvqa/defaults.yaml)


### COCO-VQA

- [OK-VQA Input Questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip)
- [OK-VQA Annotations](https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip)


### AOK-VQA
Download the AOK-VQA annotation dataset

```
export AOKVQA_DIR=YOUR_DATASET_PATH
mkdir -p ${AOKVQA_DIR}
curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}
```

```
Location_you_like
├── ${MINIGPTv2_DATASET}
│   ├── aokvqa
│       ├── aokvqa_v1p0_train.json
```


Set **image_path** to the COCO 2014 image folder.
Similarly, set **ann_path** to the location of the AOKVQA dataset
- [minigpt4/configs/datasets/aokvqa/defaults.yaml](../minigpt4/configs/datasets/aokvqa/defaults.yaml)



### OCR-VQA
Download the OCR-VQA annotation files
download the images with loadDataset.py script

```
Location_you_like
├── ${MINIGPTv2_DATASET}
│   ├── ocrvqa
│       ├── images
│       ├── dataset.json
```

Set **image_path** as the ocrvqa/images folder.
Similarly, set **ann_path** to the dataset.json
- [minigpt4/configs/datasets/ocrvqa/ocrvqa.yaml](../minigpt4/configs/datasets/ocrvqa/ocrvqa.yaml)

### GQA
Download the GQA annotation files and images

```
Location_you_like
├── ${MINIGPTv2_DATASET}
│   ├── gqa
│       ├── images
│       ├── train_balanced_questions.json
```

Set **image_path** as the gqa/images folder.
Similarly, set **ann_path** to the train_balanced_questions.json
- [minigpt4/configs/datasets/gqa/balanced_val.yaml](../minigpt4/configs/datasets/gqa/balanced_val.yaml)



### filtered Flickr-30k
Download filtered Flickr-30k images (fill this [form](https://forms.illinois.edu/sec/229675) on official website or from [kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/download?datasetVersionNumber=1)) and annotation files

```
${MINIGPTv2_DATASET}
├── filtered_flickr
│   ├── images
│   ├── captiontobbox.json
│   ├── groundedcaption.json
│   └── phrasetobbox.json
...
```

Set **image_path** as the flickr-30k images foler.
Similarly, set **ann_path** to the groundedcaption.json, captiontobbox.json and phrasetobbox.json for the 
grounded image caption, caption to bbox, and phrase to bbox datasets.

- [minigpt4/configs/datasets/flickr/default.yaml](../minigpt4/configs/datasets/flickr/default.yaml)
- [minigpt4/configs/datasets/flickr/caption_to_phrase.yaml](../minigpt4/configs/datasets/flickr/caption_to_phrase.yaml)
- [minigpt4/configs/datasets/flickr/object_to_phrase.yaml](../minigpt4/configs/datasets/flickr/object_to_phrase.yaml)


### Multi-task conversation
Download the multi-task converstation dataset

```
Location_you_like
${MINIGPTv2_DATASET}
├── multitask_conversation
│   └── multitask_conversation.json
...
```

Set **image_path** as the COCO 2014 images folder.
Similarly, set **ann_path** to the multitask_conversation.json file path

- [minigpt4/configs/datasets/multitask_conversation/default.yaml](../minigpt4/configs/datasets/multitask_conversation/default.yaml)

### Unnatural instruction
Download the filtered unnatural instruction annotation files (we remove the very long sentences from the original unnatural instruction dataset)

```
Location_you_like
├── ${MINIGPTv2_DATASET}
│   ├── unnatural_instructions
│       ├── filtered_unnatural_instruction.json
```

There is no image path.
Similarly, set **ann_path** to the filtered_unnatural_instruction.json file path

- [minigpt4/configs/datasets/nlp/unnatural_instruction.yaml](../minigpt4/configs/datasets/nlp/unnatural_instruction.yaml)

### LLaVA

```
Location_you_like
├── ${MINIGPTv2_DATASET}
│   ├── llava
│       ├── conversation_58k.json
│       ├── detail_23k.json
│       ├── complex_reasoning_77k.json
```

Set **image_path** to the COCO 2014 image folder.
Similarly, set **ann_path** to the location of the previous downloaded conversation_58k.json, 
detail_23k.json, and complex_reasoning_77k.json in conversation.yaml, detail.yaml, and reason.yaml, respectively.


- [minigpt4/configs/datasets/llava/conversation.yaml](../minigpt4/configs/datasets/llava/conversation.yaml)
- [minigpt4/configs/datasets/llava/detail.yaml](../minigpt4/configs/datasets/llava/detail.yaml) 
- [minigpt4/configs/datasets/llava/reason.yaml](../minigpt4/configs/datasets/llava/reason.yaml)
