## Download the COCO captions, RefCOCO, RefCOCO+. RefCOCOg, visual genome, textcaps, LLaVA, gqa, AOK-VQA, OK-VQA, OCR-VQA, filtered Flickr-30k, multi-task conversation, and Unnatural instruction datasets


### COCO captions
- [train2017](http://images.cocodataset.org/zips/train2017.zip)


### Visual genome
- [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

### TextCaps

-[TextCaps_0.1_train](https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json)
-[Images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)

### RefCOCO, RefCOCO+, RefCOCOg
Make sure you have the COCO 2014 images first. 

Then,
download RefCOCO, RefCOCO+, and RefCOCOg annotation files in the following links.

- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

Unzip these files to the location you like. It should have the structure like the following

```
Location_you_like
├── refcoco
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(unc).p
├── refcoco+
│   ├── instances.json
│   └── refs(unc).p
└── refcocog
    ├── instances.json
    ├── refs(google).p
    └── refs(umd).p
```

Set **image_path** in all the following dataset configuration files to the COCO 2014 image folder.
Similarly, set **ann_path** in all the following configs to the above folder (Location_you_like) that contains refcoco, refcoco+, and refcocog.

- [minigpt4/configs/datasets/coco_bbox/refcoco.yaml](../minigpt4/configs/datasets/coco_bbox/refcoco.yaml)
- [minigpt4/configs/datasets/coco_bbox/refcocog.yaml](../minigpt4/configs/datasets/coco_bbox/refcocog.yaml) 
- [minigpt4/configs/datasets/coco_bbox/refcocop.yaml](../minigpt4/configs/datasets/coco_bbox/refcocop.yaml)
- [minigpt4/configs/datasets/coco_bbox/invrefcoco.yaml](../minigpt4/configs/datasets/coco_bbox/invrefcoco.yaml)
- [minigpt4/configs/datasets/coco_bbox/invrefcocog.yaml](../minigpt4/configs/datasets/coco_bbox/invrefcocog.yaml) 
- [minigpt4/configs/datasets/coco_bbox/invrefcocop.yaml](../minigpt4/configs/datasets/coco_bbox/invrefcocop.yaml)

### LLaVA
Makesure you have the COCO 2014 images first. 

Download Llava annotation files in the following link to the place you like.

- https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json
- https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json
- https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json

Set **image_path** in all the following dataset configuration files to the COCO 2014 image folder.
Similarly, set **ann_path** to the location of the previous downloaded conversation_58k.json, 
detail_23k.json, and complex_reasoning_77k.json in conversation.yaml, detail.yaml, and reason.yaml, respectively.


- [minigpt4/configs/datasets/llava/conversation.yaml](../minigpt4/configs/datasets/llava/conversation.yaml)
- [minigpt4/configs/datasets/llava/detail.yaml](../minigpt4/configs/datasets/llava/detail.yaml) 
- [minigpt4/configs/datasets/llava/reason.yaml](../minigpt4/configs/datasets/llava/reason.yaml)


### OKVQA

- [OK-VQA Input Questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip)
- [OK-VQA Annotations](https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip)
- [okvqa_train](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/okvqa/okvqa_train.json)
- Images are from COCO

### AOK-VQA

```
export AOKVQA_DIR=YOUR_DATASET_PATH
mkdir -p ${AOKVQA_DIR}
curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}
```

### OCR-VQA
- [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**

### filtered Flickr-30k

### Multi-task conversation

### Unnatural instruction
