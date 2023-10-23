## Download the COCO captions, RefCOCO, RefCOCO+. RefCOCOg, visual genome, textcaps, LLaVA, gqa, AOK-VQA, OK-VQA, OCR-VQA, filtered Flickr-30k, multi-task conversation, and Unnatural instruction datasets

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

### COCO captions
- [train2017](http://images.cocodataset.org/zips/train2017.zip)

### RefCOCO, RefCOCO+, RefCOCOg

### Visual genome
- [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
### TextCaps
Makesure you have the COCO 2014 images first. 

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

- [minigpt4/configs/refcoco.yaml](../minigpt4/configs/refcoco.yaml)
- [minigpt4/configs/refcocog.yaml](../minigpt4/configs/refcocog.yaml) 
- [minigpt4/configs/refcocop.yaml](../minigpt4/configs/refcocop.yaml)
- [minigpt4/configs/invrefcoco.yaml](../minigpt4/configs/invrefcoco.yaml)
- [minigpt4/configs/invrefcocog.yaml](../minigpt4/configs/invrefcocog.yaml) 
- [minigpt4/configs/invrefcocop.yaml](../minigpt4/configs/invrefcocop.yaml)



### Visual Genome

### textcaps

### LLaVA

### TextVQA
- [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
### GQA
- [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- [Annotations](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/gqa/testdev_balanced_questions.json)



### GQA

### OKVQA

### AOK-VQA

### OCR-VQA
- [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**

### filtered Flickr-30k

### Multi-task conversation

### Unnatural instruction













### Pre-training datasets download:
We use the filtered synthetic captions prepared by BLIP. For more details about the dataset, please refer to [BLIP](https://github.com/salesforce/BLIP).

It requires ~2.3T to store LAION and CC3M+CC12M+SBU datasets

Image source | Filtered synthetic caption by ViT-L
--- | :---:
CC3M+CC12M+SBU | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered_large.json">Download</a>
LAION115M |  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered_large.json">Download</a>

This will download two json files 
```
ccs_synthetic_filtered_large.json
laion_synthetic_filtered_large.json
```

## prepare the data step-by-step


### setup the dataset folder and move the annotation file to the data storage folder
```
export MINIGPT4_DATASET=/YOUR/PATH/FOR/LARGE/DATASET/
mkdir ${MINIGPT4_DATASET}/cc_sbu
mkdir ${MINIGPT4_DATASET}/laion
mv ccs_synthetic_filtered_large.json ${MINIGPT4_DATASET}/cc_sbu
mv laion_synthetic_filtered_large.json ${MINIGPT4_DATASET}/laion
```

### Convert the scripts to data storate folder
```
cp convert_cc_sbu.py ${MINIGPT4_DATASET}/cc_sbu
cp download_cc_sbu.sh ${MINIGPT4_DATASET}/cc_sbu
cp convert_laion.py ${MINIGPT4_DATASET}/laion
cp download_laion.sh ${MINIGPT4_DATASET}/laion
```


### Convert the laion and cc_sbu annotation file format to be img2dataset format
```
cd ${MINIGPT4_DATASET}/cc_sbu
python convert_cc_sbu.py

cd ${MINIGPT4_DATASET}/laion
python convert_laion.py
```

### Download the datasets with img2dataset
```
cd ${MINIGPT4_DATASET}/cc_sbu
sh download_cc_sbu.sh
cd ${MINIGPT4_DATASET}/laion
sh download_laion.sh
```


The final dataset structure

```
.
├── ${MINIGPT4_DATASET}
│   ├── cc_sbu
│       ├── convert_cc_sbu.py
│       ├── download_cc_sbu.sh
│       ├── ccs_synthetic_filtered_large.json
│       ├── ccs_synthetic_filtered_large.tsv
│       └── cc_sbu_dataset
│           ├── 00000.tar
│           ├── 00000.parquet
│           ...
│   ├── laion
│       ├── convert_laion.py
│       ├── download_laion.sh
│       ├── laion_synthetic_filtered_large.json
│       ├── laion_synthetic_filtered_large.tsv
│       └── laion_dataset
│           ├── 00000.tar
│           ├── 00000.parquet
│           ...
...   
```


## Set up the dataset configuration files

Then, set up the LAION dataset loading path in 
[here](../minigpt4/configs/datasets/laion/defaults.yaml#L5) at Line 5 as 
${MINIGPT4_DATASET}/laion/laion_dataset/{00000..10488}.tar

and the Conceptual Captoin and SBU datasets loading path in 
[here](../minigpt4/configs/datasets/cc_sbu/defaults.yaml#L5) at Line 5 as 
${MINIGPT4_DATASET}/cc_sbu/cc_sbu_dataset/{00000..01255}.tar



