## Download the filtered Conceptual Captions, SBU, LAION datasets

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



