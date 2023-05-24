## Audio Dataset 

## Stage1: Pretraining 
We mainly use [WavCaps](https://github.com/XinhaoMei/WavCaps) dataset for pre-training. 

### Download 

```Bash
# install git-lfs
sudo apt update
sudo apt-get install git-lfs


git clone https://huggingface.co/datasets/cvssp/WavCaps
cd WavCaps
git lfs pull --include "*" 
```

### Processing

1. Extract zip file
```bash
# merge shards first
zip -s- FILE_NAME.zip -O COMBINED_FILE.zip
unzip COMBINED_FILE.zip
```

2. Processing
Extract raw audio data
```bash
unzip COMBINED_FILE.zip -d /target/dir
```

Create json files (annotations) for each example. Before processing, modify `dataset/audio/process.py` to set data and json path. 
```bash
python3 --dataset test --data_dir /path/to/data --json_path /path/to/json
```


3. Pack with tar
```bash
python3 dataset/audio/make_tar.py --input /path/to/data --output /path/to/web_dataset \
    --dataclass none --filename filename --num_element 500
```

To view tar file 
```
tar tf filename.tar | sed 10q
```

**To setup in one line:**
```bash
# DATASET=soundbible bbc audioset freesound
DATASET=soundbible bash dataset/audio/setup.sh
```
