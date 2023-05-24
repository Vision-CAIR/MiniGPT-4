#!/bin/bash
# TO setup dataset, run 
# `DATASET=soundbible bash setup.sh`


DATA_ROOT=/mnt/bn/zilongdata-hl/dataset/wavcaps
DATASET=${DATASET-:soundbible}

if [[ $DATASET == soundbible ]]; then
    DATA_FOLDER=SoundBible
    NUM_ELEMENT=2000
elif [[ $DATASET == bbc ]]; then
    DATA_FOLDER=BBC_Sound_Effects
    NUM_ELEMENT=500
elif [[ $DATASET == audioset ]]; then
    DATA_FOLDER=AudioSet_SL
    NUM_ELEMENT=2000
elif [[ $DATASET == freesound ]]; then
    DATA_FOLDER=FreeSound
    NUM_ELEMENT=500
else
    echo "${DATASET} not found!"
    exit
fi 

CODE_PATH=$(pwd)

# Merge zip files 
cd ${DATA_ROOT}/WavCaps/Zip_files/${DATA_FOLDER}
zip -s- ${DATA_FOLDER}.zip -O ${DATA_FOLDER}_combined.zip

# Extract zip file
cd ${DATA_ROOT}
unzip ${DATA_ROOT}/WavCaps/Zip_files/${DATA_FOLDER}/${DATA_FOLDER}_combined.zip -d raw_datasets/${DATA_FOLDER}
mv raw_datasets/${DATA_FOLDER}/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/${DATA_FOLDER}_flac raw_datasets/${DATA_FOLDER}_flac
rm -rf raw_datasets/${DATA_FOLDER}

# Process raw data to create json annotation files
cd $CODE_PATH
python3 dataset/audio/process.py --data_root ${DATA_ROOT} --dataset ${DATASET}

# Pack up tar files 
python3 dataset/audio/make_tar.py --input ${DATA_ROOT}/raw_datasets/${DATA_FOLDER}_flac \
    --output ${DATA_ROOT}/web_datasets/${DATA_FOLDER} \
    --dataclass none --filename ${DATA_FOLDER} --num_element ${NUM_ELEMENT}
