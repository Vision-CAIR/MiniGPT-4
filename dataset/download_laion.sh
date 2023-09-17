#!/bin/bash

img2dataset --url_list laion_synthetic_filtered_large.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder laion_dataset --processes_count 16 --thread_count 128 --image_size 224 \
             --enable_wandb True
