import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import time
import json
import re
from flagai.auto_model.auto_loader import AutoLoader
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from collections import defaultdict

def load_jsonl(file_path):
    data  = list()
    with open(file_path, "r", encoding="UTF-8") as fin:
        for line in fin:
            line = line.strip()
            temp = json.loads(line)
            data.append(temp)
    return data[0]

def extract_zero(string):
    pattern = r'package_\d+_dir_\d+'
    result = re.search(pattern, string)
    if result:
        extracted_string = result.group()
        return extracted_string
    else:
        print("Pattern not found in the string.")
        return ''

def save_to_json(saved_data, saved_file_name):
    print('Saved File Name:',saved_file_name)
    write_cnt = 0
    print(saved_data)
    with open(saved_file_name, "w", encoding="UTF-8") as fout:
        fout.write("{}\n".format(json.dumps(saved_data, ensure_ascii=False)))

file_to_name = load_jsonl('/workspace/file_to_name.jsonl')
name_to_file = load_jsonl('/workspace/name_to_file.jsonl')

multimodal_image_path = '/mnt/pfs-guan-ssai/nlu/ark/ark_share/pretrain/data/multimodal-100k-batch/'
file_index = os.listdir(multimodal_image_path)

for i in tqdm(range(1212)):
    if  file_index[i] in file_to_name.keys():
        print(file_to_name[file_index[i]])
        continue
    images = os.listdir(multimodal_image_path + file_index[i])
    first = images[0]
    if 'open_img_json' in first:
        values = set([i[:36] for i in images])
        if len(values)>100:
            print('File Name not in the specific type....',file_index[i])
            continue
        value_list = ['zero_'+extract_zero(value) for value in values]
    elif 'wukong' in first:
        values = set([i[:21] for i in images])
        if len(values)>100:
            print('File Name not in the specific type....',file_index[i])
            continue
    	try:
    	    value_list = ['wukong_'+re.findall('\d+',value)[1] for value in values]
    	except:
    	    print('-------------------',values)
    	    continue
    print(file_index[i], '--',list(values))
    file_to_name[file_index[i]] = list(values)
    for value in value_list:
        if value not in name_to_file.keys():
            name_to_file[value] = list()
        name_to_file[value].append(file_index[i])
 
save_to_json(name_to_file, '/workspace/name_to_file_total.jsonl')
save_to_json(file_to_name, '/workspace/file_to_name_total.jsonl')


