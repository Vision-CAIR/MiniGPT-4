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

def cn_clip_inference(model, preprocess, device, img, caption):
    image = preprocess(img).unsqueeze(0).to(device)
    text = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True) 
        text_features /= text_features.norm(dim=-1, keepdim=True)    
        sims = (image_features @ text_features.t()).item()
    return sims

def model_init():
    cn_clip_ckpt = '/root/checkpoints/cn_clip/'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root=cn_clip_ckpt)
    model.eval()
    return model, preprocess, device

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

def load_file_name():
    file_to_name = load_jsonl('/workspace/file_to_name_total.jsonl')
    name_to_file = load_jsonl('/workspace/name_to_file_total.jsonl')

    # generate name_to_file according to file_to_name
    for key,values in file_to_name.items():
        if 'open_img_json' in values[0]:
            value_list = ['zero_'+extract_zero(value) for value in values]
        elif 'wukong' in values[0]:
            value_list = ['wukong_'+re.findall('\d+',value)[1] for value in values]

        for value in value_list:
            if value not in name_to_file.keys():
                name_to_file[value] = list()
            if key not in name_to_file[value]:
                name_to_file[value].append(key)
    return file_to_name, name_to_file

def save_data(new_jsonl_path, wk_i, new_data):
    with open(new_jsonl_path + f'/putput_wukong_100m_{wk_i}.csv.json', "w", encoding="UTF-8") as fout:
        fout.write("{}".format(json.dumps(new_data, ensure_ascii=False)))

if __name__ == '__main__':

    model, preprocess, device = model_init()

    file_to_name, name_to_file = load_file_name()

    multimodal_image_path = '/mnt/pfs-guan-ssai/nlu/ark/ark_share/pretrain/data/multimodal-100k-batch/'
    raw_jsonl_path = '/mnt/pfs-guan-ssai/nlu/dingyifeng/data/wukong_json_local'
    new_jsonl_path = '/mnt/pfs-guan-ssai/nlu/wanghanzi/data/wukong_json_local'

    wk_i = 46
    # need to adjust
    file_path = new_jsonl_path + f'/putput_wukong_100m_{wk_i}.csv.json'
    data  = list()
    with open(file_path, "r", encoding="UTF-8") as fin:
        for line in fin:
            line = line.strip()
            temp = json.loads(line)
            data.append(temp)
    new_data = data.copy()
    
    path_no_exist = 0
    for index_id in tqdm(range(len(data[0]))):
        tmp = data[0][index_id]
        caption, image_name, image_path = tmp['caption'], tmp['image_name'], tmp['local_path']
        img = None

        if 'clipscore' in tmp.keys():
            continue

        for path_index in name_to_file[f'wukong_{wk_i}']:
            image_path = os.path.join(multimodal_image_path + path_index ,tmp['image_name'])
            try:
                img = Image.open(image_path)
            except:
                continue

        if img is not None:
            tmp['local_path'] = image_path
            tmp['resolution'] = [img.size[1],img.size[0]] # height,width
            try:
                tmp['clipscore'] = cn_clip_inference(model, preprocess, device, img, caption)
            except Exception as e:
                print('clip inference error:', e)
                tmp['clipscore'] = 0
            new_data[0][index_id] = tmp
        else:
            print('image path not exist: ', tmp['image_name'])
            path_no_exist += 1

        if index_id % 1000 == 0:
            save_data(new_jsonl_path, wk_i, new_data[0])

    save_data(new_jsonl_path, wk_i, new_data[0])
    
    print('file_path:',file_path,'\ntotal pairs:',len(new_data[0]),'\nimage no exist:',path_no_exist)
