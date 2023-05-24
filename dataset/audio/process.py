"""
Code for processing WavCaps dataset.
"""
import argparse

import os
from tqdm import tqdm
import glob
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='bbc', choices=['bbc', 'audioset', 'soundbible', 'freesound', 'test'])
parser.add_argument(
    "--data_root",
    type=str,
    default="/mnt/bn/zilongdata-hl/dataset/wavcaps"
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="/mnt/bn/zilongdata-hl/dataset/wavcaps/raw_datasets/test/"
)
parser.add_argument(
    "--json_path",
    type=str,
    default="/mnt/bn/zilongdata-hl/dataset/wavcaps/WavCaps/json_files/test.json",
)
args = parser.parse_args()

DATA_DIRS = {
    "bbc": "raw_datasets/BBC_Sound_Effects_flac/",
    "audioset": "raw_datasets/AudioSet_SL_flac/",
    "soundbible": "raw_datasets/SoundBible_flac/",
    "freesound": "raw_datasets/FreeSound_flac/",
}

JSON_PATHS = {
    "bbc": "WavCaps/json_files/BBC_Sound_Effects/bbc_final.json",
    "audioset": "WavCaps/json_files/AudioSet_SL/as_final.json",
    "soundbible": "WavCaps/json_files/SoundBible/sb_final.json",
    "freesound": "WavCaps/json_files/FreeSound/fsd_final.json",
}



def load_audioset_json(fname):
    """A sample example:
    {   
        'id': 'Yb0RFKhbpFJA.wav',
        'caption': 'Wind and a man speaking are heard, accompanied by buzzing and ticking.',
        'audio': 'wav_path',
        'duration': 10.0
    }
    """
    with open(fname) as f:
        data = json.load(f)

    for sample in data['data']:
        yield sample['id'].split('.')[0], sample['caption'], sample


def load_soundbible_json(fname):
    """A sample example:
    {   
        'title': 'Airplane Landing Airport',
        'description': 'Large commercial airplane landing at an airport runway.',
        'author': 'Daniel Simion',
        'href': '2219-Airplane-Landing-Airport.html',
        'caption': 'An airplane is landing.',
        'id': '2219',
        'duration': 14.1424375,
        'audio': 'wav_path',
        'download_link': 'http://soundbible.com/grab.php?id=2219&type=wav'
    }
    """
    with open(fname) as f:
        data = json.load(f)
    
    for sample in data['data']:
        yield sample['id'], sample['caption'], sample


def load_freesound_json(fname):
    """A sample example:
    {   'id': '180913',
        'file_name': 'UK Mello.wav',
        'href': '/people/Tempouser/sounds/180913/',
        'tags': ['Standard', 'ringtone', 'basic', 'traditional'],
        'description': 'Standard traditional basic ringtone, in mello tone.',
        'author': 'Tempouser',
        'duration': 3.204375,
        'download_link': 'https://freesound.org/people/Tempouser/sounds/180913/download/180913__tempouser__uk-mello.wav',
        'caption': 'A traditional ringtone is playing.',
        'audio': 'wav_path'
    }
    """
    with open(fname) as f:
        data = json.load(f)
    
    for sample in data['data']:
        yield sample['id'], sample['caption'], sample


def load_bbc_json(fname):
    """A sample example:
    {
        'description': "Timber & Wood - Rip saw, carpenters' workshop.",
        'category': "['Machines']",
        'caption': "Someone is using a rip saw in a carpenter's workshop.",
        'id': '07066104',
        'duration': 138.36,
        'audio': 'wav_path',
        'download_link': 'https://sound-effects-media.bbcrewind.co.uk/zip/07066104.wav.zip'
    }
    """
    with open(fname) as f:
        data = json.load(f)
    
    for sample in data['data']:
        yield sample['id'], sample['caption'], sample


def load_test_json(fname):
    """Using SoundBible as a text example."""
    with open(fname) as f:
        data = json.load(f)
    
    for sample in data['data']:
        yield sample['id'], sample['caption'], sample

if __name__ == '__main__':
    if args.dataset in DATA_DIRS:
        data_dir = os.path.join(args.data_root, DATA_DIRS[args.dataset])
        json_path = os.path.join(args.data_root, JSON_PATHS[args.dataset])
    else:
        data_dir = args.data_dir
        json_path = args.json_path

    file_list = glob.glob(f'{data_dir}/*.flac')
    for data_id, unsed_caption, meta_data in tqdm(list(globals()[f'load_{args.dataset}_json'](json_path))):
        file_name = os.path.join(data_dir, data_id + '.flac')
        json_save_path = os.path.join(data_dir, data_id + '.json')
        # text_save_path = os.path.join(data_dir, data_id + '.text')
        file_list.remove(file_name)

        assert os.path.exists(file_name), f'{file_name} does not exist!'
        with open(json_save_path, 'w') as f:
            json.dump(meta_data, f)
    
    if len(file_list) > 0:
        # import pdb; pdb.set_trace()
        for f in file_list:
            os.remove(f)
  

    # file_list = glob.glob(f'{data_dir}/*.flac')
    # for file_path in file_list:
    #     audio_json_save_path = file_path.replace('.flac', '.json')
    #     audio_text_save_path = file_path.replace('.flac', '.text')
