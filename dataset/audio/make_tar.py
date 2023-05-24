import argparse
import librosa
import json
import webdataset as wds
import numpy as np
import os
import random
import io
from glob import glob
from itertools import islice
import scipy.signal as sps
import soundfile as sf
from tqdm import tqdm
import tarfile


def tardir(
    file_path, tar_name, n_entry_each, audio_ext=".flac", text_ext=".json", shuffle=True, start_idx=0, delete_file=False
):
    """
    This function create the tars that includes the audio and text files in the same folder
    @param file_path      | string  | the path where audio and text files located
    @param tar_name       | string  | the tar name
    @param n_entry_each   | int     | how many pairs of (audio, text) will be in a tar
    @param audio_ext      | string  | the extension of the audio
    @param text_ext       | string  | the extension of the text
    @param shuffle        | boolean | True to shuffle the file sequence before packing up
    @param start_idx      | int     | the start index of the tar
    @param delete_file    | boolean | True to delete the audio and text files after packing up
    """
    filelist = glob(file_path+'/*'+audio_ext)

    if shuffle:
        random.shuffle(filelist)
    count = 0
    n_split = len(filelist) // n_entry_each
    if n_split * n_entry_each != len(filelist):
        n_split += 1
    size_dict = {
        os.path.basename(tar_name) + "{:06d}".format(i) + ".tar": n_entry_each
        for i in range(n_split)
    }
    if n_split * n_entry_each != len(filelist):
        size_dict[os.path.basename(tar_name) + "{:06d}".format(n_split - 1) + ".tar"] = (
            len(filelist) - (n_split - 1) * n_entry_each
        )
    for i in tqdm(range(start_idx, n_split + start_idx), desc='Creating .tar file:'):
        with tarfile.open(tar_name + "{:06d}".format(i) + ".tar", "w") as tar_handle:
            for j in range(count, len(filelist)):
                audio = filelist[j]
                basename = ".".join(audio.split(".")[:-1])
                text_file_path = os.path.join(file_path, basename + text_ext)
                audio_file_path = os.path.join(file_path, audio)
                tar_handle.add(audio_file_path)
                tar_handle.add(text_file_path)
                if delete_file:
                    os.remove(audio_file_path)
                    os.remove(text_file_path)
                if (j + 1) % n_entry_each == 0:
                    count = j + 1
                    break
        tar_handle.close()
    # Serializing json
    json_object = json.dumps(size_dict, indent=4)
    # Writing to sample.json
    with open(os.path.join(os.path.dirname(tar_name), "sizes.json"), "w") as outfile:
        outfile.write(json_object)
    return size_dict

def packup(input, output, filename, dataclass='all', num_element=512, start_idx=0, delete_file=False):
    if not os.path.exists(os.path.join(input, dataclass)):
        print(
            "Dataclass {} does not exist, this folder does not exist. Skipping it.".format(
                dataclass
            )
        )
        return
    if os.path.exists(os.path.join(output, dataclass)):
        tardir(
            os.path.join(input, dataclass),
            os.path.join(output, dataclass, filename),
            num_element,
            start_idx=start_idx,
            delete_file=delete_file,
        )
    else:
        os.makedirs(os.path.join(output, dataclass))
        tardir(
            os.path.join(input, dataclass),
            os.path.join(output, dataclass, filename),
            num_element,
            start_idx=start_idx,
            delete_file=delete_file,
        )
    return

def load_from_tar(
    file_path,
    file_path_type="local",
    audio_ext="flac",
    text_ext="json",
    samplerate=32000,
    mono=True,
    max_len=1000000,
    dtype="float64",
    res_type="kaiser_best",
):
    """
    This function load the tar files to 3 entry tuple (audios, texts, names) accordingly
    @param file_path      | string  | the path where audio and text files located
    @param file_path_type | string  | this is meant to control the prefix of the address in case people forget to include it
                                      if file_path_type is "local" and 'file:\\' is not shown as a prefix, it will be added automatically
    @param audio_ext      | string  | the extension of the audio
    @param text_ext       | string  | the extension of the text
    @param samplerate     | int     | the sample rate of the audio
    @param mono           | boolean | if the audio is in mono channel
    @param max_len        | int     | max len of the audio, if exceed, will random crop; elif deficit, will pad
    @param dtype          | string  | the type of the dtype of the audio sample representation
    @param res_type       | string  | the resample method
    """
    if file_path_type == "local" and ("file:\\" not in file_path):
        file_path = "file:\\" + file_path
    dataset = wds.WebDataset(file_path)
    audios = []
    texts = []
    names = []
    for sample in dataset:
        for key, value in sample.items():
            if key == audio_ext:
                audio_data, orig_sr = sf.read(io.BytesIO(value))
                if samplerate is not None:
                    audio_data = librosa.resample(
                        audio_data,
                        orig_sr=orig_sr,
                        target_sr=samplerate,
                        res_type=res_type,
                    )
                if len(audio_data) > max_len:
                    overflow = len(audio_data) - max_len
                    idx = np.random.randint(0, overflow + 1)
                    if np.random.rand() > 0.5:
                        audio_data = audio_data[idx : idx + max_len]
                    else:
                        audio_data = audio_data[
                            len(audio_data)
                            + 1
                            - idx
                            - max_len : len(audio_data)
                            + 1
                            - idx
                        ]
                else:
                    audio_data = np.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        constant_values=0,
                    )
                if mono:
                    audio_data = librosa.to_mono(audio_data)
                audios.append((audio_data, samplerate))
            elif key == text_ext:
                texts.append(value)
            elif key == "__key__":
                names.append(value)
    return audios, texts, names


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="input folder, expecting subdirectory like train, valid or test",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output, generating tar files at output/dataclass/filename_{}.tar",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="",
        help="the filename of the tar, generating tar files at output/dataclass/filename_{}.tar",
    )
    parser.add_argument(
        "--dataclass", type=str, default="all", help="train or test or valid or all"
    )
    parser.add_argument(
        "--num_element", type=int, default=512, help="pairs of (audio, text) to be included in a single tar"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="start index of the tar"
    )
    parser.add_argument(
        "--delete_file", action='store_true', help="delete the input file when making tars"
    )
    args = parser.parse_args()


    if args.dataclass == "all":
        for x in ["train", "valid", "test"]:
            packup(args.input, args.output,  args.filename,  x,  args.num_element,  args.start_idx,  args.delete_file)
    elif args.dataclass == "none":
        os.makedirs(args.output, exist_ok=True)
        tardir(
            args.input,
            os.path.join(args.output, args.filename),
            args.num_element,
            start_idx=args.start_idx,
            delete_file=args.delete_file,
        )
    else:  # if dataclass is in other name
        packup(args.input, args.output,  args.filename,  args.dataclass,  args.num_element,  args.start_idx,  args.delete_file)