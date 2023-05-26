import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.audio_base_dataset_builder import AudioBaseDatasetBuilder
from minigpt4.datasets.datasets.audio_caption import GenericAudioDataset



class GenericAudioBuilder(AudioBaseDatasetBuilder):
    train_dataset_cls = GenericAudioDataset

    def _download_ann(self):
        pass

    def _download_aud(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            audio_processor=self.audio_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
    

@registry.register_builder("bbc")
class BBCBuilder(GenericAudioBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/bbc/defaults.yaml"}


@registry.register_builder("audioset")
class AudioSetBuilder(GenericAudioBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/audioset/defaults.yaml"}


@registry.register_builder("soundbible")
class SoundBibleBuilder(GenericAudioBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/soundbible/defaults.yaml"}


@registry.register_builder("freesound")
class FreeSoundBuilder(GenericAudioBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/freesound/defaults.yaml"}
