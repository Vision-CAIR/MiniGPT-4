import json

from torch.utils.data import Dataset, default_collate
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset


class GenericAudioDataset(BaseDataset):
    def __init__(self, audio_processor, text_processor, location):
        super().__init__(x_processor=audio_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode(wds.torch_audio, handler=wds.warn_and_continue),
            wds.to_tuple("flac", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.x_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "audio": sample[0],
            # [clips_per_video, channel, mel_bins, time_steps]
            "text_input": self.text_processor(sample[1]["caption"]),
        }
