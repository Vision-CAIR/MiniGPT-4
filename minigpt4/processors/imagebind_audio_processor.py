from typing import Union, List

import torch
import torchaudio
from omegaconf import OmegaConf
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler, RandomMultiClipSampler
from torch import Tensor

from imagebind.data.data_utils import waveform2melspec, get_constant_clip_timepoints, \
    get_random_clip_timepoints
from minigpt4.processors.base_processor import BaseProcessor
from torchvision import transforms
from minigpt4.common.registry import registry
from minigpt4.processors.audio_augment import SpecAugmentation


class ImageBindAudioBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, target_sr=None, clip_duration=None, clips_per_video=None,
                 num_mel_bins=None, target_length=None, clip_sample_method="Random"):
        super().__init__()
        self.mean = -4.268 if mean is None else mean
        self.std = 9.138 if std is None else std
        self.target_sr = 16000 if target_sr is None else target_sr
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.clip_sampler = self._construct_clip_sampler(clip_duration, clips_per_video, clip_sample_method)
        self.normalize = transforms.Normalize(self.mean, self.std)

    def _construct_clip_sampler(self, clip_duration, clips_per_video, clip_sample_method):
        if clip_duration is None or clips_per_video is None:
            return None
        if clip_sample_method == "Constant":
            return ConstantClipsPerVideoSampler(
                clip_duration=clip_duration, clips_per_video=clips_per_video
            )
        elif clip_sample_method == "Random":
            return RandomMultiClipSampler(clip_duration=clip_duration, num_clips=clips_per_video)
        else:
            raise NotImplementedError

    def waveform_resample(self, waveform: Tensor, origin_sr: int) -> Tensor:
        return torchaudio.functional.resample(
            waveform, orig_freq=origin_sr, new_freq=self.target_sr
        )

    def clip_sample(self, waveform: Tensor) -> List[Tensor]:
        if self.clip_sampler is None:
            return [waveform]
        elif isinstance(self.clip_sampler, ConstantClipsPerVideoSampler):
            all_clips_timepoints = get_constant_clip_timepoints(self.clip_sampler, waveform.size(1) / self.target_sr)
        elif isinstance(self.clip_sampler, RandomMultiClipSampler):
            all_clips_timepoints = get_random_clip_timepoints(self.clip_sampler, waveform.size(1) / self.target_sr)
        else:
            raise NotImplementedError
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            start_pos = int(clip_timepoints[0] * self.target_sr)
            end_pos = int(clip_timepoints[1] * self.target_sr)
            waveform_clip = waveform[:, start_pos: end_pos]
            all_clips.append(waveform_clip)
        return all_clips

    def waveform_melspec(self, waveforms: Union[List[Tensor], Tensor]) -> List[Tensor]:
        if isinstance(waveforms, Tensor):
            return waveform2melspec(waveforms, self.target_sr, self.num_mel_bins, self.target_length)
        else:
            return [waveform2melspec(waveform, self.target_sr, self.num_mel_bins, self.target_length)
                    for waveform in waveforms]


@registry.register_processor("imagebind_audio_train")
class ImageBindAudioTrainProcessor(ImageBindAudioBaseProcessor):
    def __init__(self, mean=None, std=None, target_sr=None, clip_duration=None, clips_per_video=None,
                 clip_sample_method="Random", num_mel_bins=None, target_length=None, time_drop_width=13,
                 time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2, mask_type='mixture'):
        super().__init__(mean=mean, std=std, target_sr=target_sr,
                         clip_duration=clip_duration, clips_per_video=clips_per_video,
                         num_mel_bins=num_mel_bins, target_length=target_length,
                         clip_sample_method=clip_sample_method)
        self.spec_augment = SpecAugmentation(time_drop_width, time_stripes_num,
                                             freq_drop_width, freq_stripes_num, mask_type)

    def __call__(self, item):
        # item: Tuple[Tensor, int]
        waveform, origin_sr = item[0], item[1]
        waveform = self.waveform_resample(waveform, origin_sr)
        waveform_clips = self.clip_sample(waveform)
        melspec_clips = self.waveform_melspec(waveform_clips)
        normed_melspecs = [self.normalize(clip) for clip in melspec_clips]
        all_clips = torch.stack(normed_melspecs, dim=0)
        # all_clips: [clips_per_video, channel, mel_bins, time_steps]
        # augment: [batch_size, channel, time_steps, freq_bins]
        augmented_clips = self.spec_augment(all_clips.transpose(-2, -1)).transpose(-2, -1)
        return augmented_clips

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        target_sr = cfg.get("target_sr", 16000)
        clip_duration = cfg.get("clip_duration", 2)
        clips_per_video = cfg.get("clips_per_video", 3)
        num_mel_bins = cfg.get("num_mel_bins", 128)
        target_length = cfg.get("target_length", 204)
        time_drop_width = cfg.get("time_drop_width", 13)
        time_stripes_num = cfg.get("time_stripes_num", 2)
        # 13 * 2 / 204 = 12.75% Time Mask
        freq_drop_width = cfg.get("freq_drop_width", 8)
        freq_stripes_num = cfg.get("freq_stripes_num", 2)
        # 8 * 2 / 128 = 12.5% Freq Mask
        mask_type = cfg.get("mask_type", 'mixture')

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            mean=mean, std=std, target_sr=target_sr,
            clip_duration=clip_duration, clips_per_video=clips_per_video,
            num_mel_bins=num_mel_bins, target_length=target_length,
            time_drop_width=time_drop_width, time_stripes_num=time_stripes_num,
            freq_drop_width=freq_drop_width, freq_stripes_num=freq_stripes_num,
            mask_type=mask_type
        )


@registry.register_processor("imagebind_audio_eval")
class ImageBindAudioEvalProcessor(ImageBindAudioBaseProcessor):
    def __init__(self, mean=None, std=None, target_sr=None, clip_duration=None, clips_per_video=None,
                 clip_sample_method="Constant", num_mel_bins=None, target_length=None):
        super().__init__(mean=mean, std=std, target_sr=target_sr,
                         clip_duration=clip_duration, clips_per_video=clips_per_video,
                         num_mel_bins=num_mel_bins, target_length=target_length,
                         clip_sample_method=clip_sample_method)

    def __call__(self, item):
        # item: Tuple[Tensor, int]
        waveform, origin_sr = item[0], item[1]
        waveform = self.waveform_resample(waveform, origin_sr)
        waveform_clips = self.clip_sample(waveform)
        melspec_clips = self.waveform_melspec(waveform_clips)
        normed_melspecs = [self.normalize(clip) for clip in melspec_clips]
        all_clips = torch.stack(normed_melspecs, dim=0)
        # all_clips: [clips_per_video, channel, mel_bins, time_steps]
        return all_clips

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        target_sr = cfg.get("target_sr", 16000)
        clip_duration = cfg.get("clip_duration", 2)
        clips_per_video = cfg.get("clips_per_video", 3)
        num_mel_bins = cfg.get("num_mel_bins", 128)
        target_length = cfg.get("target_length", 204)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            mean=mean, std=std, target_sr=target_sr,
            clip_duration=clip_duration, clips_per_video=clips_per_video,
            num_mel_bins=num_mel_bins, target_length=target_length
        )

