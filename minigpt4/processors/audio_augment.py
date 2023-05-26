#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
    Implemenation of SpecAugment++,
    Adapated from Qiuqiang Kong's trochlibrosa:
    https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/augmentation.py
"""

import torch
import torch.nn as nn


class DropStripes:

    def __init__(self, dim, drop_width, stripes_num):
        """ Drop stripes.
        args:
            dim: int, dimension along which to drop
            drop_width: int, maximum width of stripes to drop
            stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def __call__(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4
        batch_size = input.shape[0]
        total_width = input.shape[self.dim]

        for n in range(batch_size):
            self.transform_slice(input[n], total_width)

        return input

    def transform_slice(self, e, total_width):
        """ e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn: bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn: bgn + distance] = 0


class MixStripes:

    def __init__(self, dim, mix_width, stripes_num):
        """ Mix stripes
        args:
            dim: int, dimension along which to mix
            mix_width: int, maximum width of stripes to mix
            stripes_num: int, how many stripes to mix
        """

        super(MixStripes, self).__init__()

        assert dim in [2, 3]

        self.dim = dim
        self.mix_width = mix_width
        self.stripes_num = stripes_num

    def __call__(self, input):
        """input: (batch_size, channel, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        batch_size = input.shape[0]
        total_width = input.shape[self.dim]

        rand_sample = input[torch.randperm(batch_size)]
        for i in range(batch_size):
            self.transform_slice(input[i], rand_sample[i], total_width)
        return input

    def transform_slice(self, input, random_sample, total_width):

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.mix_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                input[:, bgn: bgn + distance, :] = 0.5 * input[:, bgn: bgn + distance, :] + \
                                                   0.5 * random_sample[:, bgn: bgn + distance, :]
            elif self.dim == 3:
                input[:, :, bgn: bgn + distance] = 0.5 * input[:, :, bgn: bgn + distance] + \
                                                   0.5 * random_sample[:, :, bgn: bgn + distance]


class CutStripes:

    def __init__(self, dim, cut_width, stripes_num):
        """ Cutting stripes with another randomly selected sample in mini-batch.
        args:
            dim: int, dimension along which to cut
            cut_width: int, maximum width of stripes to cut
            stripes_num: int, how many stripes to cut
        """

        super(CutStripes, self).__init__()

        assert dim in [2, 3]

        self.dim = dim
        self.cut_width = cut_width
        self.stripes_num = stripes_num

    def __call__(self, input):
        """input: (batch_size, channel, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        batch_size = input.shape[0]
        total_width = input.shape[self.dim]

        rand_sample = input[torch.randperm(batch_size)]
        for i in range(batch_size):
            self.transform_slice(input[i], rand_sample[i], total_width)
        return input

    def transform_slice(self, input, random_sample, total_width):

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.cut_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                input[:, bgn: bgn + distance, :] = random_sample[:, bgn: bgn + distance, :]
            elif self.dim == 3:
                input[:, :, bgn: bgn + distance] = random_sample[:, :, bgn: bgn + distance]


class SpecAugmentation:

    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num,
                 mask_type='mixture'):
        """Spec augmetation and SpecAugment++.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        [ref] Wang H, Zou Y, Wang W., 2021. SpecAugment++: A Hidden Space
        Data Augmentation Method for Acoustic Scene Classification. arXiv
        preprint arXiv:2103.16858.

        Args:
            time_drop_width: int
            time_stripes_num: int
            freq_drop_width: int
            freq_stripes_num: int
            mask_type: str, mask type in SpecAugment++ (zero_value, mixture, cutting)
        """

        super(SpecAugmentation, self).__init__()

        if mask_type == 'zero_value':
            self.time_augmentator = DropStripes(dim=2, drop_width=time_drop_width,
                                                stripes_num=time_stripes_num)
            self.freq_augmentator = DropStripes(dim=3, drop_width=freq_drop_width,
                                                stripes_num=freq_stripes_num)
        elif mask_type == 'mixture':
            self.time_augmentator = MixStripes(dim=2, mix_width=time_drop_width,
                                               stripes_num=time_stripes_num)
            self.freq_augmentator = MixStripes(dim=3, mix_width=freq_drop_width,
                                               stripes_num=freq_stripes_num)
        elif mask_type == 'cutting':
            self.time_augmentator = CutStripes(dim=2, cut_width=time_drop_width,
                                               stripes_num=time_stripes_num)
            self.freq_augmentator = CutStripes(dim=3, cut_width=freq_drop_width,
                                               stripes_num=freq_stripes_num)
        else:
            raise NameError('No such mask type in SpecAugment++')

    def __call__(self, inputs):
        # x should be in size [batch_size, channel, time_steps, freq_bins]
        x = self.time_augmentator(inputs)
        x = self.freq_augmentator(x)
        return x
