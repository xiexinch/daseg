import os.path as osp
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmseg.datasets import CustomDataset
from mmseg.datasets import DATASETS as MMSEG_DATASETS
from daseg.datasets import (DATASETS, build_dataset, MixDataset, RepeatDataset,
                            ConcatDataset)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(128, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(128, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


@MMSEG_DATASETS.register_module()
class ToyDataset(object):
    def __init__(self, cnt=0):
        self.cnt = cnt

    def __item__(self, idx):
        return idx

    def __len__(self):
        return 100


def test_build_mix_dataset():
    source_cfg = dict(type='ToyDataset')
    target_cfg = dict(type='ToyDataset')
    dataset = MixDataset(source_cfg=source_cfg, target_cfg=target_cfg)


@patch('mmseg.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmseg.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_dataset_wrapper():
    # CustomDataset.load_annotations = MagicMock()
    # CustomDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset_a = CustomDataset(img_dir=MagicMock(), pipeline=[])
    len_a = 10
    dataset_a.img_infos = MagicMock()
    dataset_a.img_infos.__len__.return_value = len_a
    dataset_b = CustomDataset(img_dir=MagicMock(), pipeline=[])
    len_b = 20
    dataset_b.img_infos = MagicMock()
    dataset_b.img_infos.__len__.return_value = len_b

    concat_dataset = ConcatDataset([dataset_a, dataset_b])
    assert concat_dataset[5] == 5
    assert concat_dataset[25] == 15
    assert len(concat_dataset) == len(dataset_a) + len(dataset_b)

    repeat_dataset = RepeatDataset(dataset_a, 10)
    assert repeat_dataset[5] == 5
    assert repeat_dataset[15] == 5
    assert repeat_dataset[27] == 7
    assert len(repeat_dataset) == 10 * len(dataset_a)
