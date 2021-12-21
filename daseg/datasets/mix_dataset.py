import random
import json
import os.path as osp
import torch

from torch.utils.data import Dataset
from .builder import DATASETS
from mmseg.datasets import build_dataset as build_mmseg_dataset, build_dataloader as build_mmseg_dataloader
from mmgen.datasets import build_dataset as build_mmgen_dataset

from mmgen.datasets.pipelines import Compose


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(overall_class_stats.items(),
                           key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class MixDataset(Dataset):
    def __init__(self, source_cfg: dict, target_cfg: dict, pipeline):
        self.source_dataset = build_mmseg_dataset(source_cfg)
        self.target_dataset = build_mmseg_dataset(target_cfg)
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.target_dataset) * len(self.source_dataset)

    def __getitem__(self, idx: int):
        source_data = self.source_dataset[idx // len(self.target_dataset)]
        target_data = self.target_dataset[idx]

        return self.pipeline(
            dict(**source_data,
                 target_img_metas=target_data['img_metas'],
                 target_img=target_data['img']))

    def __repr__(self):
        source_dataset_name = self.source_dataset.__class__
        target_dataset_name = self.target_dataset.__class__
        return (
            f'source_dataset: {source_dataset_name}, total {len(self.source_dataset)} images;'
            f'target_dataset: {target_dataset_name}, total {len(self.target_dataset)} images.'
        )
