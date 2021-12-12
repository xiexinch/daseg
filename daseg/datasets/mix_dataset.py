import random

from torch.utils.data import Dataset
from .builder import DATASETS
from mmseg.datasets import build_dataset as build_mmseg_dataset
from mmgen.datasets import build_dataset as build_mmgen_dataset


@DATASETS.register_module()
class MixDataset(Dataset):
    def __init__(self, source_cfg: dict, target_cfg: dict):
        self.source_dataset = build_mmseg_dataset(source_cfg)
        self.target_dataset = build_mmseg_dataset(target_cfg)

    def __len__(self):
        return self.target_dataset.__len__()

    def __getitem__(self, idx: int):
        source_img_metas, source_img, source_gt_mask = self.source_dataset[
            random.randint(0, len(self.source_dataset))]
        target_img_metas, target_img, _ = self.target_dataset[idx]
        return dict(source_img_metas=source_img_metas,
                    source_img=source_img,
                    source_gt_mask=source_gt_mask,
                    target_img_metas=target_img_metas,
                    target_img=target_img)

    def __repr__(self):
        source_dataset_name = self.source_dataset.__class__
        target_dataset_name = self.target_dataset.__class__
        source_img_nums = self.source_dataset.__len__()
        target_img_nums = self.target_dataset.__len__()
        return (
            f'source_dataset: {source_dataset_name}, total {source_img_nums} images;'
            f'target_dataset: {target_dataset_name}, total {target_img_nums} images.'
        )
