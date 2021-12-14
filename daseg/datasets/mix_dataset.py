import random

from torch.utils.data import Dataset
from .builder import DATASETS
from mmseg.datasets import build_dataset as build_mmseg_dataset, build_dataloader as build_mmseg_dataloader
from mmgen.datasets import build_dataset as build_mmgen_dataset

from mmgen.datasets.pipelines import Compose


@DATASETS.register_module()
class MixDataset(Dataset):
    def __init__(self, source_cfg: dict, target_cfg: dict, pipeline):
        self.source_dataset = build_mmseg_dataset(source_cfg)
        self.target_dataset = build_mmseg_dataset(target_cfg)
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return self.target_dataset.__len__()

    def __getitem__(self, idx: int):
        source_data = self.source_dataset[random.randint(
            0,
            len(self.source_dataset) - 1)]
        source_img_metas = source_data['img_metas']
        source_img = source_data['img']
        source_gt_mask = source_data['gt_semantic_seg']

        target_data = self.target_dataset[idx]
        target_img_metas = target_data['img_metas']
        target_img = target_data['img']

        return self.pipeline(
            dict(source_img_metas=source_img_metas,
                 source_img=source_img,
                 source_gt_mask=source_gt_mask,
                 target_img_metas=target_img_metas,
                 target_img=target_img))

    def __repr__(self):
        source_dataset_name = self.source_dataset.__class__
        target_dataset_name = self.target_dataset.__class__
        source_img_nums = self.source_dataset.__len__()
        target_img_nums = self.target_dataset.__len__()
        return (
            f'source_dataset: {source_dataset_name}, total {source_img_nums} images;'
            f'target_dataset: {target_dataset_name}, total {target_img_nums} images.'
        )
