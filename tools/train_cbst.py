import argparse
import copy
import os
import os.path as osp
import time

import numpy as np
import mmcv
from mmcv.runner.utils import set_random_seed
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from torch import distributed
from torch._C import device
from daseg.datasets.builder import build_dataloader

from daseg.utils import get_root_logger
from daseg.models import build_train_model
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataset
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus',
                            type=int,
                            help='number of gpus to use '
                            '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids',
                            type=int,
                            nargs='+',
                            help='ids of gpus to use '
                            '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger()

    meta = {}
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_train_model(cfg.model,
                              train_cfg=cfg.train_cfg,
                              test_cfg=cfg.test_cfg)
    num_classes = cfg.num_classes
    source_dataset = build_dataset(cfg.data.source_dataset.train)
    target_dataset = build_dataset(cfg.data.target_dataset.train)
    target_pseudo_label_path = cfg.data.target_pseudo_path
    mmcv.mkdir_or_exist(target_pseudo_label_path)

    num_rounds = cfg.get('num_rounds', 4)
    epoch_per_round = cfg.get('epoch_per_round', 2)

    for round_idx in range(num_rounds):
        source_test_dataset = build_dataset(cfg.data.source_dataset.test)
        test_dataloader = build_dataloader(source_test_dataset,
                                           samples_per_gpu=1,
                                           workers_per_gpu=2,
                                           dist=distributed,
                                           shuffle=False,
                                           persistent_workers=False)
        test_model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(
            test_model,
            test_dataloader,
            show=False,
        )

        conf_dict = {k: [] for k in range()}
        pred_cls_num = np.zeros(num_classes)

        for idx_cls in range(num_classes):
            idx = results == idx_cls


if __name__ == '__main__':
    main()
