import argparse
import copy
import os.path as osp
import time
import math
import shutil
from PIL import Image
import json

import numpy as np
import mmcv
from mmcv.runner.utils import set_random_seed
import torch
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, HOOKS, build_optimizer,
                         build_runner, load_checkpoint)
from torch.utils.data import ConcatDataset

from daseg.utils import get_root_logger
from daseg.models import build_train_model
from mmseg.apis import single_gpu_forward
from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel
from mmcv.utils import build_from_cfg


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
    logger = get_root_logger(log_file)

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
    load_checkpoint(model, cfg.model_checkpoint)

    num_classes = cfg.num_classes
    source_dataset = build_dataset(cfg.data.source_dataset.train)
    target_dataset = build_dataset(cfg.data.target_dataset.pretrain)
    target_pseudo_label_path = cfg.data.target_pseudo_path
    mmcv.mkdir_or_exist(target_pseudo_label_path)

    num_rounds = cfg.get('num_rounds', 4)

    # source_test_dataset = build_dataset(cfg.data.source_dataset.test)
    target_train_dataloader = build_dataloader(target_dataset,
                                               samples_per_gpu=1,
                                               workers_per_gpu=2,
                                               dist=False,
                                               shuffle=False,
                                               persistent_workers=False)
    target_portion = cfg.target_portion
    model = MMDataParallel(model, device_ids=[0])
    # pseudo-label list: delete after training
    pseudo_label_list = []
    train_model = copy.deepcopy(model)
    test_model = copy.deepcopy(model)
    optimizer = build_optimizer(train_model, cfg.optimizer)
    for round_idx in range(num_rounds):

        # runner config
        lr_config = copy.deepcopy(cfg.lr_config)
        optimizer_config = copy.deepcopy(cfg.optimizer_config)
        checkpoint_config = copy.deepcopy(cfg.checkpoint_config)
        log_config = copy.deepcopy(cfg.log_config)
        momentum_config = copy.deepcopy(cfg.get('momentum_config', None))
        # numpy array

        labels, confs = single_gpu_forward(test_model, target_train_dataloader,
                                           cfg.temp_dirs)

        if osp.exists(cfg.conf_dict_path) and osp.exists(
                cfg.pred_cls_num_path):
            print()
            logger.info(f'load conf_dict from {cfg.conf_dict_path}')
            # conf_dict = np.load(cfg.conf_dict_path, allow_pickle=True)
            conf_dict = mmcv.load(cfg.conf_dict_path)
            pred_cls_num = np.load(cfg.pred_cls_num_path, allow_pickle=True)
        else:
            conf_dict = {k: [] for k in range(num_classes)}
            # pred_cls_num = np.zeros(num_classes)
            pred_cls_num = torch.zeros(num_classes)
            # get confidence vectors
            print()
            logger.info('get confidence vectors')
            prog_bar = mmcv.ProgressBar(len(labels))
            for i, label_path in enumerate(labels):
                seg_logits = torch.from_numpy(np.load(label_path)[0]).cuda()
                conf = torch.from_numpy(np.load(confs[i])[0]).cuda()
                for idx_cls in range(num_classes):
                    cls_hit_map = seg_logits == idx_cls
                    pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + torch.sum(
                        cls_hit_map)
                    if cls_hit_map.any():
                        # conf_cls_temp = conf[idx_cls][cls_hit_map].astype(
                        #     np.float32)
                        conf_cls_temp = conf[idx_cls][cls_hit_map]
                        len_cls_temp = conf_cls_temp.size().numel()
                        conf_cls = conf_cls_temp[0:len_cls_temp:4]
                        conf_dict[idx_cls].extend(conf_cls.cpu().numpy())
                prog_bar.update()
            mmcv.mkdir_or_exist(cfg.temp_root)
            mmcv.dump(conf_dict, cfg.conf_dict_path)
            # np.save(cfg.conf_dict_path, conf_dict)
            np.save(cfg.pred_cls_num_path, pred_cls_num)

        # kc parameters
        print()
        logger.info('kc parameters')
        cls_size = np.zeros(num_classes, dtype=np.float32)
        for i in range(num_classes):
            cls_size[i] = pred_cls_num[i]
        if osp.exists(cfg.cls_thresh_path):
            cls_thresh = np.load(cfg.cls_thresh_path, allow_pickle=True)
        else:
            cls_thresh = np.ones(num_classes, dtype=np.float32)  # 阈值
            cls_select_size = np.zeros(num_classes, dtype=np.float32)  # ？这是什么
            # class-balance
            for idx_cls in conf_dict.keys():
                if conf_dict[idx_cls] is not None:
                    conf = torch.Tensor(
                        conf_dict[idx_cls]).cuda().sort(descending=True).values
                    # conf_dict[idx_cls].sort(reverse=True)
                    len_cls = len(conf)
                    cls_select_size[int(idx_cls)] = int(
                        math.floor(len_cls * target_portion))
                    len_cls_thresh = int(
                        cls_select_size[int(idx_cls)])  # 取前 q%
                    if len_cls_thresh != 0:
                        # cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh - 1]
                        cls_thresh[int(idx_cls)] = conf[len_cls_thresh -
                                                        1].cpu().numpy()
                        print(cls_thresh[int(idx_cls)])
                    conf_dict[idx_cls] = None
            mmcv.mkdir_or_exist(cfg.temp_root)
            np.save(cfg.cls_thresh_path, cls_thresh)

        # mine_id 是什么意思
        # num_mine_id = len(np.nonzero(cls_size / np.sum(cls_size) < 1e-3)[0])
        # choose the min mine_id
        # id_all = np.argsort(cls_size / np.sum(cls_size))
        # rare_id = id_all[:cfg.rare_cls_nums]
        # mine_id = id_all[:num_mine_id]

        # 逐步增大 target 阈值
        target_portion = min(target_portion + cfg.target_port_step,
                             cfg.max_target_portion)

        # 生成伪标签
        print()
        logger.info('pseudo-label generation')

        loader_indices = target_train_dataloader.batch_sampler
        target_dataroot = cfg.data.target_dataset.train['data_root']
        target_img_dir = cfg.data.target_dataset.train['img_dir']
        target_img_prefix = target_dataroot + target_img_dir
        prog_bar = mmcv.ProgressBar(len(labels))
        for batch_indices, data in zip(loader_indices,
                                       target_train_dataloader):
            for i in batch_indices:
                conf = np.load(confs[i]).squeeze(0)
                weighted_prob = conf.transpose(1, 2, 0) / cls_thresh
                weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob,
                                                              axis=2),
                                                    dtype=np.uint8)
                seg_map_path = data['img_metas'][0].data[0][0][
                    'filename'].replace(target_img_prefix, '').replace(
                        target_train_dataloader.dataset.img_suffix,
                        target_train_dataloader.dataset.seg_map_suffix)
                img_name = seg_map_path.split('/')[-1]
                save_path = target_pseudo_label_path + seg_map_path
                pseudo_label_list.append(save_path)
                save_dir = save_path.replace(img_name, '')
                mmcv.mkdir_or_exist(save_dir)
                Image.fromarray(weighted_pred_trainIDs.astype(
                    np.uint8)).save(save_path)
                prog_bar.update()

        logger.info('pseudo-label generation finished')
        # 删除临时结果
        shutil.rmtree(cfg.temp_dirs)

        target_dataset_cfg = cfg.data.target_dataset.train
        target_dataset_cfg['ann_dir'] = target_pseudo_label_path.replace(
            target_dataset_cfg['data_root'], '')
        target_dataset = build_dataset(target_dataset_cfg)

        mix_dataset = ConcatDataset(
            [RepeatDataset(source_dataset, 2), target_dataset])
        # mix_dataset = ConcatDataset([source_dataset, target_dataset])
        mix_dataloader = build_dataloader(mix_dataset,
                                          samples_per_gpu=4,
                                          workers_per_gpu=2,
                                          dist=False,
                                          shuffle=False,
                                          persistent_workers=False)
        train_model.train()
        runner = build_runner(cfg.runner,
                              default_args=dict(
                                  model=train_model,
                                  batch_processor=None,
                                  optimizer=optimizer,
                                  work_dir=f'{cfg.work_dir}/round_{round_idx}',
                                  logger=logger,
                                  meta=meta))

        # register hooks
        runner.register_training_hooks(lr_config, optimizer_config,
                                       checkpoint_config, log_config,
                                       momentum_config)
        runner.timestamp = timestamp
        # register eval hooks

        val_dataset = build_dataset(cfg.data.target_dataset.test,
                                    dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg),
                             priority='LOW')
        # user-defined hooks
        if cfg.get('custom_hooks', None):
            custom_hooks = cfg.custom_hooks
            assert isinstance(custom_hooks, list), \
                f'custom_hooks expect list type, but got {type(custom_hooks)}'
            for hook_cfg in cfg.custom_hooks:
                assert isinstance(hook_cfg, dict), \
                    'Each item in custom_hooks expects dict type, but got ' \
                    f'{type(hook_cfg)}'
                hook_cfg = hook_cfg.copy()
                priority = hook_cfg.pop('priority', 'NORMAL')
                hook = build_from_cfg(hook_cfg, HOOKS)
                runner.register_hook(hook, priority=priority)
        # if cfg.resume_from:
        #     runner.resume(cfg.resume_from)
        # elif cfg.load_from:
        #     runner.load_checkpoint(cfg.load_from)
        runner.run([mix_dataloader], cfg.workflow)
        train_model = copy.deepcopy(runner.model)
        test_model = copy.deepcopy(runner.model)


if __name__ == '__main__':
    main()
