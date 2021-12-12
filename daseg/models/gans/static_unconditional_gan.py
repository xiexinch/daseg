from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from mmseg.models import build_segmentor, build_loss as build_seg_loss
from mmgen.models import (set_requires_grad, BaseGAN, build_module as
                          build_gen_module)

from daseg.models.builder import MODELS, build_module


@MODELS.register_module()
class CustomStaticUnconditionalGAN(BaseGAN):
    def __init__(self,
                 segmentor,
                 discriminator,
                 seg_loss,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 seg_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self._segmentor_cfg = deepcopy(segmentor)
        self.segmentor = build_segmentor(segmentor)

        # support no discriminator in testing
        if discriminator is not None:
            self.discriminator = build_gen_module(discriminator)
        else:
            self.discriminator = None

        # support no gan_loss in testing
        if gan_loss is not None:
            self.gan_loss = build_gen_module(gan_loss)
        else:
            self.gan_loss = None

        if disc_auxiliary_loss:
            self.disc_auxiliary_losses = build_gen_module(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            self.disc_auxiliary_losses = None

        self.seg_loss = build_seg_loss(seg_loss)

        if seg_auxiliary_loss:
            self.seg_auxiliary_losses = build_seg_loss(seg_auxiliary_loss)
            if not isinstance(self.seg_auxiliary_losses, nn.ModuleList):
                self.seg_auxiliary_losses = nn.ModuleList(
                    [self.seg_auxiliary_losses])
        else:
            self.seg_auxiliary_losses = None

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.segmentor_ema = deepcopy(self.segmentor)

        # self.sourc_img_metas_key = 'source_img_metas'
        # self.source_img_key = 'source_img'
        # self.source_gt_mask_key = 'source_gt_mask'
        # self.target_img_metas_key = 'target_img_metas'
        # self.target_img_key = 'target_img'

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg('use_ema', False)
        # TODO: finish ema part

    def train_step(self,
                   data_batch: dict,
                   optimizer: dict,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        # get source images
        source_imgs = data_batch['source_img'].data
        source_gt_masks = data_batch['source_gt_mask'].data
        target_imgs = data_batch['target_img'].data
        # print(data_batch.keys())
        # source_imgs, source_gt_masks, target_imgs = data_batch

        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = source_imgs.shape[0]

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # 训练 Discriminator
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()
        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            source_seg_logits = self.segmentor.forward_dummy(source_imgs)
            target_seg_logits = self.segmentor.forward_dummy(target_imgs)

        # disc pred for target imgs and source imgs
        disc_pred_target = self.discriminator(target_seg_logits)
        disc_pred_source = self.discriminator(source_seg_logits)
        # get data dict to compute losses for disc
        data_dict = dict(segmentor=self.segmentor,
                         disc=self.discriminator,
                         disc_pred_target=disc_pred_target,
                         disc_pred_source=disc_pred_source,
                         target_imgs=target_imgs,
                         source_imgs=source_imgs,
                         iteration=curr_iter,
                         batch_size=batch_size,
                         loss_scaler=loss_scaler)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(loss_disc,
                                optimizer['discriminator'],
                                loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['discriminator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['discriminator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['discriminator'].step()

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(target_seg_logits=target_seg_logits.cpu(),
                           source_seg_logits=source_seg_logits.cpu())
            outputs = dict(log_vars=log_vars_disc,
                           num_samples=batch_size,
                           results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        # 训练 Segmentor
        set_requires_grad(self.discriminator, False)
        optimizer['segmentor'].zero_grad()

        # TODO: add noise sampler to customize noise sampling
        target_seg_logits = self.segmentor.forward_dummy(target_imgs)
        source_seg_logits = self.segmentor.forward_dummy(source_imgs)
        disc_pred_target = self.discriminator(target_seg_logits)
        disc_pred_source = self.discriminator(source_seg_logits)

        data_dict_ = dict(
            segmentor=self.segmentor,
            disc=self.discriminator,
            disc_pred_target=disc_pred_target,
            disc_pred_source=disc_pred_source,
            #   target_imgs=target_imgs,
            #   source_imgs=source_imgs,
            source_seg_logits=source_seg_logits,
            source_gt_masks=source_gt_masks,
            iteration=curr_iter,
            batch_size=batch_size,
            loss_scaler=loss_scaler)
        loss_seg, log_vars_seg = self._get_seg_loss(data_dict_)
        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_seg))

        if loss_scaler:
            loss_scaler.scale(loss_seg).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(loss_seg, optimizer['segmentor'],
                                loss_id=1) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_seg.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['segmentor'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['segmentor'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['segmentor'].step()

        log_vars = {}
        log_vars.update(log_vars_seg)
        log_vars.update(log_vars_disc)

        results = dict(target_seg_logits=target_seg_logits.cpu(),
                       source_seg_logits=source_seg_logits.cpu())
        outputs = dict(log_vars=log_vars_disc,
                       num_samples=batch_size,
                       results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    def _get_disc_loss(self, outputs_dict: dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_target'] = self.gan_loss(
            outputs_dict['disc_pred_target'],
            target_is_real=False,
            is_disc=True)
        losses_dict['loss_disc_source'] = self.gan_loss(
            outputs_dict['disc_pred_source'],
            target_is_real=True,
            is_disc=True)

        # disc auxiliary loss
        if self.with_disc_auxiliary_loss:
            for loss_module in self.disc_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                if loss_ is None:
                    continue
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)
        return loss, log_var

    def _get_seg_loss(self, outputs_dict: dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # seg loss
        source_seg_logits = outputs_dict['source_seg_logits']
        source_gt_masks = outputs_dict['source_gt_masks']
        loss = self.segmentor.decode_head.losses(source_seg_logits,
                                                 source_gt_masks)
        losses_dict['loss_source_seg'] = loss

        # TODO seg auxiliary loss

        # parse losses
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var
