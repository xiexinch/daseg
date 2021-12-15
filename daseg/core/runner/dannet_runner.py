import os.path as osp
import platform
import shutil
import mmcv
import warnings
import time
from mmcv.parallel import collate, is_module_wrapper
from mmcv.runner.checkpoint import save_checkpoint
from mmgen.core import DynamicIterBasedRunner, optimizer
from mmcv.runner import RUNNERS, HOOKS, get_host_info
from .iter_loader import IterLoader


@RUNNERS.register_module()
class TwoDataloaderRunner(DynamicIterBasedRunner):
    def train(self, source_dataloader, target_dataloader, **kwargs):
        if is_module_wrapper(self.model):
            _model = self.model.module
        else:
            _model = self.model
        self.model.train()
        self.mode = 'train'
        if self.optimizer_from_model:
            self.optimizer = _model.optimizer

        self.source_loader = source_dataloader
        self.target_loader = target_dataloader
        self.data_loader = self.target_loader
        # self.data_loader = self.source_loader
        self._epoch = self.data_loader.epoch
        self.call_hook('before_fetch_train_data')
        source_data_batch = next(self.source_loader)
        target_data_batch = next(self.target_loader)
        self.call_hook('before_train_iter')

        if self.pass_training_status:
            running_status = dict(iteration=self.iter, epoch=self.epoch)
            kwargs['running_status'] = running_status
        # ddp reducer for tracking dynamic computational graph
        if self.is_dynamic_ddp:
            kwargs.update(dict(ddp_reducer=self.model.reducer))
        if self.with_fp16_grad_scaler:
            kwargs.update(dict(loss_scaler=self.loss_scaler))
        if self.use_apex_amp:
            kwargs.update(dict(use_apex_amp=True))

        outputs = self.model.train_step(source_data_batch, target_data_batch,
                                        self.optimizer, **kwargs)

        # the loss scaler should be updated after ``train_step``
        if self.with_fp16_grad_scaler:
            self.scaler.update()

        # further check for the cases where the optimizer is built in
        # `train_step`.
        if self.optimizer is None:
            if hasattr(_model, 'optimizer'):
                self.optimizer_from_model = True
                self.optimizer = _model.optimizer

        # check if self.optimizer from model and track it
        if self.optimizer_from_model:
            self.optimizer = _model.optimizer
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def run(self,
            source_dataloader,
            target_dataloader,
            workflow,
            max_iters=None,
            **kwargs):
        assert isinstance(source_dataloader, list)
        assert isinstance(target_dataloader, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(source_dataloader) == len(target_dataloader) == len(
            workflow)

        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        source_iter_loaders = [IterLoader(x, self) for x in source_dataloader]
        target_iter_loaders = [IterLoader(x, self) for x in target_dataloader]
        # source_iter_loaders = [IterLoader(x) for x in source_dataloader]
        # target_iter_loaders = [IterLoader(x) for x in target_dataloader]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(source_iter_loaders[i], target_iter_loaders[i])
            time.sleep(1)
            self.call_hook('after_epoch')
            self.call_hook('after_run')

    def save_checkppoint(self,
                         out_dir,
                         filename_tmpl='iter_{}.pth',
                         meta=None,
                         save_optimizer=True,
                         create_symlink=True):
        if meta is None:
            meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
        elif isinstance(meta, dict):
            meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        _loss_scaler = self.loss_scaler if self.with_fp16_grad_scaler else None
        save_checkpoint(self.model,
                        filepath,
                        optimizer=optimizer,
                        loss_scaler=_loss_scaler,
                        save_apex_amp=self.use_apex_amp,
                        meta=meta)
        # save segmentor
        save_checkpoint(self.model.segmentor,
                        osp.join(out_dir, f'segmentor_{filename}'))
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
