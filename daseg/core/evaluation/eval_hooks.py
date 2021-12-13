import os.path as osp
from mmseg.core import DistEvalHook as DistEvalHook_, EvalHook as EvalHook_
from mmseg.apis import single_gpu_test, multi_gpu_test

from daseg.datasets import build_dataloader
from torch.nn.modules.batchnorm import _BatchNorm
import torch.distributed as dist


class EvalHook(EvalHook_):
    def _do_evaluate(self, runner):
        results = single_gpu_test(runner.model.module.segmentor,
                                  self.dataloader,
                                  show=False,
                                  pre_eval=self.pre_eval)
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(DistEvalHook_):
    def _do_evaluate(self, runner):
        if self.broadcast_bn_buffer:
            model = runner.model.module.segmentor
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)
        if not self._should_evaluate(runner):
            return

        tmp_dir = self.tmpdir
        if tmp_dir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = multi_gpu_test(runner.model.module.segmentor,
                                 self.dataloader,
                                 tmpdir=tmpdir,
                                 gpu_collect=self.gpu_collect,
                                 pre_eval=self.pre_eval)

        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
