# copy from mmgeneration
from functools import partial
import mmcv
import torch
from mmcv.parallel import collate, is_module_wrapper
import torch.distributed as dist
from torch.utils.data import DataLoader


class IterLoader:
    """Iteration based dataloader.

    This wrapper for dataloader is to matching the iter-based training
    proceduer.

    Args:
        dataloader (object): Dataloader in PyTorch.
        runner (object): ``mmcv.Runner``
    """
    def __init__(self, dataloader, runner):
        self._dataloader = dataloader
        self.runner = runner
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        """The number of current epoch.

        Returns:
            int: Epoch number.
        """
        return self._epoch

    def update_dataloader(self, curr_scale):
        """Update dataloader.

        Update the dataloader according to the `curr_scale`. This functionality
        is very helpful in training progressive growing GANs in which the
        dataloader should be updated according to the scale of the models in
        training.

        Args:
            curr_scale (int): The scale in current stage.
        """
        # update dataset, sampler, and samples per gpu in dataloader
        if hasattr(self._dataloader.dataset, 'update_annotations'):
            update_flag = self._dataloader.dataset.update_annotations(
                curr_scale)
        else:
            update_flag = False
        if update_flag:
            # the sampler should be updated with the modified dataset
            assert hasattr(self._dataloader.sampler, 'update_sampler')
            samples_per_gpu = None if not hasattr(
                self._dataloader.dataset, 'samples_per_gpu'
            ) else self._dataloader.dataset.samples_per_gpu
            self._dataloader.sampler.update_sampler(self._dataloader.dataset,
                                                    samples_per_gpu)
            # update samples per gpu
            if samples_per_gpu is not None:
                if dist.is_initialized():
                    # samples = samples_per_gpu
                    # self._dataloader.collate_fn = partial(
                    #     collate, samples_per_gpu=samples)
                    self._dataloader = DataLoader(
                        self._dataloader.dataset,
                        batch_size=samples_per_gpu,
                        sampler=self._dataloader.sampler,
                        num_workers=self._dataloader.num_workers,
                        collate_fn=partial(collate,
                                           samples_per_gpu=samples_per_gpu),
                        shuffle=False,
                        worker_init_fn=self._dataloader.worker_init_fn)

                    self.iter_loader = iter(self._dataloader)
                else:
                    raise NotImplementedError(
                        'Currently, we only support dynamic batch size in'
                        ' ddp, because the number of gpus in DataParallel '
                        'cannot be obtained easily.')

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)
