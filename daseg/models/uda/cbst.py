<<<<<<< HEAD
from daseg.models.builder import UDA
from .base_uda import UDADecorator
=======
import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import build_segmentor
from .uda_decorator import UDADecorator, get_module
from daseg.models.builder import UDA
>>>>>>> 4be082c3119107bf2d71e1accfeb53757c5f0560


@UDA.register_module()
class CBST(UDADecorator):
    def __init__(self, **cfg):
<<<<<<< HEAD
        super().__init__(**cfg)
=======
        super(CBST, self).__init__(**cfg)
        self.max_iters = cfg['max_iters']
        self.curr_iter = 0
>>>>>>> 4be082c3119107bf2d71e1accfeb53757c5f0560

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        outputs = dict(log_vars=log_vars,
                       num_samples=len(data_batch['img_metas']))
        return outputs

<<<<<<< HEAD
    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas, target_gt_semantic_seg):
        pass
=======
    def get_weights_k(self, target_logits: torch.Tensor):
        return torch.ones(target_logits.shape)

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        log_vars = {}

        # generate pseudo label by target domain image
        with torch.no_grad():
            target_logits = self.get_model().encode_decode(
                target_img, target_img_metas)

        #

        # train with source img

        return log_vars
>>>>>>> 4be082c3119107bf2d71e1accfeb53757c5f0560
