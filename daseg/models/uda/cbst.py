from daseg.models.builder import UDA
from .base_uda import UDADecorator


@UDA.register_module()
class CBST(UDADecorator):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        outputs = dict(log_vars=log_vars,
                       num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas, target_gt_semantic_seg):
        pass
