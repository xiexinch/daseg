_base_ = [
    '../datasets/city2dark_v2.py',
]

# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='UDA',
    model=dict(
        type='EncoderDecoder',
        backbone=dict(type='ResNetV1c',
                      depth=50,
                      num_stages=4,
                      out_indices=(0, 1, 2, 3),
                      dilations=(1, 1, 2, 4),
                      strides=(1, 2, 1, 1),
                      norm_cfg=norm_cfg,
                      norm_eval=False,
                      style='pytorch',
                      contract_dilation=True),
        decode_head=dict(type='FCNHead',
                         in_channels=2048,
                         in_index=3,
                         channels=512,
                         num_convs=2,
                         concat_input=True,
                         dropout_ratio=0.1,
                         num_classes=19,
                         norm_cfg=norm_cfg,
                         align_corners=False,
                         loss_decode=dict(type='CrossEntropyLoss',
                                          use_sigmoid=False,
                                          loss_weight=1.0)),
        auxiliary_head=dict(type='FCNHead',
                            in_channels=1024,
                            in_index=2,
                            channels=256,
                            num_convs=1,
                            concat_input=False,
                            dropout_ratio=0.1,
                            num_classes=19,
                            norm_cfg=norm_cfg,
                            align_corners=False,
                            loss_decode=dict(type='CrossEntropyLoss',
                                             use_sigmoid=False,
                                             loss_weight=0.4)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')))
model_checkpoint = 'checkpoints/fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth'
train_cfg = None
test_cfg = dict(mode='whole')

runner = dict(type='EpochBasedRunner', max_epochs=10)

# schedules configs
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)
# runtime settings
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
# runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(by_epoch=True, metric='mIoU', pre_eval=True)

# pretrain settings
num_classes = 19
cudnn_benchmark = False
num_rounds = 4
epoch_per_round = 4
target_portion = 0.2
target_port_step = 0.05
max_target_portion = 0.5
rare_cls_nums = 3
temp_root = 'work_dirs/cbst'
temp_dirs = 'work_dirs/cbst/results'
conf_dict_path = 'work_dirs/cbst/conf_dict.json'
pred_cls_num_path = 'work_dirs/cbst/pred_cls_num.npy'
cls_thresh_path = 'work_dirs/cbst/cls_thresh.npy'
