# dataset settings
source_dataset_type = 'CityscapesDataset'
source_data_root = 'data/cityscapes/'
target_dataset_type = 'CityscapesDataset'
target_data_root = 'data/cityscapes/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

mix_dataset_pipeline = [
    dict(type='Collect',
         keys=['source_img', 'source_gt_mask', 'target_img'],
         meta_keys=[])
]

source_dataset_cfg_train = dict(type=source_dataset_type,
                                data_root=source_data_root,
                                img_dir='leftImg8bit/train',
                                ann_dir='gtFine/train',
                                pipeline=train_pipeline)
source_dataset_cfg_test = dict(type=source_dataset_type,
                               data_root=source_data_root,
                               img_dir='leftImg8bit/val',
                               ann_dir='gtFine/val',
                               pipeline=test_pipeline)

target_dataset_cfg_train = dict(type=source_dataset_type,
                                data_root=source_data_root,
                                img_dir='leftImg8bit/train',
                                ann_dir='gtFine/train',
                                pipeline=train_pipeline)
target_dataset_cfg_test = dict(type=source_dataset_type,
                               data_root=source_data_root,
                               img_dir='leftImg8bit/val',
                               ann_dir='gtFine/val',
                               pipeline=test_pipeline)

# mix dataset
data = dict(samples_per_gpu=2,
            workers_per_gpu=4,
            train=dict(type='MixDataset',
                       source_cfg=source_dataset_cfg_train,
                       target_cfg=target_dataset_cfg_train,
                       pipeline=mix_dataset_pipeline),
            val=dict(type='MixDataset',
                     source_cfg=source_dataset_cfg_test,
                     target_cfg=target_dataset_cfg_test,
                     pipeline=mix_dataset_pipeline),
            test=dict(type='MixDataset',
                      source_cfg=source_dataset_cfg_test,
                      target_cfg=target_dataset_cfg_test,
                      pipeline=mix_dataset_pipeline))