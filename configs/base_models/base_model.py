# define GAN model
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CustomStaticUnconditionalGAN',
    # pspnet r50
    segmentor=dict(
        type='EncoderDecoder',
        pretrained='open-mmlab://resnet50_v1c',
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
        decode_head=dict(type='PSPHead',
                         in_channels=2048,
                         in_index=3,
                         channels=512,
                         pool_scales=(1, 2, 3, 6),
                         dropout_ratio=0.1,
                         num_classes=19,
                         norm_cfg=norm_cfg,
                         align_corners=False,
                         loss_decode=dict(type='CrossEntropyLoss',
                                          use_sigmoid=False,
                                          loss_weight=1.0)),
        auxiliary_head=None,
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')),
    discriminator=dict(type='DCGANDiscriminator',
                       input_scale=32,
                       output_scale=4,
                       in_channels=19,
                       out_channels=100),
    seg_loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    gan_loss=dict(type='GANLoss', gan_type='vanilla'))

train_cfg = None

test_cfg = None
