# define GAN model
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='AdapSegNet',
    # pspnet r50
    segmentor=dict(
        type='EncoderDecoder',
        backbone=dict(type='ResNetV1c',
                      depth=101,
                      num_stages=4,
                      out_indices=(0, 1, 2, 3),
                      dilations=(1, 1, 2, 4),
                      strides=(1, 2, 1, 1),
                      norm_cfg=norm_cfg,
                      norm_eval=False,
                      style='pytorch',
                      contract_dilation=True),
        decode_head=dict(type='ASPPHead',
                         in_channels=2048,
                         in_index=3,
                         channels=512,
                         dilations=(1, 12, 24, 36),
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
        test_cfg=dict(mode='whole')),
    discriminator=dict(type='DCGANDiscriminator',
                       input_scale=128,
                       output_scale=4,
                       in_channels=19,
                       base_channels=64,
                       out_channels=100),
    seg_loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    gan_loss=dict(type='GANLoss', gan_type='vanilla'),
    segmentor_checkpoint=
    'segmentor_checkpoints/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth'

    # noqa
)

train_cfg = None
test_cfg = None
