_base_ = [
    'base_models/dannet.py', 'datasets/city2dark.py', 'default_runtime.py'
]

# use dynamic runner
runner = dict(type='TwoDataloaderRunner',
              is_dynamic_ddp=False,
              pass_training_status=True)

# define optimizer
optimizer = dict(segmentor=dict(type='SGD',
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=0.0005),
                 discriminator=dict(type='Adam', lr=0.0002,
                                    betas=(0.5, 0.999)))
lr_config = None
