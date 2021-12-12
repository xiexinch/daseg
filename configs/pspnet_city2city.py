_base_ = ['./base_model.py', 'dataset.py', 'default_runtime.py']

# define optimizer
optimizer = dict(segmentor=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
                 discriminator=dict(type='Adam', lr=0.0002,
                                    betas=(0.5, 0.999)))
lr_config = None
total_iters = 20000
