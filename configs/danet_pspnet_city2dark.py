_base_ = ['./base_model.py', 'city2dark.py', 'default_runtime.py']

# use dynamic runner
runner = dict(type='TwoDataloaderRunner',
              is_dynamic_ddp=False,
              pass_training_status=True)
