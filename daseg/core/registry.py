from mmcv.utils import Registry, build_from_cfg

METRICS = Registry('metric')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return modules
    return build_from_cfg(cfg, registry, default_args)


def build_metric(cfg):
    return build(cfg, METRICS)
