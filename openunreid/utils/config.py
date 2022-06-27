# Credit to https://github.com/open-mmlab/OpenLidarPerceptron/blob/master/pcdet/config.py  # noqa

from pathlib import Path

import yaml
from easydict import EasyDict


def log_config_to_file(cfg, pre="cfg"):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print("\n%s.%s = edict()" % (pre, key))
            log_config_to_file(cfg[key], pre=pre + "." + key)
            continue
        print("%s.%s: %s" % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval

    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split(".")
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, "NotFoundKey: %s" % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, "NotFoundKey: %s" % subkey
        try:
            value = literal_eval(v)
        except Exception:
            value = v

        if (not isinstance(value, type(d[subkey]))) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(",")
            for src in key_val_list:
                cur_key, cur_val = src.split(":")
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif (not isinstance(value, type(d[subkey]))) and isinstance(d[subkey], list):
            val_list = value.split(",")
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert isinstance(
                value, type(d[subkey])
            ), "type {} does not match original type {}".format(
                type(value), type(d[subkey])
            )
            d[subkey] = value


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, "r") as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            new_config = yaml.load(f)
        config.update(EasyDict(new_config))

    cfg.DATA_ROOT = Path(cfg.DATA_ROOT)
    cfg.LOGS_ROOT = Path(cfg.LOGS_ROOT)

    return config


cfg = EasyDict()
cfg.LOCAL_RANK = 0
