from omegaconf import DictConfig, ListConfig
from typing import Dict

def omegaconf_to_dict(d)->Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""

    if isinstance(d, int) or isinstance(d, str) or isinstance(d, float):
        return d

    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        elif isinstance(v, ListConfig):
            ret[k] = [omegaconf_to_dict(i) for i in v]
        else:
            ret[k] = v
    return ret