import gym

from utils.log import logger

from .wrap.hier_env import UpperLayerWrapper
from .wrap.prepocess_wrapper import ObsUnpackDictWrapper
from .regularization.bounded_rationality import SwitchingCostWrapper
from .noise.map_noise import MapNoiseWrapper

_register_envs = {
    'upper_layer': UpperLayerWrapper,
    'switch_cost': SwitchingCostWrapper,
    'obs_unpack_dict': ObsUnpackDictWrapper,
    'map_noise': MapNoiseWrapper,
}

def wrap_envs(config, env):
    if 'wrapper' not in config['task']['env']:
        return env
    wrap_config = config['task']['env']['wrapper']

    if wrap_config is None or len(wrap_config) == 0:
        return env

    wrapper_list = []
    for wrap_info in wrap_config:
        if isinstance(wrap_info, str):
            env = _register_envs[wrap_info](env, config)
            wrapper_list.append(wrap_info)
        elif isinstance(wrap_info, dict):
            if 'argv' in wrap_info:
                env = _register_envs[wrap_info['name']](env, config, **wrap_info['argv'])
            else:
                env = _register_envs[wrap_info['name']](env, config)

            wrapper_list.append(wrap_info['name'])
        else:
            raise NotImplementedError()
    logger.info(wrapper_list)
    return env