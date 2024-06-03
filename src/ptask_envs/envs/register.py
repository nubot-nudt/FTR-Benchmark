from loguru import logger

from .wrap.hier_max_env import HierMaxEnv
from .wrap.hier_weight_env import HierWeightEnv
from .wrap.prepocess_wrapper import ObsUnpackDictWrapper
from .regularization.bounded_rationality import SwitchingCostWrapper
from .noise.map_noise import MapNoiseWrapper
from .wrap.spread_reward_env import SpreadRewardWrapper

_register_envs = {
    'hier_max': HierMaxEnv,
    'hier_weight': HierWeightEnv,
    'switch_cost': SwitchingCostWrapper,
    'obs_unpack_dict': ObsUnpackDictWrapper,
    'map_noise': MapNoiseWrapper,
    'spread_reward': SpreadRewardWrapper,
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
                argv = wrap_info.copy()
                env = _register_envs[argv.pop('name')](env, config, **argv)

            wrapper_list.append(wrap_info['name'])
        else:
            raise NotImplementedError()
    logger.info(wrapper_list)
    return env
