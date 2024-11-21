# -*- coding: utf-8 -*-
"""
====================================
@File Name ：__init__.py.py
@Time ： 2024/9/29 下午12:06
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import gymnasium as gym

from . import agents
from .crossing_env import CrossingEnv, CrossingEnvCfg

gym.register(
    id="Ftr-Crossing-Direct-v0",
    entry_point="ftr_envs.tasks.crossing.crossing_env:CrossingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrossingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "ftr_algo_ppo_cfg_entry_point": f"{agents.__name__}:ftr_algo_ppo_cfg.yaml",
        "ftr_algo_trpo_cfg_entry_point": f"{agents.__name__}:ftr_algo_trpo_cfg.yaml",
        "ftr_algo_sac_cfg_entry_point": f"{agents.__name__}:ftr_algo_sac_cfg.yaml",
        "ftr_algo_ddpg_cfg_entry_point": f"{agents.__name__}:ftr_algo_ddpg_cfg.yaml",
        "ftr_algo_td3_cfg_entry_point": f"{agents.__name__}:ftr_algo_td3_cfg.yaml",
        "ftr_algo_mappo_cfg_entry_point": f"{agents.__name__}:ftr_algo_mappo_cfg.yaml",
        "ftr_algo_happo_cfg_entry_point": f"{agents.__name__}:ftr_algo_happo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
