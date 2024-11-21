# -*- coding: utf-8 -*-
"""
====================================
@File Name ：__init__.py.py
@Time ： 2024/10/15 下午2:12
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import gymnasium as gym

from . import agents
from .push_cube_env import PushCubeEnv, PushCubeEnvCfg

gym.register(
    id="Ftr-PushCube-v0",
    entry_point="ftr_envs.tasks.push_cube.push_cube_env:PushCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PushCubeEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
