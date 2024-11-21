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
from .trans_cargo_env import TransCargoEnv, TransCargoEnvCfg

gym.register(
    id="Ftr-TransCargo-v0",
    entry_point="ftr_envs.tasks.trans_cargo.trans_cargo_env:TransCargoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TransCargoEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
