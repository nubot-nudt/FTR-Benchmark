# -*- coding: utf-8 -*-
"""
====================================
@File Name ：__init__.py.py
@Time ： 2024/11/5 下午4:33
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import gymnasium as gym

from . import agents, anymal_d_cfg

gym.register(
    id="Isaac-Velocity-Rough-Anymal-D-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": anymal_d_cfg.AnymalDRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)