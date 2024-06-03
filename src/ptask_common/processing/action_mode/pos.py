# -*- coding: utf-8 -*-
"""
====================================
@File Name ：pos.py
@Time ： 2024/3/12 下午5:05
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import numpy as np
from gym import spaces

import torch

from .define import ActionMode, ActionModeFactory


class Position4Mode(ActionMode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gym_space(self):
        return spaces.Box(np.array([-1] * 4), np.array([1] * 4))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        actions_v_w[:, 0] = default_v

        result['vel'] = actions_v_w
        result['flipper'] = actions * self.flipper_pos_max

        return result


class Position5Mode(ActionMode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gym_space(self):
        return spaces.Box(np.array([-1] * 5), np.array([1] * 5))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        actions_v_w[:, 0] = (actions[:, 0] + 1) / 2 * self.max_w

        result['vel'] = actions_v_w
        result['flipper'] = actions[:, 1:] * self.flipper_pos_max

        return result


class Position2Mode(ActionMode):

    @property
    def gym_space(self):
        return spaces.Box(np.array([-1] * 2), np.array([1] * 2))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        actions_v_w[:, 0] = default_v

        actions_flipper = torch.zeros((num_envs, 4), device=device)
        actions_flipper[:, 0] = actions_flipper[:, 1] = actions[:, 0]
        actions_flipper[:, 2] = actions_flipper[:, 3] = actions[:, 1]

        result['vel'] = actions_v_w
        result['flipper'] = actions_flipper

        return result


ActionModeFactory.register('pos_4', Position4Mode)
ActionModeFactory.register('pos_5', Position5Mode)
ActionModeFactory.register('pos_2', Position2Mode)
