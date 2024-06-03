# -*- coding: utf-8 -*-
"""
====================================
@File Name ：only.py
@Time ： 2024/5/17 下午9:21
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
from itertools import product

import numpy as np
from gym import spaces

import torch

from .define import ActionMode, ActionModeFactory


class OnlyFrontMode(ActionMode):
    @property
    def gym_space(self):
        return spaces.Box(np.array([-1] * 2), np.array([1] * 2))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        num_envs = len(actions)
        device = actions.device

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        actions_v_w[:, 0] = default_v
        actions_v_w[:, 1] = default_w

        actions_flipper = torch.ones((num_envs, 4), device=device)
        actions_flipper[:, :2] = actions[:, :] * self.flipper_dt

        return {
            'vel': actions_v_w,
            'flipper': actions_flipper
        }


class OnlyRearMode(ActionMode):

    @property
    def gym_space(self):
        return spaces.Box(np.array([-1] * 2), np.array([1] * 2))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        num_envs = len(actions)
        device = actions.device

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        actions_v_w[:, 0] = default_v
        actions_v_w[:, 1] = default_w

        actions_flipper = torch.ones((num_envs, 4), device=device)
        actions_flipper[:, 2:] = actions[:, :] * self.flipper_dt

        return {
            'vel': actions_v_w,
            'flipper': actions_flipper
        }


ActionModeFactory.register('only_front', OnlyFrontMode)
ActionModeFactory.register('only_rear', OnlyRearMode)
