# -*- coding: utf-8 -*-
"""
====================================
@File Name ：seg_pos.py
@Time ： 2024/4/12 上午11:00
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import numpy as np
from gym import spaces

import torch

from .define import ActionMode, ActionModeFactory


class SegPos4Mode(ActionMode):

    seg_num = 12

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
        result['flipper'] = (torch.floor((actions + 1) / 2 * (self.seg_num + 1)) - self.seg_num / 2) / self.seg_num * self.flipper_pos_max

        return result


ActionModeFactory.register('seg_pos_4', SegPos4Mode)