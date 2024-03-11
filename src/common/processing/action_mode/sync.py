
from itertools import product

import numpy as np
from gym import spaces

import torch

from .define import ActionMode, ActionModeFactory

class SynchronousMode(ActionMode):

    def __init__(self, max_v=0.5, flipper_dt=2, max_w=None):
        super().__init__(max_v, flipper_dt, max_w)

    @property
    def gym_space(self):
        return spaces.Box(np.array([-1] * 4), np.array([1] * 4))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        actions_v_w[:, 0] = actions[:, 0] * self.max_v / 2 + self.max_v / 2
        actions_v_w[:, 1] = actions[:, 1] * self.max_w / 2 + self.max_w / 2

        actions_flipper = torch.zeros((num_envs, 4), device=device)
        actions_flipper[:, 0] = actions_flipper[:, 1] = actions[:, 2] * self.flipper_dt
        actions_flipper[:, 2] = actions_flipper[:, 3] = actions[:, 3] * self.flipper_dt

        result['vel'] = actions_v_w
        result['flipper'] = actions_flipper

        return result

ActionModeFactory.register('synchronous_control', SynchronousMode)