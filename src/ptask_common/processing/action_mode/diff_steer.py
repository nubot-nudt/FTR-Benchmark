import sys

import numpy as np
from gym import spaces
import pysnooper

import torch

from .define import ActionMode, ActionModeFactory


class DiffSteer5(ActionMode):

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

        v_index = (actions[:, 0] > 0)
        w_index = (actions[:, 0] < 0)

        actions_v_w[v_index, 0] = actions[v_index, 0] * self.max_v
        actions_v_w[w_index, 1] = (-actions[w_index, 0] * 2 - 1) * self.max_w

        result['vel'] = actions_v_w
        result['flipper'] = actions[:, 1:] * self.flipper_dt

        return result


ActionModeFactory.register('diff_5', DiffSteer5)


