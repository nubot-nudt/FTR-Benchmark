from itertools import product

import numpy as np
from gym import spaces

import torch

from .define import ActionMode, ActionModeFactory


class ContinuousStd6(ActionMode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gym_space(self):
        return spaces.Box(np.array([-1] * 6), np.array([1] * 6))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        actions_v_w[:, 0] = actions[:, 0] * self.max_v / 2 + self.max_v / 2
        actions_v_w[:, 1] = actions[:, 1] * self.max_w / 2 + self.max_w / 2

        result['vel'] = actions_v_w
        result['flipper'] = actions[:, 2:] * self.flipper_dt

        return result


class ContinuousStd5(ActionMode):

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
        actions_v_w[:, 0] = actions[:, 0] * self.max_v / 2 + self.max_v / 2
        actions_v_w[:, 1] = default_w

        result['vel'] = actions_v_w
        result['flipper'] = actions[:, 1:] * self.flipper_dt

        return result


class ContinuousStd4(ActionMode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gym_space(self):
        return spaces.Box(np.array([-1] * 4), np.array([1] * 4))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        default_action_v_w = torch.zeros((num_envs, 2), device=device)
        default_action_v_w[:, 0] = default_v
        default_action_v_w[:, 1] = default_w

        result['vel'] = default_action_v_w
        result['flipper'] = actions * self.flipper_dt

        return result


class DiscreteStd4(ActionMode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gym_space(self):
        return spaces.Tuple([spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3)])

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        default_action_v_w = torch.zeros((num_envs, 2), device=device)
        default_action_v_w[:, 0] = default_v
        default_action_v_w[:, 1] = default_w

        result['vel'] = default_action_v_w
        result['flipper'] = (actions[:, :] - 1) * self.flipper_dt

        return result


class DiscreteStd5(ActionMode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gym_space(self):
        return spaces.Tuple(
            [spaces.Discrete(7), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3)])

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        w_indices = actions[:, 0] > 4
        v_indices = actions[:, 0] <= 4

        actions_v_w[w_indices, 1] = (actions[w_indices, 0] - 5.5) * 2 * self.max_v * 2 / 3
        actions_v_w[v_indices, 0] = (actions[v_indices, 0] + 1) * self.max_v / 5

        result['vel'] = actions_v_w
        result['flipper'] = (actions[:, 1:] - 1) * self.flipper_dt

        return result


class DiscreteStd1(ActionMode):
    _discrete_std_1_map = torch.tensor(list(product(range(7), [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1])))

    def __init__(self, max_v=0.5, flipper_dt=2, max_w=None):
        super().__init__(max_v, flipper_dt, max_w)

    @property
    def gym_space(self):
        return spaces.Discrete(len(self._discrete_std_1_map))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        actions_t = self._discrete_std_1_map[actions.long()].view(num_envs, -1)

        actions_v_w = torch.zeros((num_envs, 2), device=device)
        w_indices = actions_t[:, 0] > 4
        v_indices = actions_t[:, 0] <= 4

        actions_v_w[w_indices, 1] = (actions_t[w_indices, 0] - 5.5) * 2 * self.max_v * 2 / 3
        actions_v_w[v_indices, 0] = (actions_t[v_indices, 0] + 1) * self.max_v / 5

        result['vel'] = actions_v_w
        result['flipper'] = actions_t[:, 1:] * self.flipper_dt

        return result


class DiscreteFlipper1(ActionMode):
    _discrete_flipper_1_map = torch.tensor(list(product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1])))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gym_space(self):
        return spaces.Discrete(len(self._discrete_flipper_1_map))

    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        result = dict()

        num_envs = len(actions)
        device = actions.device

        actions_t = self._discrete_flipper_1_map[actions.long()].view(num_envs, -1)

        default_action_v_w = torch.zeros((num_envs, 2), device=device)
        default_action_v_w[:, 0] = default_v
        default_action_v_w[:, 1] = default_w

        result['vel'] = default_action_v_w
        result['flipper'] = actions_t[:, :] * self.flipper_dt

        return result


ActionModeFactory.register('con_del_6', ContinuousStd6)
ActionModeFactory.register('con_del_5', ContinuousStd5)
ActionModeFactory.register('con_del_4', ContinuousStd4)
ActionModeFactory.register('dis_del_4', DiscreteStd4)
ActionModeFactory.register('dis_del_5', DiscreteStd5)
ActionModeFactory.register('dis_del_1', DiscreteStd1)
