from enum import Enum
from itertools import product

import numpy as np
from gym import spaces

import torch


class ActionMode(Enum):
    # [v, w, fl, fr, rl, rr]
    continuous_std_6 = 'continuous_std_6'

    # [w, fl, fr, rl, rr]
    continuous_std_5 = 'continuous_std_5'

    # [fl, fr, rl, rr]
    continuous_std_4 = 'continuous_std_4'

    # [front, rear]
    continuous_std_2 = 'continuous_std_2'

    # [ 速度5档, 转弯3, 摆臂4*3 ]
    discrete_std_6 = 'discrete_std_6'

    # [ [速度5档 + 左右转弯], 四个摆臂（三基元） ]
    discrete_std_5 = 'discrete_std_5'

    # [ (速度5档 + 左右转弯) * 四个摆臂（三基元） ]
    discrete_std_1 = 'discrete_std_1'


_discrete_std_1_map = torch.tensor(list(product(range(7), [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1])))


class ActionModeExecute():
    def __init__(self, mode: ActionMode, max_v=0.5, flipper_dt=2):
        self.mode = mode
        self.max_v = max_v
        self.max_w = self.max_v
        self.flipper_dt = flipper_dt

    def convert_actions_to_std_dict(self, actions: torch.Tensor, default_v=0.2, default_w=0.0):
        """
        转化actions为标准运动参数
        :param actions: 在某个action_mode下产生的actions
        :param default_w: 在v缺少的时的线速度
        :param default_v: 在w缺少的时的加速度
        :return: 返回一个字典，vel是机器人线速度和角速度[v, w]，flipper为四个摆臂的位置[fl, fr, rl, rr]
        """

        result = dict()

        default_v = torch.clip(torch.tensor(default_v), -self.max_v, self.max_v)
        default_w = torch.clip(torch.tensor(default_w), -self.max_w, self.max_w)
        mode = self.mode
        max_v = self.max_v
        flipper_dt = self.flipper_dt
        num_envs = len(actions)
        device = actions.device

        if mode == ActionMode.continuous_std_6:
            actions_t = torch.zeros((num_envs, 2), device=device)
            actions_t[:, 0] = actions[:, 0] * self.max_v / 2 + self.max_v / 2
            actions_t[:, 1] = actions[:, 1] * self.max_v / 2 + self.max_v / 2

            result['vel'] = actions[:, :2] * self.max_v
            result['flipper'] = actions[:, 2:] * self.flipper_dt

        elif mode == ActionMode.continuous_std_5:
            actions_t = torch.zeros((num_envs, 2), device=device)
            actions_t[:, 0] = default_v
            actions_t[:, 1] = actions[:, 0] * self.max_v / 2 + self.max_v / 2

            result['vel'] = actions_t
            result['flipper'] = actions[:, 1:] * self.flipper_dt

        elif mode == ActionMode.continuous_std_4:
            actions_t = torch.zeros((num_envs, 2), device=device)
            actions_t[:, 0] = default_v
            actions_t[:, 1] = default_w

            result['vel'] = actions_t
            result['flipper'] = actions * self.flipper_dt

        elif mode == ActionMode.continuous_std_2:
            actions_v_w = torch.zeros((num_envs, 2), device=device)
            actions_v_w[:, 0] = default_v
            actions_v_w[:, 1] = default_w

            actions_flipper = torch.zeros((num_envs, 4), device=device)
            actions_flipper[:, 0] = actions_flipper[:, 1] = actions[:, 0]
            actions_flipper[:, 2] = actions_flipper[:, 3] = actions[:, 1]

            result['vel'] = actions_v_w
            result['flipper'] = actions_flipper * self.flipper_dt

        elif mode == ActionMode.discrete_std_1:
            actions_t = _discrete_std_1_map[actions.long()].view(num_envs, -1)

            actions_v_w = torch.zeros((num_envs, 2), device=device)
            w_indices = actions_t[:, 0] > 4
            v_indices = actions_t[:, 0] <= 4

            actions_v_w[w_indices, 1] = (actions_t[w_indices, 0] - 5.5) * 2 * max_v * 2 / 3
            actions_v_w[v_indices, 0] = (actions_t[v_indices, 0] + 1) * max_v / 5

            result['vel'] = actions_v_w
            result['flipper'] = actions_t[:, 1:] * flipper_dt

        elif mode == ActionMode.discrete_std_5:
            actions_v_w = torch.zeros((num_envs, 2), device=device)
            w_indices = actions[:, 0] > 4
            v_indices = actions[:, 0] <= 4

            actions_v_w[w_indices, 1] = (actions[w_indices, 0] - 5.5) * 2 * self.max_v * 2 / 3
            actions_v_w[v_indices, 0] = (actions[v_indices, 0] + 1) * self.max_v / 5

            result['vel'] = actions_v_w
            result['flipper'] = (actions[:, 1:] - 1) * self.flipper_dt

        elif mode == ActionMode.discrete_std_6:
            actions_v_w = torch.zeros((num_envs, 2), device=device)

            actions_v_w[:, 0] = (actions[:, 0] + 1) * self.max_v / 5
            actions_v_w[:, 1] = (actions[:, 1] - 1) * self.max_w

            result['vel'] = actions_v_w
            result['flipper'] = (actions[:, 2:] - 1) * self.flipper_dt

        else:
            raise NotImplementedError()

        return result

    @property
    def action_mode(self):
        return self.mode

    def get_gym_info(self, mode=None):

        if mode is None:
            mode = self.mode

        action_mode_maps = {
            ActionMode.continuous_std_6: {
                'num_actions': 6,
                'space': spaces.Box(
                    np.array([-1] * 6),
                    np.array([-1] * 6),
                ),
            },
            ActionMode.continuous_std_5: {
                'num_actions': 5,
                'space': spaces.Box(
                    np.array([-1] * 5),
                    np.array([1] * 5),
                ),
            },
            ActionMode.continuous_std_4: {
                'num_actions': 4,
                'space': spaces.Box(
                    np.array([-1] * 4),
                    np.array([1] * 4),
                ),
            },
            ActionMode.continuous_std_2: {
                'num_actions': 2,
                'space': spaces.Box(
                    np.array([-1, -1]),
                    np.array([1, 1]),
                ),
            },
            ActionMode.discrete_std_1: {
                'num_actions': 1,
                'space': spaces.Discrete(len(_discrete_std_1_map)),
            },
            ActionMode.discrete_std_5: {
                'num_actions': 5,
                'space': spaces.Tuple([
                    spaces.Discrete(7), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3)
                ]),
            },
            ActionMode.discrete_std_6: {
                'num_actions': 6,
                'space': spaces.Tuple([
                    spaces.Discrete(5), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3)
                ]),
            }
        }

        return action_mode_maps[mode]
