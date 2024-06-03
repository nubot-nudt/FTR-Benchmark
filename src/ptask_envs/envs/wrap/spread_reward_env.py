# -*- coding: utf-8 -*-
"""
====================================
@File Name ：spread_reward_env.py
@Time ： 2024/5/17 下午7:30
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
from collections import deque

import torch
from gym import Env

from ptask_envs.envs.wrap.base import IsaacGymEnvWrapper


class SpreadRewardWrapper(IsaacGymEnvWrapper):

    def __init__(self, env: Env, cfg, N=5, gamma=0.8):
        super().__init__(env)
        self.device = cfg['rl_device']
        self.N = N
        self.reward_deques = [deque(maxlen=N) for i in range(self.num_envs)]
        self._coef = torch.tensor([gamma ** i for i in range(N, 0, -1)], device=self.device)

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)

        new_rewards = torch.zeros((self.num_envs, ), device=self.device)
        for i, q in enumerate(self.reward_deques):
            new_rewards[i] = rewards[i]
            for j in range(len(q)):
                new_rewards[i] += self._coef[self.N - j - 1] * q[j]

            if dones[i] > 0:
                q.clear()
            else:
                q.append(rewards[i])

        return obs, new_rewards, dones, info