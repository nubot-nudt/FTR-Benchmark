from typing import Tuple

import gym
import torch
from gym.core import ObsType, ActType

from bottle import Bottle

app = Bottle()

from ptask_envs.envs.wrap.base import IsaacGymEnvWrapper


class NumpyAloneEnv(IsaacGymEnvWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)

        action = action.view(self.num_envs, -1)
        obs, rewards, dones, info = self.env.step(action)

        if isinstance(obs, dict):
            obs = obs['obs']
        return obs[0, :].cpu(), rewards[0].item(), dones[0].item(), info

    def reset(self):
        self.task.reset()
        actions = torch.zeros((self.num_envs, self.task.num_actions), device=self.task.rl_device)
        obs, _, _, _ = self.step(actions)

        return obs


