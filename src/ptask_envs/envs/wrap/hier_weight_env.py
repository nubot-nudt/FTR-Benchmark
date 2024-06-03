import gym
import torch
import numpy as np

from .hier_max_env import HierMaxEnv


class HierWeightEnv(HierMaxEnv):

    def __init__(self, env, cfg, option_cfg):
        super().__init__(env, cfg, option_cfg)
        self.action_space = gym.spaces.Box(np.array([0] * self.option_num), np.array([1] * self.option_num))

    def step(self, actions):
        options_actions = torch.zeros((self.num_envs, self.old_num_actions, self.option_num), device=self.sim_device)

        local_obs = self.obs.to(self.sim_device)
        for i in range(self.option_num):
            options_actions[:, :, i] = self.predictors[i].get_action(local_obs)

        new_actions = (options_actions * actions.view(self.num_envs, 1, self.option_num).to(self.sim_device)).sum(dim=-1)
        self.obs, rewards, dones, info = self.env.step(new_actions)

        return self.obs, rewards, dones, info


