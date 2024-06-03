import random

import torch

from ptask_envs.envs.wrap.base import IsaacGymEnvWrapper

from ptask_common.processing.robot_info.obs import ObsSlice


class MapNoiseWrapper(IsaacGymEnvWrapper):

    def __init__(self, env, cfg, std=0.005):
        super().__init__(env)

        self.std = std
        self.obs_slice = ObsSlice(cfg['task']['task']['state_space'])

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)

        map = self.obs_slice.get_array_by_name(obs, 'height_maps')
        noise = torch.normal(mean=0, std=self.std, size=map.size(), device=map.device)
        noise = torch.clip(noise, -self.std, self.std)
        self.obs_slice.set_array_by_name(obs, 'height_maps', map + noise)
        # print(self.obs_slice.get_array_by_name(obs, 'height_maps').view(15, 7))
        # obs += -0.05
        return obs, rewards, dones, info
