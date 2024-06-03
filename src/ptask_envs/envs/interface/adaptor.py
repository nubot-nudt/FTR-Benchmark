from typing import Tuple

from ptask_envs.omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from gym.core import ObsType, ActType
import torch

from ptask_envs.envs.wrap.base import IsaacGymEnvWrapper

class KeyboardEnvAdaptor(IsaacGymEnvWrapper):

    def __init__(self, env: VecEnvRLGames):
        super().__init__(env)
        self.action_mode = self.task.action_mode

    def parse_obs(self, obs):
        obs = self.obs['obs'].cpu().numpy()
        return obs

    def step(self, actions: ActType) -> Tuple[ObsType, float, bool, dict]:

        if actions is None:
            return

        actions_t = torch.zeros((self.num_envs, 6))

        if 'vel_type' in actions:
            vel_type = actions['vel_type']
            vels = actions.get('vels', 0)
            if not isinstance(vels, torch.Tensor):
                vels = torch.tensor(vels)

            actions_t[:, 0:2] = vels

        if 'flipper_type' in actions:
            flipper_type = actions['flipper_type']
            flippers = actions.get('flippers', [0, 0, 0, 0])
            if not isinstance(flippers, torch.Tensor):
                flippers = torch.tensor(flippers)
            flippers = flippers.float()

            if flipper_type == 'dt':
                actions_t[:, 2:6] = flippers

        self.obs, rewards, dones, self.info = self.env.step(actions_t)

        return self.obs, rewards, dones, self.info


