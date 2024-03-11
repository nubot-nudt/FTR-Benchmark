import torch

from isaacgym_ext.wrap.base import IsaacGymEnvWrapper

class SwitchingCostWrapper(IsaacGymEnvWrapper):

    def __init__(self, env, cfg, cost_coef):
        super().__init__(env)

        self.cost_coef = cost_coef

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)

        additional_rewards = 0
        if hasattr(self, 'last_actions'):
            additional_rewards = torch.sum((self.last_actions - actions) ** 2, dim=1)
            self.last_actions = actions

        return obs, rewards + self.cost_coef * additional_rewards, dones, info

