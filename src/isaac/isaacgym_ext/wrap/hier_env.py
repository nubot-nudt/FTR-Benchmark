
import gym
import torch

from autonomic.executor import RlgamesPredictor

from isaacgym_ext.wrap.base import IsaacGymEnvWrapper

class UpperLayerWrapper(IsaacGymEnvWrapper):

    def __init__(self, env, cfg, option_cfg):
        super().__init__(env)
        self.cfg = cfg
        self.option_cfg = option_cfg

        self.option_num_actions = self.task._num_actions
        self.load_predictor()
        self.action_space = gym.spaces.Discrete(len(self.predictors))

    def load_predictor(self):
        self.predictors = []

        for cfg in self.option_cfg:
            if cfg.pop('name') == 'rlgames':
                self.predictors.append(
                    RlgamesPredictor.load_model(**cfg, device=self.cfg['rl_device'])
                )
            else:
                raise NotImplemented()

    def step(self, actions):
        options_actions = torch.zeros((self.num_envs, self.option_num_actions))
        actions = actions.long()
        # print(actions)
        for i, option in enumerate(actions):
            options_actions[i, :] = self.predictors[option].get_action(self.obs['obs'][i])

        self.obs, rewards, dones, info = self.env.step(options_actions)


        return self.obs, rewards, dones, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs


