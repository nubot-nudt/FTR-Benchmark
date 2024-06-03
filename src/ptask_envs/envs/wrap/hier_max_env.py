import gym
import torch

from ptask_common.autonomic.executor.rlgames import RlgamesPredictor
# from ptask_ftr.benchmark import BenchmarkPredictor

from ptask_envs.envs.wrap.base import IsaacGymEnvWrapper


class HierMaxEnv(IsaacGymEnvWrapper):

    def __init__(self, env, cfg, option_cfg):
        super().__init__(env)
        self.cfg = cfg
        self.option_cfg = option_cfg
        self.sim_device = self.cfg['sim_device']

        self.old_num_actions = self.task._num_actions
        self.load_predictor()
        self.option_num = len(self.predictors)
        self.action_space = gym.spaces.Discrete(self.option_num)

    def load_predictor(self):
        self.predictors = []

        for cfg in self.option_cfg.copy():
            name = cfg.pop('name')
            if name == 'rlgames':
                self.predictors.append(
                    RlgamesPredictor.load_model(**cfg, device=self.sim_device)
                )
            elif name == 'benchmark':
                self.predictors.append(
                    BenchmarkPredictor(**cfg, device=self.sim_device)
                )
            else:
                raise NotImplemented()

    def step(self, actions):
        options_actions = torch.zeros((self.num_envs, self.old_num_actions))
        actions = actions.long()

        for i, option in enumerate(actions):
            options_actions[i, :] = self.predictors[option].get_action(self.obs[i])

        self.obs, rewards, dones, info = self.env.step(options_actions)

        return self.obs, rewards, dones, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs
