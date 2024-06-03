'''

'''
from typing import Union, Tuple
import os

from gym import Env
from gym.core import ObsType, ActType

import torch
import gym
from gym import spaces
import yaml
from types import SimpleNamespace
from ptask_envs.envs.start import launch_isaacgym_env

from ptask_ftr.utils.process_sarl import process_sarl
from ptask_ftr.utils.process_marl import process_MultiAgentRL


class SARLWrap(gym.Wrapper):

    def __init__(self, env: Env, cfg):
        super().__init__(env)
        self.cfg = cfg

    @property
    def rl_device(self):
        return self.cfg['rl_device']

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, info = super().step(action)
        return obs.to(self.rl_device), rew.to(self.rl_device), done.to(self.rl_device), info

    def reset(self, **kwargs) -> Union[ObsType, tuple[ObsType, dict]]:
        return super().reset(**kwargs).to(self.rl_device)

    def get_state(self):
        return self.env.task.obs_buf.to(self.rl_device)


class MARLWrap(SARLWrap):

    def __init__(self, env: Env, cfg):
        super().__init__(env, cfg)
        self.num_agents = 2
        self.num_observations = self.task.num_observations
        self.observation_space = [spaces.Box(low=-1, high=1, shape=(self.num_observations,)) for _ in
                                  range(self.num_agents)]
        self.share_observation_space = [spaces.Box(low=-1, high=1, shape=(self.num_observations,)) for _ in
                                        range(self.num_agents)]
        self.action_space = [spaces.Box(low=-1, high=1, shape=(2,)) for _ in range(self.num_agents)]

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return torch.transpose(torch.stack([obs] * self.num_agents), 1, 0), torch.transpose(
            torch.stack([obs] * self.num_agents), 1, 0), None

    def step(self, action):
        action = torch.cat(action, dim=-1)
        obs, rew, done, info = super().step(action)
        return torch.transpose(torch.stack([obs] * self.num_agents), 1, 0), \
            torch.transpose(torch.stack([obs] * self.num_agents), 1, 0), \
            torch.transpose(torch.stack([rew.unsqueeze(-1)] * self.num_agents), 1, 0), \
            torch.transpose(torch.stack([done] * self.num_agents), 1, 0), \
            None, None


def train_ftr():
    with launch_isaacgym_env() as ret:
        cfg_dict = ret['config']
        cfg_train = cfg_dict['train']['params']

        algo = cfg_train['algo']['name']
        logdir = cfg_dict['experiment']

        test = cfg_dict['test']
        if test:
            from ptask_common.utils.torch import init_load_device
            init_load_device('cpu')
        else:
            os.makedirs(logdir, exist_ok=True)

        with open(os.path.join(logdir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(cfg_dict, f)

        args = {
            'algo': algo,
            'model_dir': cfg_dict['checkpoint'],
            'max_iterations': int(cfg_dict['max_iterations']),
            'logdir': logdir,
        }
        args = SimpleNamespace(**args)

        if algo in ["ppo", "ddpg", "sac", "td3", "trpo"]:
            env = SARLWrap(ret['env'], cfg_dict)
            sarl = process_sarl(args, env, cfg_train, logdir)
            iterations = cfg_train["learn"]["max_iterations"]

            if args.max_iterations > 0:
                iterations = args.max_iterations

            sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
        elif algo in ["mappo", "happo", "hatrpo", "maddpg", "ippo"]:
            env = MARLWrap(ret['env'], cfg_dict)
            runner = process_MultiAgentRL(args, env=env, config=cfg_train, model_dir=args.model_dir)

            # test
            if args.model_dir != "":
                runner.eval(1000)
            else:
                runner.run()
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    train_ftr()
