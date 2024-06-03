# -*- coding: utf-8 -*-
"""
====================================
@File Name ：register.py
@Time ： 2024/5/22 下午12:42
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import os.path

from gym import spaces
import torch
import yaml
import pathlib
import numpy as np
from functools import partial

from ptask_common.autonomic.executor import BaseExecutor, ExecutorFactory


def load_rl(checkpoint, algo, cfg, device='cpu'):
    seed = cfg['seed']
    cfg_train = cfg['train']['params']

    action_space = spaces.Box(
        np.ones(4, dtype=np.float32) * -1.0, np.ones(4, dtype=np.float32) * 1.0
    )

    marl_action_space = spaces.Box(
        np.ones(2, dtype=np.float32) * -1.0, np.ones(2, dtype=np.float32) * 1.0
    )

    observation_space = spaces.Box(
        np.ones(115, dtype=np.float32) * -np.Inf,
        np.ones(115, dtype=np.float32) * np.Inf,
    )

    from ptask_ftr.algorithms.rl.ppo import ActorCritic as PPOActorCritic
    from ptask_ftr.algorithms.rl.trpo import ActorCritic as TRPOActorCritic
    from ptask_ftr.algorithms.rl.sac.module import MLPActorCritic as SACMLPActorCritic
    from ptask_ftr.algorithms.rl.ddpg import MLPActorCritic as DDPGMLPActorCritic
    from ptask_ftr.algorithms.rl.td3 import MLPActorCritic as TD3MLPActorCritic

    networks = {
        'ppo': PPOActorCritic,
        'trpo': TRPOActorCritic,
        'sac': SACMLPActorCritic,
        'ddpg': DDPGMLPActorCritic,
        'td3': TD3MLPActorCritic,
    }

    if algo in ['sac', 'ddpg', 'td3']:
        learn_cfg = cfg['train']['params']['learn']

        ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]] * learn_cfg["hidden_layer"])
        if algo == 'sac':
            model = networks[algo](observation_space, action_space, **ac_kwargs).to(device)
        else:
            act_noise = learn_cfg["act_noise"]
            model = networks[algo](observation_space, action_space, act_noise=act_noise, device=device, **ac_kwargs).to(
                device)

        model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device)))
        model.eval()
        return partial(model.act, deterministic=True)
    elif algo in ['ppo', 'trpo']:
        learn_cfg = cfg['train']['params']['learn']
        init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        model_cfg = cfg_train["policy"]
        asymmetric = learn_cfg['asymmetric']
        model = networks[algo](observation_space.shape, observation_space.shape, action_space.shape,
                               init_noise_std, model_cfg, asymmetric=asymmetric)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device)))
        model.eval()
        return model.act_inference
    elif algo in ['mappo']:
        num_agents = 2
        use_single_network = cfg_train["use_single_network"]
        model_dir = os.path.join(checkpoint, f'models_seed{seed}')
        from ptask_ftr.algorithms.marl.mappo_policy import MAPPO_Policy as Policy
        policy = []
        for agent_id in range(num_agents):
            policy.append(Policy(cfg_train, observation_space, observation_space, marl_action_space, device=device))

        for agent_id in range(num_agents):
            if use_single_network:
                policy_model_state_dict = torch.load(str(model_dir) + '/model_agent' + str(agent_id) + '.pt')
                policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

        def get_actions(obs):
            actions = []
            for i in range(num_agents):
                value, action, _, _, _ = policy[i].get_actions(obs, obs, torch.tensor(1), torch.tensor(1), torch.tensor(1))
                actions.append(action)
            return torch.cat(actions, dim=-1)

        return get_actions


class FTRExecutor(BaseExecutor):

    def __init__(self, checkpoint, device='cpu'):
        checkpoint = pathlib.Path(checkpoint).absolute()
        runs_dir = checkpoint.parent

        _, _, algo, _ = runs_dir.name.split('_')

        with open(runs_dir / 'config.yaml') as f:
            cfg = yaml.safe_load(f)

        if algo in ['sac', 'ppo', 'trpo', 'ddpg', 'td3', 'mappo']:
            self.act_func = load_rl(checkpoint, algo, cfg, device)
        else:
            raise NotImplementedError(f'{algo}')

    def get_action(self, obs):
        return self.act_func(obs)

    def predict(self, *args, **kwargs):
        pass


ExecutorFactory().register('ftr', FTRExecutor)
