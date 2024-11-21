# -*- coding: utf-8 -*-
"""
====================================
@File Name ：train.py
@Time ： 2024/9/30 下午7:30
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import argparse
import shutil
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")

parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--algo", type=str, required=True, help="Name of the RL algorithm to use.")
parser.add_argument("--task", type=str, default="Ftr-Crossing-Direct-v0", help="Name of the task to train on.")
parser.add_argument("--seed", type=int, default=40, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint.")
parser.add_argument("--epoch", type=int, default=3000, help="RL Policy training iterations.")
parser.add_argument("--sim_device", type=str, default="cpu", help="Simulation device to use.")
parser.add_argument("--rl_device", type=str, default="cuda:0", help="RL device to use.")
parser.add_argument("--terrain", type=str, default="cur_steps_up", help="Name of the terrain to use.")
parser.add_argument("--experiment", type=str, default="", help="Name of the experiment to run.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--play", action="store_true", default=False, help="Interval between video recordings (in steps).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

default_experiment = f"{args_cli.terrain}_{args_cli.algo}_test" if args_cli.play \
    else f"{args_cli.terrain}_{args_cli.algo}_seed{args_cli.seed}"

if len(args_cli.experiment) == 0:
    args_cli.experiment = default_experiment
elif "@(default)" in args_cli.experiment:
    args_cli.experiment = args_cli.experiment.replace("@(default)", default_experiment)


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args + [
    f"env.scene.num_envs={args_cli.num_envs}",
    f"env.sim.device={args_cli.sim_device}",
    f"env.terrain_name={args_cli.terrain}",
    f"agent.params.config.seed={args_cli.seed}",
    f"agent.params.config.max_iterations={args_cli.epoch}",
    f"agent.params.config.device={args_cli.rl_device}",
    f"agent.params.config.checkpoint={args_cli.checkpoint}",
    f"agent.params.config.experiment={args_cli.experiment}",
    f"agent.params.config.test={args_cli.play}",

]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os

import gym
import gymnasium
import omni.isaac.lab_tasks  # noqa: F401
import torch
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_yaml
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

import ftr_envs.tasks
import ftr_envs.utils.omega_conf
from ftr_algo.utils.process_marl import process_MultiAgentRL
from ftr_algo.utils.process_sarl import process_sarl
from ftr_envs.envs.metrics_env import MaxEpochException, MetricsEnv


class SARLWrap(gym.Wrapper):
    def __init__(self, env: gymnasium.Env, env_cfg, agent_cfg):
        super().__init__(env)
        self.env_cfg = env_cfg
        self.agent_cfg = agent_cfg
        self.rl_device = self.agent_cfg['params']['config']["device"]
        self.observation_space = gym.spaces.Box(-np.Inf, np.Inf, (self.env.observation_space.shape[1],))
        self.state_space = gym.spaces.Box(-np.Inf, np.Inf, (self.env.observation_space.shape[1],))
        self.action_space = gym.spaces.Box(-1, 1, (self.env.unwrapped.flipper_num,))
        self.is_vector_env = env.unwrapped.is_vector_env

    def step(self, action):
        obs, rew, done, trunc, info = super().step(action)
        self.obs = obs['policy'].to(self.rl_device)
        return (
            self.obs,
            rew.to(self.rl_device),
            done.to(self.rl_device) | trunc.to(self.rl_device),
            info,
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.obs = obs['policy'].to(self.rl_device)
        return self.obs

    def get_state(self):
        return self.obs


class MARLWrap(SARLWrap):
    def __init__(self, env, env_cfg, agent_cfg):
        super().__init__(env, env_cfg, agent_cfg)
        self.num_agents = agent_cfg["params"]["config"]["num_agents"]
        self.num_observations = self.env.observation_space.shape[1]
        self.num_states = self.num_observations
        self.observation_space = [
            gym.spaces.Box(low=-1, high=1, shape=(self.num_observations,)) for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            gym.spaces.Box(low=-1, high=1, shape=(self.num_states,)) for _ in range(self.num_agents)
        ]
        action_num = self.env.action_space.shape[1]
        assert action_num % self.num_agents == 0
        self.action_space = [
            gym.spaces.Box(low=-1, high=1, shape=(action_num // self.num_agents,)) for _ in range(self.num_agents)
        ]

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        state = obs
        return (
            torch.transpose(torch.stack([obs] * self.num_agents), 1, 0),
            torch.transpose(torch.stack([state] * self.num_agents), 1, 0),
            None,
        )

    def step(self, action):
        action = torch.cat(action, dim=-1)
        obs, rew, done,  info = super().step(action)
        state = obs
        return (
            torch.transpose(torch.stack([obs] * self.num_agents), 1, 0),
            torch.transpose(torch.stack([state] * self.num_agents), 1, 0),
            torch.transpose(torch.stack([rew.unsqueeze(-1)] * self.num_agents), 1, 0),
            torch.transpose(torch.stack([done] * self.num_agents), 1, 0),
            None,
            None,
        )


@hydra_task_config(args_cli.task, f"ftr_algo_{args_cli.algo}_cfg_entry_point")
def train_ftr(env_cfg, agent_cfg):
    torch.load = partial(torch.load, map_location=torch.device(args_cli.rl_device))
    shutil.rmtree("outputs")

    cfg_train = agent_cfg["params"]
    algo = cfg_train["algo"]["name"]
    logdir = cfg_train['learn']["full_experiment_name"]
    test = agent_cfg["params"]["config"]["test"]
    epoch = args_cli.epoch
    dump_yaml(os.path.join(logdir, "env_config.yaml"), env_cfg)
    dump_yaml(os.path.join(logdir, "agent_config.yaml"), agent_cfg)

    env = gymnasium.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if test:
        env = MetricsEnv(env, saved_dir=logdir, max_epoch=epoch)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(logdir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gymnasium.wrappers.RecordVideo(env, **video_kwargs)

    args = {
        "algo": algo,
        "model_dir": cfg_train["config"]["checkpoint"],
        "max_iterations": int(cfg_train["config"]["max_iterations"]),
        "logdir": logdir,
    }
    args = SimpleNamespace(**args)

    if algo in ["ppo", "ddpg", "sac", "td3", "trpo"]:
        env = SARLWrap(env, env_cfg, agent_cfg)
        sarl = process_sarl(args, env, cfg_train, logdir)
        iterations = cfg_train["learn"]["max_iterations"]

        if args.max_iterations > 0:
            iterations = args.max_iterations

        def run_sarl():
            sarl.run(
                num_learning_iterations=iterations,
                log_interval=cfg_train["learn"]["save_interval"],
            )
        if test:
            try:
                run_sarl()
            except MaxEpochException:
                print("[INFO] Max epoch reached. Exiting.")
        else:
            run_sarl()
    elif algo in ["mappo", "happo", "hatrpo", "maddpg", "ippo"]:
        env = MARLWrap(env, env_cfg, agent_cfg)
        runner = process_MultiAgentRL(args, env=env, config=cfg_train, model_dir=args.model_dir)

        # test
        if args.model_dir != "":
            runner.eval(1000)
        else:
            if test:
                try:
                    runner.run()
                except MaxEpochException:
                    print("[INFO] Max epoch reached. Exiting.")
            else:
                runner.run()
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    train_ftr()
