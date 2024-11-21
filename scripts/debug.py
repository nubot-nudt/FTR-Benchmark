# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""
import numpy as np

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--task", type=str, default="Ftr-Crossing-Direct-v0", help="Name of the task to train on.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args + [
    f"env.scene.num_envs={args_cli.num_envs}",
]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import omni.isaac.lab_tasks  # noqa: F401
import torch
from omni.isaac.lab_tasks.utils import parse_env_cfg

import ftr_envs.tasks
import ftr_envs.utils.omega_conf

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # env_cfg.terrain_name = "cur_steps_down"
    # env_cfg.terrain_name = "cur_stairs_up"
    env_cfg.terrain_name = "ground"
    if env_cfg.terrain_name == "ground":
        env_cfg.initial_flipper_range = (60, 60)
        env_cfg.forward_vel_range = (0., 0.)
        coef = -1
        N = 1000000
    elif env_cfg.terrain_name == "cur_steps_down":
        env_cfg.initial_flipper_range = (0, 0)
        env_cfg.forward_vel_range = (0.2, 0.3)
        env_cfg.robot_render_config["flipper"]["auxiliary_wheel_radius"] = 0.01
        coef = 0
        N = 100
    else:
        env_cfg.initial_flipper_range = (0, 60)
        env_cfg.forward_vel_range = (0.2, 0.3)
        coef = -1
        N = 100

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    datas = []
    env.reset()
    for i in range(N):
        actions = -torch.ones(env.num_envs, env.flipper_num) * coef
        ret = env.step(actions)

        time = env.unwrapped.current_time
        fin_pos = env.unwrapped.flipper_positions[0, 0]
        datas.append({
            "time": time,
            "pos": np.round(np.rad2deg(fin_pos)),
            "lin_x": env.unwrapped.robot_lin_velocities[0, 0].clone(),
        })
    df = pd.DataFrame(datas)

    plt.subplot(1, 2, 1)
    df["diff_time"] = df["time"].diff()
    df["diff_pos"] = df["pos"].diff()
    # plt.plot(df["time"], np.clip(df["diff_pos"] / df["diff_time"], -100, 100))
    plt.plot(df["time"], df["lin_x"])
    plt.subplot(1, 2, 2)
    plt.plot(df["time"], df["pos"])
    # plt.plot(df["time"], df["pos"])
    print(df.head())
    plt.show()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
