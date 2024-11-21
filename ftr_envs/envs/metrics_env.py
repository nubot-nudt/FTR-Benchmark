# -*- coding: utf-8 -*-
"""
====================================
@File Name ：metrics_env.py
@Time ： 2024/10/10 下午2:33
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import sys
import pickle

import numpy as np
import pandas as pd
from gymnasium import Wrapper

from ftr_envs.tasks.crossing.ftr_env import FtrEnv


class MaxEpochException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MetricsEnv(Wrapper):

    def __init__(self, env: FtrEnv, saved_dir, max_epoch=100):
        self.env = env
        super().__init__(env)
        self.num_envs = self.env.unwrapped.num_envs
        self.saved_dir = saved_dir
        self.max_epoch = max_epoch
        self.episode_start_time = np.ones(self.num_envs) * self.env.unwrapped.current_time
        self.trajectories = [list() for _ in range(self.num_envs)]
        self.result = list()
        self.mdps = list()
        self.epoch_num = 0
        self.step_num = np.zeros(self.num_envs)

    def step(self, action):
        self.step_num += 1
        last_obs = self.obs
        self.obs, self.reward, done, trunc, info = self.env.step(action)

        self.record_metrics(last_obs, self.reward)
        reset = done | trunc
        reset_env_ids = reset.nonzero(as_tuple=False).squeeze(-1)
        for ids in reset_env_ids:
            self.epoch_num += 1

            mdp = list()
            for traj in self.trajectories[ids]:
                traj["epoch"] = self.epoch_num
                mdp.append({
                    "step": traj["step"],
                    "obs": traj.pop("obs"),
                    "reward": traj.pop("reward"),
                    "action": traj.pop("action"),
                })
            self.mdps.append(mdp)
            self.step_num[ids] = 0
            self.result.extend(self.trajectories[ids])
            self.trajectories[ids].clear()

        if self.epoch_num >= self.max_epoch:
            self.save_result()
            raise MaxEpochException("Max epoch reached")

        return self.obs, self.reward, done, trunc, info

    def save_result(self):
        df = pd.DataFrame(self.result)
        df.to_csv(self.saved_dir + "/metrics.csv", index=False)
        with open(self.saved_dir + "/mdps.pickle", "wb") as f:
            pickle.dump(self.mdps, f)

    def record_metrics(self, obses, rewards):
        positions = self.env.unwrapped.positions.numpy()
        orientations = self.env.unwrapped.orientations_3.numpy()
        lin_velocities = self.env.unwrapped.robot_lin_velocities.numpy()
        ang_velocities = self.env.unwrapped.robot_ang_velocities.numpy()
        sim_time = self.env.unwrapped.current_time - self.episode_start_time
        flippers = self.env.unwrapped.flipper_positions.numpy()

        obses = obses["policy"].numpy().copy()
        rewards = rewards.numpy().copy()
        actions = self.env.unwrapped.actions.numpy().copy()

        for i in range(self.num_envs):
            x, y, z = positions[i].tolist()
            roll, pitch, yaw = orientations[i].tolist()

            lin_x, lin_y, lin_z = lin_velocities[i].tolist()
            ang_roll, ang_pitch, ang_yaw = ang_velocities[i].tolist()

            flipper = flippers[i]

            self.trajectories[i].append(
                {
                    "step": self.step_num[i],
                    "pitch": pitch,
                    "roll": roll,
                    "yaw": yaw,
                    "x": x,
                    "y": y,
                    "z": z,
                    "lin_x": lin_x,
                    "lin_y": lin_y,
                    "lin_z": lin_z,
                    "ang_roll": ang_roll,
                    "ang_pitch": ang_pitch,
                    "ang_yaw": ang_yaw,
                    "time": sim_time[i],
                    "fin_fl": flipper[0],
                    "fin_fr": flipper[1],
                    "fin_rl": flipper[2],
                    "fin_rr": flipper[3],

                    "obs": obses[i],
                    "reward": rewards[i],
                    "action": actions[i],
                }
            )

    def reset(self, *args, **kwargs):
        self.obs, info = self.env.reset(*args, **kwargs)
        self.trajectories = [list() for _ in range(self.num_envs)]
        return self.obs, info

    def __del__(self):
        self.save_result()
