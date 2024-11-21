# -*- coding: utf-8 -*-
"""
====================================
@File Name ：crossing.py
@Time ： 2024/9/29 下午12:07
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
from typing import Sequence

import einops
import numpy as np
import torch
from omni.isaac.lab.envs import VecEnvObs
from omni.isaac.lab.sim import PhysxCfg

from ftr_envs.utils.torch import add_noise

from .ftr_env import FtrEnv, FtrEnvCfg, configclass


@torch.jit.script
def point_in_rotated_ellipse(x, y, h, k, a, b, theta):
    """
    其中 (h, k) 是椭圆的中心坐标，a 和 b 分别是椭圆在旋转前 x 轴和 y 轴上的半长轴和半短轴的长度，theta 是椭圆的旋转角度（弧度制）。
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    term1 = ((x - h) * cos_theta + (y - k) * sin_theta) ** 2 / a ** 2
    term2 = ((x - h) * sin_theta - (y - k) * cos_theta) ** 2 / b ** 2
    return term1 + term2 <= 1


@torch.jit.script
def out_of_range(pos, start, target):
    center = (start + target) / 2
    op = (target - start)[:, :2]
    d_max = op[:, :2].norm(dim=-1, p=2)
    theta = torch.arctan(op[:, 1] / op[:, 0])
    return ~point_in_rotated_ellipse(
        pos[:, 0],
        pos[:, 1],
        center[:, 0],
        center[:, 1],
        d_max / 2 + 0.2,
        d_max / 4 + 0.1,
        theta,
    )


@configclass
class CrossingEnvCfg(FtrEnvCfg):
    # env
    num_actions = 4
    num_observations = 115
    num_states = 0


class CrossingEnv(FtrEnv):
    cfg: CrossingEnvCfg

    def __init__(self, cfg: CrossingEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.pitch_t = np.deg2rad(30)
        if self.cfg.terrain_name == "cur_stairs_down":
            self.cfg.forward_vel_range = (0.1, 0.2)
        elif self.cfg.terrain_name in ("cur_stairs_up", ):
            self.pitch_t = np.deg2rad(45)
        elif self.cfg.terrain_name in ("cur_mixed", ):
            self.cfg.sim.physx = PhysxCfg(
                min_position_iteration_count=32,
                max_velocity_iteration_count=0,
            )
        elif self.cfg.terrain_name == "exp_stair33_up":
            self.pitch_t = np.deg2rad(40)
        elif self.cfg.terrain_name.startswith("exp_"):
            self.cfg.forward_vel_range = (0.2, 0.2)

        super().__init__(cfg, render_mode, **kwargs)

    def _get_observations(self) -> VecEnvObs:
        height_map = self.calc_scanned_height_maps()
        hmap_shape = height_map.shape
        hmap_mean = einops.reduce(height_map, "n h w -> n", reduction="mean")
        hmap_mean = einops.repeat(hmap_mean, "n -> n h w", h=hmap_shape[1], w=hmap_shape[2])

        obs = torch.cat([
            (height_map - hmap_mean).view(self.num_envs, -1),
            add_noise(self.orientations_3[:, :2] / np.pi, self.orientation_noise_std),
            self.forward_vel_commands[:, 0:1],
            add_noise(self.robot_ang_velocities[:, :], self.angular_vel_noise_std),
            add_noise(self.flipper_positions[:, :], self.flipper_pos_noise_std),
        ], dim=-1)
        return {
            'policy': obs,
        }

    def _calculate_metrics_shutdown(self, i):
        N = 5

        if len(self.history_positions[i]) < N:
            return 0

        trajectory = torch.stack(list(self.history_positions[i])[-N:], dim=0)
        d_list = (trajectory[:, :2] - self.start_positions[i][:2]).norm(dim=1)

        forward_d = torch.diff(d_list)

        if torch.sum(forward_d[forward_d >= 0]) <= 1e-3:
            return -1

        return 0

    def _get_rewards(self) -> torch.Tensor:
        # shutdown reward
        self.reward_buf += 0.2 * torch.tensor(
            [self._calculate_metrics_shutdown(i) for i in range(self.num_envs)]
        ).float()
        # robot height reward
        self.reward_buf += -2 * torch.relu(
            self.positions[:, 2]
            - self.track_wheel_radius
            - self.current_frame_height_maps[:, 15:, :].view(self.num_envs, -1).max(dim=-1)[0]
        )
        # excess reward
        self.reward_buf += -0.5 * (
                ((self.orientations_3[:, 1].abs() > self.pitch_t) * self.orientations_3[:, 1]) ** 2
        )
        self.reward_buf += -0.5 * (
                ((self.orientations_3[:, 0].abs() > 0) * self.orientations_3[:, 0]) ** 2
        )
        # ang reward
        self.reward_buf -= (torch.tensor([0.1, 0.2, 0.1]) * self.robot_ang_velocities ** 2).sum(dim=-1)
        return self.reward_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        super()._get_dones()
        # end in target
        target_idx = ((self.positions[:, :2] - self.target_positions[:, :2]).norm(dim=-1) <= 0.25)
        self.reward_buf[target_idx] += 100

        # end with rollover
        rollover_idx = (torch.any(torch.abs(torch.rad2deg(self.orientations_3[:, :2])) >= 80, dim=-1))
        self.reward_buf[rollover_idx] -= 50
        # out of range
        out_range_idx = out_of_range(self.positions, self.start_positions, self.target_positions)
        # timeout
        timeout_idx = (self.episode_length_buf >= self.max_episode_length)

        self.reset_terminated += target_idx + rollover_idx + out_range_idx
        self.reset_time_outs += timeout_idx
        return self.reset_terminated[:], self.reset_time_outs[:]

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions[:] = actions
        # delta
        # flipper_delta = torch.where(actions >= 0.5, 1, torch.where(actions <= -0.5, -1, 0)) * self.flipper_dt
        # flipper_delta = torch.zeros_like(actions)
        flipper_delta = actions * self.flipper_dt
        self.flipper_target_pos = torch.clip(
            torch.deg2rad(flipper_delta) + self.flipper_positions,
            -np.deg2rad(60), np.deg2rad(60)
        )
