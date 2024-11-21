# -*- coding: utf-8 -*-
"""
====================================
@File Name ：ftr_env.py
@Time ： 2024/9/29 下午12:11
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import os
from functools import cached_property
from itertools import cycle
from typing import Any, Sequence
from collections import deque

import carb
import einops
import numpy as np
import omni.isaac.lab.sim as sim_utils
import torch
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.world import World
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, VecEnvObs, VecEnvStepReturn
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from ftr_envs.assets.articulation.ftr import FtrWheelArticulation
from ftr_envs.assets.ftr import FTR_CFG, FTR_SIM_CFG
from ftr_envs.assets.terrain.terrain import Terrain
from ftr_envs.utils.torch import add_noise, rand_range


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        return data.numpy()

    return np.array(data)


def to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)

    if isinstance(data, torch.Tensor):
        return data

    return torch.tensor(data)


@configclass
class FtrEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 30
    action_scale = 100.0
    num_actions = 1
    num_observations = 4
    num_states = 0

    # simulation
    sim = FTR_SIM_CFG

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=0.0, replicate_physics=True)
    terrain_name = "cur_steps_down"

    # robot
    robot: ArticulationCfg = FTR_CFG
    forward_vel_range = (0.2, 0.3)
    initial_flipper_range = (0, 0)
    robot_config = {
        "sync_flipper_control": False,

        "flipper_material_friction": 10,
        "wheel_material_friction": 10,

        "chassis_wheel_render_mass": 3,
        "flipper_wheel_render_mass": 1,
        "flipper_pos_max": 60,
    }
    robot_render_config = {
        "flipper": {
            "only_render_front_flipper": False,
            "drive_wheel_radius": 0.09,
            "auxiliary_wheel_radius": 0.09,
        },
        "track": {
            "render_radius": 0.1,
        }
    }
    noise = {
        "hmap_noise_std": 0.1,
        "flipper_drive_noise_std": 0.01,
        "baselink_drive_noise_std": 0.01,
        "flipper_pos_noise_std": 0.01,
        "angular_vel_noise_std": 0.2,
        "orientation_noise_std": 0.01,
    }


class FtrEnv(DirectRLEnv):
    cfg: FtrEnvCfg

    def __init__(self, cfg: FtrEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.terrain_cfg = Terrain(cfg.terrain_name)

        self.sync_flipper_control = self.cfg.robot_config["sync_flipper_control"]
        self.only_front_flipper = self.cfg.robot_render_config["flipper"]["only_render_front_flipper"]
        self.flipper_num = 4
        if self.sync_flipper_control:
            self.flipper_num = int(self.flipper_num / 2)
        if self.only_front_flipper:
            self.flipper_num = int(self.flipper_num / 2)
        self.cfg.num_actions = self.flipper_num
        self.cfg.num_observations += (-4 + self.flipper_num)
        self.track_wheel_radius = self.cfg.robot_render_config["track"]["render_radius"]

        super().__init__(cfg, render_mode, **kwargs)
        self.world = World.instance()

        self.hmap_noise_std = self.cfg.noise["hmap_noise_std"]
        self.flipper_drive_noise_std = self.cfg.noise["flipper_drive_noise_std"]
        self.baselink_drive_noise_std = self.cfg.noise["baselink_drive_noise_std"]
        self.orientation_noise_std = self.cfg.noise["orientation_noise_std"]
        self.flipper_pos_noise_std = self.cfg.noise["flipper_pos_noise_std"]
        self.angular_vel_noise_std = self.cfg.noise["angular_vel_noise_std"]

        self.flipper_dt = 5

        self.extractor = torch.nn.AvgPool2d(3)

        self.forward_range = self.cfg.forward_vel_range
        self.initial_flipper_range = self.cfg.initial_flipper_range
        self.forward_vel_commands = torch.zeros(self.num_envs, 1)
        self.flipper_target_pos = torch.zeros(self.num_envs, self.flipper_num)
        self._prepare_reset_info()

        self.start_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.start_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.flipper_positions = torch.zeros((self.num_envs, self.flipper_num), device=self.device)
        self.orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.orientations_3 = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_lin_velocities = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_ang_velocities = torch.zeros((self.num_envs, 3), device=self.device)
        N = 5
        self.history_positions = [deque(maxlen=N) for _ in range(self.num_envs)]

        self.height_map_length = (2.25, 1.05)
        self.height_map_size = (45, 21)
        self.current_frame_height_maps = torch.zeros((self.num_envs, *self.height_map_size), device=self.device)

    def _apply_action(self):
        real_forward_vel_cmd = add_noise(torch.cat(
            [self.forward_vel_commands, torch.zeros(self.num_envs, 1)], dim=-1
        ), std=self.baselink_drive_noise_std)
        real_flipper_cmd = add_noise(
            self._calc_comp_flipper_pos(self.flipper_target_pos),
            std=self.flipper_pos_noise_std
        )

        self._robot.set_v_w(real_forward_vel_cmd)
        self._robot.set_all_flipper_position_targets(
            real_flipper_cmd,
            clip_value=np.deg2rad(self.cfg.robot_config["flipper_pos_max"])
        )

    def _setup_scene(self):
        self._robot = FtrWheelArticulation(self.cfg.robot, device=self.device)
        self._robot.set_robot_env(self.cfg.robot_config, self.cfg.robot_render_config)
        self._robot.load_all_wheel_radius()
        self.scene.articulations["robot"] = self._robot

        stage = self.scene.stage
        self.terrain_cfg.apply(stage)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.terrain_cfg.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._robot.write_root_state_to_sim(torch.zeros(len(env_ids), 13), env_ids=env_ids)

        reset_infos = [self._reset_info_generate() for _ in range(len(env_ids))]
        self._robot.write_root_pose_to_sim(torch.stack([i["pose"] for i in reset_infos]), env_ids=env_ids)
        self.flipper_positions[env_ids, :] = torch.deg2rad(rand_range(
            self.initial_flipper_range,
            (len(env_ids), self.flipper_num),
            device=self.device
        ))
        self._robot.set_all_flipper_positions(self._calc_comp_flipper_pos(self.flipper_positions))
        self.forward_vel_commands[env_ids] = rand_range(self.forward_range, (len(env_ids), 1), device=self.device)

        self.start_positions[env_ids] = torch.stack([i["start_point"] for i in reset_infos])
        self.orientations[env_ids] = torch.stack([i["start_orient"] for i in reset_infos])
        self.target_positions[env_ids] = torch.stack([i["target_point"] for i in reset_infos])

        # clear history data
        for i in env_ids:
            self.history_positions[i].clear()

    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    def _post_physics_step(self):
        self.positions[:] = self._robot.data.root_pos_w
        self.orientations[:] = self._robot.data.root_quat_w
        self.robot_lin_velocities[:] = self._robot.data.root_lin_vel_b
        self.robot_ang_velocities[:] = self._robot.data.root_ang_vel_b
        self.orientations_3[:] = torch.stack(
            list(torch.from_numpy(quat_to_euler_angles(i)).to(self.device) for i in self.orientations.cpu())
        )
        self.flipper_positions[:] = self.get_flipper_pos()
        self.calc_current_frame_height_maps()

        # update history data
        for i in range(self.num_envs):
            self.history_positions[i].append(self.positions[i].clone())

    def get_flipper_pos(self):
        flipper_pos = self._robot.get_all_flipper_positions()
        if self.sync_flipper_control and self.only_front_flipper:
            flipper_pos = flipper_pos[:, [0]]
        elif self.sync_flipper_control and not self.only_front_flipper:
            flipper_pos = flipper_pos[:, [0, 2]]
        elif not self.sync_flipper_control and self.only_front_flipper:
            flipper_pos = flipper_pos[:, [0, 1]]

        return flipper_pos

    def _calc_comp_flipper_pos(self, flipper_pos):
        if self.sync_flipper_control and self.only_front_flipper:
            comp_flipper_pos = torch.cat([
                torch.repeat_interleave(flipper_pos, 2, dim=-1),
                torch.ones(self.num_envs, 2) * np.deg2rad(120)
            ], dim=-1)
        elif self.sync_flipper_control and not self.only_front_flipper:
            comp_flipper_pos = torch.repeat_interleave(flipper_pos, 2, dim=-1)
        elif not self.sync_flipper_control and self.only_front_flipper:
            comp_flipper_pos = torch.cat([
                flipper_pos,
                torch.ones(self.num_envs, 2) * np.deg2rad(120)
            ], dim=-1)
        else:
            comp_flipper_pos = flipper_pos
        return comp_flipper_pos

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._post_physics_step()
        self.reset_terminated = torch.zeros_like(self.reset_terminated)
        self.reset_time_outs = torch.zeros_like(self.reset_time_outs)
        self.reward_buf = torch.zeros(self.num_envs)

        # subclass imp
        ...

        return self.reset_terminated[:], self.reset_time_outs[:]

    def _prepare_reset_info(self):
        self._reset_info = self.terrain_cfg.birth

        # 对数据进行格式统一化
        for info in self._reset_info:
            if len(info["start_orient"]) == 3:
                info["start_orient"] = euler_angles_to_quat(to_numpy(info["start_orient"]))

            for key, value in info.items():
                info[key] = to_tensor(value).float()

            info['pose'] = torch.cat([info['start_point'], info['start_orient']])
        _data = cycle(self._reset_info)
        self._reset_info_generate = lambda: next(_data)

    def calc_current_frame_height_maps(self):
        lower = self.terrain_cfg.map.lower
        upper = self.terrain_cfg.map.upper
        for i in range(self.num_envs):
            pos = self.positions[i].cpu()
            if not (lower[0] < pos[0] < upper[0]) or not (lower[1] < pos[1] < upper[1]):
                carb.log_error(f"The position of the robot seems to be abnormal. {pos=}")
                continue

            angle = torch.rad2deg(self.orientations_3[i]).cpu().numpy()[2]
            local_map = self.terrain_cfg.map.get_obs(pos, angle, self.height_map_length)
            if local_map is None:
                continue

            if local_map.shape != self.height_map_size:
                carb.log_error("Your map doesn't seem big enough.")
                continue

            local_map = torch.from_numpy(local_map).to(self.device).clone()
            self.current_frame_height_maps[i, :, :] = local_map

    def calc_scanned_height_maps(self, base_robot_frame=True):
        height_maps = -torch.ones((self.num_envs, 15, 7), device=self.device)
        ext_map = self.extractor(torch.reshape(self.current_frame_height_maps, (-1, 1, *self.height_map_size)))
        if base_robot_frame:
            ext_map -= einops.repeat(self.positions[:, 2] - self.track_wheel_radius, 'n -> n c w h', c=1, w=15, h=7)
        height_maps[:, :, :] = ext_map.view(height_maps.shape)
        return add_noise(height_maps, std=self.hmap_noise_std)

    @cached_property
    def max_episode_length(self):
        return int(self.cfg.episode_length_s / (self.physics_dt * self.cfg.decimation))

    @property
    def current_time(self):
        return self.world.current_time
