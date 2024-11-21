# -*- coding: utf-8 -*-
"""
====================================
@File Name ：push_cube_env.py
@Time ： 2024/10/15 下午2:12
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import math
from collections.abc import Sequence

import numpy as np
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
import torch
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
)
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from omni.isaac.lab.utils import configclass

from ftr_envs.assets.articulation.ftr import FtrWheelArticulation
from ftr_envs.assets.ftr import FTR_CFG

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(3.0, 3.0),
    border_width=1.0,
    num_rows=8,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.03), noise_step=0.01, border_width=0.0
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.03), platform_width=0.2
        ),
    },
)


@configclass
class TransCargoEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 25.0
    possible_agents = ["left_robot", "center_robot", "right_robot"]
    num_actions = {"left_robot": 6, "center_robot": 6, "right_robot": 6}
    num_observations = {"left_robot": 18, "center_robot": 18, "right_robot": 18}
    num_states = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # robot
    left_robot: ArticulationCfg = FTR_CFG.replace(prim_path="/World/envs/env_.*/left_robot")
    center_robot: ArticulationCfg = FTR_CFG.replace(prim_path="/World/envs/env_.*/center_robot")
    right_robot: ArticulationCfg = FTR_CFG.replace(prim_path="/World/envs/env_.*/right_robot")
    robot_config = {
        "sync_flipper_control": False,

        "flipper_material_friction": 1,
        "wheel_material_friction": 1,

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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    # terrain_name = "ground"
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain0",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    cube_obj = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(1, 4, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )


class TransCargoEnv(DirectMARLEnv):
    cfg: TransCargoEnvCfg

    def __init__(self, cfg: TransCargoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.last_cube_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_scene(self):
        self.left_robot = FtrWheelArticulation(self.cfg.left_robot, self.device)
        self.center_robot = FtrWheelArticulation(self.cfg.center_robot, self.device)
        self.right_robot = FtrWheelArticulation(self.cfg.right_robot, self.device)

        self.left_robot.set_robot_env(self.cfg.robot_config, self.cfg.robot_render_config)
        self.center_robot.set_robot_env(self.cfg.robot_config, self.cfg.robot_render_config)
        self.right_robot.set_robot_env(self.cfg.robot_config, self.cfg.robot_render_config)

        self.left_robot.load_all_wheel_radius()
        self.center_robot.load_all_wheel_radius()
        self.right_robot.load_all_wheel_radius()

        self.scene.articulations["left_robot"] = self.left_robot
        self.scene.articulations["center_robot"] = self.center_robot
        self.scene.articulations["right_robot"] = self.right_robot

        self.cube_obj = RigidObject(cfg=self.cfg.cube_obj)
        self.scene.rigid_objects["cube"] = self.cube_obj

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        def action_to_wheel_vel(action):
            action = action.clamp(-1, 1)
            return (action - 0.5) * 2 * 0.2

        self.left_robot.set_right_and_left_velocities(action_to_wheel_vel(self.actions["left_robot"]))
        self.center_robot.set_right_and_left_velocities(action_to_wheel_vel(self.actions["center_robot"]))
        self.right_robot.set_right_and_left_velocities(action_to_wheel_vel(self.actions["right_robot"]))

        left_flipper_pos = self.left_robot.get_all_flipper_positions(degree=True) + self.actions["left_robot"][:,
                                                                                    2:] * 4
        center_flipper_pos = self.center_robot.get_all_flipper_positions(degree=True) + self.actions["center_robot"][:,
                                                                                        2:] * 4
        right_flipper_pos = self.right_robot.get_all_flipper_positions(degree=True) + self.actions["right_robot"][:,
                                                                                      2:] * 4
        self.left_robot.set_all_flipper_position_targets(left_flipper_pos, degree=True, clip_value=120)
        self.center_robot.set_all_flipper_position_targets(center_flipper_pos, degree=True, clip_value=120)
        self.right_robot.set_all_flipper_position_targets(right_flipper_pos, degree=True, clip_value=120)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {
            "left_robot": torch.cat([
                self.left_robot.positions - self.cube_obj.data.root_pos_w,
                self.left_robot.orientations,
                self.left_robot.lin_velocities,
                self.left_robot.get_all_flipper_positions(),
                self.cube_obj.data.root_quat_w,
            ], dim=-1),
            "center_robot": torch.cat([
                self.center_robot.positions - self.cube_obj.data.root_pos_w,
                self.center_robot.orientations,
                self.center_robot.lin_velocities,
                self.center_robot.get_all_flipper_positions(),
                self.cube_obj.data.root_quat_w,
            ], dim=-1),
            "right_robot": torch.cat([
                self.right_robot.positions - self.cube_obj.data.root_pos_w,
                self.right_robot.orientations,
                self.right_robot.lin_velocities,
                self.right_robot.get_all_flipper_positions(),
                self.cube_obj.data.root_quat_w,
            ], dim=-1),

        }

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        cube_pos = self.cube_obj.data.root_pos_w
        forward_dis = self.last_cube_pos[:, 0] - cube_pos[:, 0]
        self.last_cube_pos = cube_pos.clone()
        reward = forward_dis
        return {
            "left_robot": reward,
            "center_robot": reward,
            "right_robot": reward,
        }

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        terminated[self.cube_obj.data.root_pos_w[:, 2] < 0.2] += 1
        terminated[self.left_robot.positions[:, 2] < -1] += 1
        terminated[self.center_robot.positions[:, 2] < -1] += 1
        terminated[self.right_robot.positions[:, 2] < -1] += 1

        time_out = (self.episode_length_buf >= self.max_episode_length)
        return {
            "left_robot": terminated,
            "center_robot": terminated,
            "right_robot": terminated,
        }, {
            "left_robot": time_out,
            "center_robot": time_out,
            "right_robot": time_out,
        }

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self.default_cube_pos = torch.tensor([10., 0, 2.0], device=self.device)
        self.default_cube_quat = torch.tensor([1, 0, 0, 0], device=self.device)

        self.left_robot.write_root_state_to_sim(torch.zeros(len(env_ids), 13, device=self.device), env_ids=env_ids)
        self.right_robot.write_root_state_to_sim(torch.zeros(len(env_ids), 13, device=self.device), env_ids=env_ids)
        self.center_robot.write_root_state_to_sim(torch.zeros(len(env_ids), 13, device=self.device), env_ids=env_ids)

        self.left_robot.write_root_pose_to_sim(torch.tensor([10, -1.0, 0.7, 0, 0, 0, 1], device=self.device),
                                               env_ids=env_ids)
        self.center_robot.write_root_pose_to_sim(torch.tensor([10, 0, 0.7, 0, 0, 0, 1], device=self.device),
                                                 env_ids=env_ids)
        self.right_robot.write_root_pose_to_sim(torch.tensor([10, 1.0, 0.7, 0, 0, 0, 1], device=self.device),
                                                env_ids=env_ids)

        self.cube_obj.write_root_pose_to_sim(torch.cat([self.default_cube_pos, self.default_cube_quat]), env_ids=env_ids)
        self.cube_obj.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)

        self.last_cube_pos[env_ids] = self.default_cube_pos
