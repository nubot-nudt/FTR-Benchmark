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
from typing import Dict

import einops
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
from ftr_envs.assets.ftr import FTR_CFG, FTR_SIM_CFG

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(3.0, 3.0),
    border_width=1.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.15),
            step_width=0.2,
            platform_width=1.8,
            border_width=0.5,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.15),
            step_width=0.2,
            platform_width=1.8,
            border_width=0.5,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.0
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.4, grid_width=0.45, grid_height_range=(0.01, 0.1), platform_width=0.2
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=1.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=1.0, border_width=0.25
        ),
    },
)


@configclass
class PreyEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 60.0
    possible_agents = ["food", "predator1", "predator2", "predator3"]
    num_actions = {"food": 6, "predator1": 6, "predator2": 6, "predator3": 6}
    num_observations = {"food": 28, "predator1": 28, "predator2": 28, "predator3": 28}
    num_states = -1

    # simulation
    sim: SimulationCfg = FTR_SIM_CFG

    # robot
    food_robot: ArticulationCfg = FTR_CFG.replace(prim_path="/World/envs/env_.*/food_robot")
    predator1_robot: ArticulationCfg = FTR_CFG.replace(prim_path="/World/envs/env_.*/predator1_robot")
    predator2_robot: ArticulationCfg = FTR_CFG.replace(prim_path="/World/envs/env_.*/predator2_robot")
    predator3_robot: ArticulationCfg = FTR_CFG.replace(prim_path="/World/envs/env_.*/predator3_robot")
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


class PreyEnv(DirectMARLEnv):
    cfg: PreyEnvCfg

    def __init__(self, cfg: PreyEnvCfg, render_mode: str | None = None, **kwargs):
        self.robots: Dict[str, FtrWheelArticulation] = dict()
        self.predator_keys = ["predator1", "predator2", "predator3"]
        super().__init__(cfg, render_mode, **kwargs)
        self.rewards = {
            name: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            for name in self.cfg.possible_agents
        }
        self.pos_dict = {name: (robot.positions - self.robots["food"].positions) for name, robot in self.robots.items()}

    def _setup_scene(self):
        self.robots["food"] = FtrWheelArticulation(self.cfg.food_robot, self.device)
        self.robots["predator1"] = FtrWheelArticulation(self.cfg.predator1_robot, self.device)
        self.robots["predator2"] = FtrWheelArticulation(self.cfg.predator2_robot, self.device)
        self.robots["predator3"] = FtrWheelArticulation(self.cfg.predator3_robot, self.device)

        self.robots["food"].set_robot_color((0.213, 0.68, 0.18))

        for name, robot in self.robots.items():
            robot.set_robot_env(self.cfg.robot_config, self.cfg.robot_render_config)
            robot.load_all_wheel_radius()
            self.scene.articulations[name] = robot

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
        def action_to_wheel_vel(action, max_v):
            action = action.clamp(-1, 1)
            return (action - 0.5) * 2 * max_v

        self.robots["food"].set_right_and_left_velocities(action_to_wheel_vel(self.actions["food"], 0.5))
        for name in self.predator_keys:
            self.robots[name].set_right_and_left_velocities(action_to_wheel_vel(self.actions[name], 0.2))

        for name, robot in self.robots.items():
            flipper_pos = robot.get_all_flipper_positions(degree=True)
            pos_cmd = flipper_pos + self.actions[name][:, 2:] * 4
            # pos_cmd = flipper_pos + 1
            robot.set_all_flipper_position_targets(pos_cmd, degree=True, clip_value=60)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs_dict = {}
        for name, robot in self.robots.items():
            obs_dict[name] = torch.cat([
                self.robots[name].positions,
                self.robots[name].projected_gravity,
                self.robots[name].lin_velocities,
                self.robots[name].ang_velocities,
                self.robots[name].get_all_flipper_positions(),
                *[pos for pos in self.pos_dict.values()]
            ], dim=-1)

        return obs_dict

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        for name in self.predator_keys:
            self.rewards[name] += 0.1 * (2 - self.dis_dict[name])
            self.rewards["food"] += 0.5
        return self.rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.pos_dict = {name: (robot.positions - self.robots["food"].positions) for name, robot in self.robots.items()}
        self.dis_dict = {name: torch.norm(self.pos_dict[name], dim=-1) for name in self.predator_keys}

        for name, robot in self.robots.items():
            self.rewards[name][:] = 0
            # failed
            failed_idx = (robot.positions[:, 2] < -1)
            self.rewards[name][failed_idx] -= 200
            terminated[failed_idx] = True

        for name in self.predator_keys:
            pred_idx = (self.dis_dict[name] < 1)
            terminated[pred_idx] = True
            self.rewards[name][pred_idx] += 1000
            self.rewards["food"][pred_idx] -= 100

        time_out = (self.episode_length_buf >= self.max_episode_length)
        return {name: terminated for name in self.cfg.possible_agents}, {name: time_out for name in self.cfg.possible_agents}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        for robot in self.robots.values():
            robot.write_root_state_to_sim(torch.zeros(len(env_ids), 13, device=self.device), env_ids=env_ids)
            robot.set_all_flipper_positions(torch.ones(len(env_ids), 4, device=self.device, dtype=torch.float32) * 90,
                                            degree=True,
                                            indices=env_ids)

        self.robots["food"].write_root_pose_to_sim(torch.tensor([0, 0.0, 0.5, 0, 0, 0, 0.6], device=self.device),
                                                   env_ids=env_ids)

        for name, y in zip(self.predator_keys, np.linspace(-4, 4, len(self.predator_keys))):
            self.robots[name].write_root_pose_to_sim(
                torch.tensor([6, y, 0.5, 0, 0, 0, 0.6], device=self.device, dtype=torch.float32), env_ids=env_ids)
