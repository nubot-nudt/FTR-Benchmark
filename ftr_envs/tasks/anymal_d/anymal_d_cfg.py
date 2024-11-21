# -*- coding: utf-8 -*-
"""
====================================
@File Name ：anymal_d_cfg.py
@Time ： 2024/11/5 下午4:34
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""


import math
from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains import TerrainImporterCfg

from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip
from omni.isaac.lab.terrains.config import rough
import omni.isaac.lab.sim as sim_utils

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

    },
)
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.3), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(math.pi, math.pi)
        ),
    )

@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-3.14, -3.14)},
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        )
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
