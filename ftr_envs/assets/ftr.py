# -*- coding: utf-8 -*-
"""
====================================
@File Name ：ftr.py
@Time ： 2024/9/29 下午2:13
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
from pathlib import Path

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg


FTR_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/pumbaa_wheel",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).parent / "usd" / "ftr" / "ftr_v1.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "baselink_wheel": ImplicitActuatorCfg(
            joint_names_expr=[
                *[f"L{i+1}RevoluteJoint" for i in range(8)],
                *[f"R{i+1}RevoluteJoint" for i in range(8)],
            ],
            stiffness=1,
            damping=100,
        ),
        "flipper_wheel": ImplicitActuatorCfg(
            joint_names_expr=[
                *[f"LF{i+1}RevoluteJoint" for i in range(5)],
                *[f"LR{i+1}RevoluteJoint" for i in range(5)],
                *[f"RL{i+1}RevoluteJoint" for i in range(5)],
                *[f"RR{i+1}RevoluteJoint" for i in range(5)],
            ],
            stiffness=1,
            damping=100,
        ),
        "flipper_joint": ImplicitActuatorCfg(
            joint_names_expr=[
                "front_left_flipper_joint",
                "front_right_flipper_joint",
                "rear_left_flipper_joint",
                "rear_right_flipper_joint",
            ],
            stiffness=3e4,
            damping=1000,
            effort_limit=1000,
            velocity_limit=180,
            armature=100,

            # stiffness=1000,
            # damping=0,
        ),

    },
)

FTR_SIM_CFG = SimulationCfg(
    dt=1 / 100,
    render_interval=5,
    disable_contact_processing=False,
    physx=PhysxCfg(
        min_position_iteration_count=32,
        max_velocity_iteration_count=0,
    ),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=5.0,
        dynamic_friction=5.0,
        restitution=0.0,
    ),

)